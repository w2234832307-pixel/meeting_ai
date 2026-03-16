"""
GPU池管理器：自动检测GPU数量，实现任务分配和负载均衡
支持多GPU并行处理，提高系统吞吐量
"""
import logging
import torch
import asyncio
from typing import List, Optional, Dict, Any, AsyncContextManager
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU信息"""
    device_id: int
    device_name: str
    total_memory: float  # GB
    available_memory: float  # GB
    is_available: bool


class GPUPool:
    """
    GPU池管理器
    
    功能：
    1. 自动检测可用GPU数量
    2. 轮询/负载均衡分配任务到不同GPU
    3. 管理GPU使用状态和并发控制
    """
    
    def __init__(self, max_concurrent_per_gpu: int = 1):
        """
        初始化GPU池
        
        Args:
            max_concurrent_per_gpu: 每张GPU的最大并发任务数（默认1，避免显存溢出）
        """
        self.max_concurrent_per_gpu = max_concurrent_per_gpu
        self.gpu_list: List[GPUInfo] = []
        self.gpu_semaphores: Dict[int, asyncio.Semaphore] = {}
        self.gpu_usage_counter: Dict[int, int] = {}  # 每张GPU的使用计数（用于轮询）
        self._lock = asyncio.Lock()
        
        # 初始化GPU池
        self._initialize_gpu_pool()
    
    def _initialize_gpu_pool(self):
        """初始化GPU池：检测所有可用GPU"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ 未检测到CUDA，GPU池将为空（使用CPU模式）")
            self.gpu_list = []
            return
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"🔍 检测到 {gpu_count} 张GPU，开始初始化GPU池...")
        
        for i in range(gpu_count):
            try:
                # 获取GPU信息
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                
                # 尝试分配一个小tensor来测试GPU是否可用
                try:
                    test_tensor = torch.zeros(1, device=f"cuda:{i}")
                    del test_tensor
                    torch.cuda.empty_cache()
                    is_available = True
                except Exception as e:
                    logger.warning(f"⚠️ GPU {i} ({device_name}) 不可用: {e}")
                    is_available = False
                
                gpu_info = GPUInfo(
                    device_id=i,
                    device_name=device_name,
                    total_memory=total_memory,
                    available_memory=total_memory,  # 初始值，实际使用中可动态更新
                    is_available=is_available
                )
                
                if is_available:
                    self.gpu_list.append(gpu_info)
                    # 为每张GPU创建Semaphore（控制并发）
                    self.gpu_semaphores[i] = asyncio.Semaphore(self.max_concurrent_per_gpu)
                    self.gpu_usage_counter[i] = 0
                    logger.info(
                        f"✅ GPU {i}: {device_name} ({total_memory:.1f}GB) - "
                        f"最大并发: {self.max_concurrent_per_gpu}"
                    )
                else:
                    logger.warning(f"⚠️ 跳过不可用GPU {i}: {device_name}")
                    
            except Exception as e:
                logger.error(f"❌ 初始化GPU {i} 时出错: {e}")
                continue
        
        if not self.gpu_list:
            logger.warning("⚠️ 没有可用的GPU，系统将使用CPU模式")
        else:
            logger.info(f"✅ GPU池初始化完成，共 {len(self.gpu_list)} 张可用GPU")
    
    def get_gpu_count(self) -> int:
        """获取可用GPU数量"""
        return len(self.gpu_list)
    
    def get_all_gpu_devices(self) -> List[str]:
        """获取所有GPU设备字符串列表（如 ['cuda:0', 'cuda:1', ...]）"""
        return [f"cuda:{gpu.device_id}" for gpu in self.gpu_list]
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """获取所有GPU的详细信息（用于监控）"""
        return [
            {
                "device_id": gpu.device_id,
                "device_name": gpu.device_name,
                "total_memory_gb": gpu.total_memory,
                "available_memory_gb": gpu.available_memory,
                "is_available": gpu.is_available,
                "current_usage": self.gpu_usage_counter.get(gpu.device_id, 0)
            }
            for gpu in self.gpu_list
        ]
    
    async def acquire_gpu(self, strategy: str = "round_robin") -> Optional[str]:
        """
        获取一张可用的GPU（带并发控制）
        
        Args:
            strategy: 分配策略
                - "round_robin": 轮询分配（默认，负载均衡）
                - "least_used": 选择使用次数最少的GPU
        
        Returns:
            GPU设备字符串（如 "cuda:0"），如果没有可用GPU则返回None
        """
        if not self.gpu_list:
            return None
        
        async with self._lock:
            if strategy == "round_robin":
                # 轮询：选择使用次数最少的GPU
                selected_gpu = min(self.gpu_list, key=lambda g: self.gpu_usage_counter.get(g.device_id, 0))
            elif strategy == "least_used":
                # 选择使用次数最少的GPU（与round_robin相同，但语义更清晰）
                selected_gpu = min(self.gpu_list, key=lambda g: self.gpu_usage_counter.get(g.device_id, 0))
            else:
                # 默认使用第一张GPU
                selected_gpu = self.gpu_list[0]
            
            device_id = selected_gpu.device_id
            device_str = f"cuda:{device_id}"
            
            # 获取该GPU的Semaphore（会阻塞直到有可用槽位）
            semaphore = self.gpu_semaphores[device_id]
            await semaphore.acquire()
            
            # 更新使用计数
            self.gpu_usage_counter[device_id] = self.gpu_usage_counter.get(device_id, 0) + 1
            
            logger.debug(f"🎯 分配GPU: {device_str} (使用次数: {self.gpu_usage_counter[device_id]})")
            return device_str
    
    async def release_gpu(self, device_str: str):
        """
        释放GPU资源
        
        Args:
            device_str: GPU设备字符串（如 "cuda:0"）
        """
        if not device_str or not device_str.startswith("cuda:"):
            return
        
        try:
            device_id = int(device_str.split(":")[1])
            if device_id in self.gpu_semaphores:
                self.gpu_semaphores[device_id].release()
                logger.debug(f"🔓 释放GPU: {device_str}")
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ 无效的GPU设备字符串: {device_str}, 错误: {e}")
    
    @asynccontextmanager
    async def get_gpu_context(self, strategy: str = "round_robin"):
        """
        获取GPU上下文管理器（推荐使用方式）
        
        用法：
            async with gpu_pool.get_gpu_context() as device:
                if device:
                    # 使用device（如 "cuda:0"）处理任务
                    pass
        """
        device = await self.acquire_gpu(strategy)
        try:
            yield device
        finally:
            if device:
                await self.release_gpu(device)
    
    def get_total_concurrent_capacity(self) -> int:
        """
        获取总并发容量（所有GPU的最大并发任务数之和）
        
        例如：7张GPU，每张最大并发1 = 总容量7
        """
        return len(self.gpu_list) * self.max_concurrent_per_gpu
    
    async def acquire_multiple_gpus(self, count: int, strategy: str = "round_robin") -> List[str]:
        """
        获取多张GPU（用于单个任务的多GPU加速）
        
        Args:
            count: 需要获取的GPU数量
            strategy: 分配策略
        
        Returns:
            GPU设备字符串列表（如 ["cuda:0", "cuda:1", ...]）
        """
        if not self.gpu_list or count <= 0:
            return []
        
        acquired_gpus = []
        async with self._lock:
            # 按策略选择GPU
            available_gpus = sorted(
                self.gpu_list,
                key=lambda g: self.gpu_usage_counter.get(g.device_id, 0)
            )[:count]
            
            for gpu in available_gpus:
                device_id = gpu.device_id
                device_str = f"cuda:{device_id}"
                
                # 获取该GPU的Semaphore
                semaphore = self.gpu_semaphores[device_id]
                await semaphore.acquire()
                
                # 更新使用计数
                self.gpu_usage_counter[device_id] = self.gpu_usage_counter.get(device_id, 0) + 1
                acquired_gpus.append(device_str)
                
                logger.debug(f"🎯 多GPU模式：分配GPU {device_str}")
        
        return acquired_gpus
    
    async def release_multiple_gpus(self, gpu_devices: List[str]):
        """
        释放多张GPU资源
        
        Args:
            gpu_devices: GPU设备字符串列表
        """
        for device_str in gpu_devices:
            await self.release_gpu(device_str)


# 全局GPU池实例（单例模式）
_global_gpu_pool: Optional[GPUPool] = None


def get_gpu_pool(max_concurrent_per_gpu: int = 1) -> GPUPool:
    """
    获取全局GPU池实例（单例模式）
    
    Args:
        max_concurrent_per_gpu: 每张GPU的最大并发任务数（仅在首次创建时生效）
    
    Returns:
        GPUPool实例
    """
    global _global_gpu_pool
    if _global_gpu_pool is None:
        _global_gpu_pool = GPUPool(max_concurrent_per_gpu=max_concurrent_per_gpu)
    return _global_gpu_pool


def reset_gpu_pool():
    """重置全局GPU池（用于测试或重新初始化）"""
    global _global_gpu_pool
    _global_gpu_pool = None
