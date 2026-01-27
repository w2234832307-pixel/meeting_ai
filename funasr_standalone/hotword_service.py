"""
FunASR热词管理服务
支持动态加载、更新热词列表，用于提升ASR识别准确率
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


class HotwordService:
    """热词管理服务"""
    
    def __init__(self, config_path: str = None, auto_reload: bool = True):
        """
        初始化热词服务
        
        Args:
            config_path: 热词配置文件路径（相对于funasr_standalone目录）
            auto_reload: 是否自动检测文件变化并重新加载（默认True）
        """
        if config_path is None:
            # 默认使用 funasr_standalone/hotwords.json
            config_path = Path(__file__).parent / "hotwords.json"
        
        self.config_path = Path(config_path)
        self.hotwords_cache: Dict[str, List[str]] = {}
        self.auto_reload = auto_reload  # 自动重载开关
        self.last_mtime = 0  # 文件最后修改时间
        self._load_hotwords()
    
    def _load_hotwords(self, force: bool = False) -> None:
        """
        从配置文件加载热词
        
        Args:
            force: 是否强制重新加载（不检查文件时间）
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"⚠️ 热词配置文件不存在: {self.config_path}，将创建默认配置")
                self._create_default_config()
                return
            
            # 检查文件修改时间
            current_mtime = self.config_path.stat().st_mtime
            if not force and current_mtime == self.last_mtime:
                # 文件未修改，跳过加载
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 过滤掉"说明"等非热词字段
            self.hotwords_cache = {
                k: v for k, v in data.items() 
                if isinstance(v, list) and k not in ["说明", "description", "备注"]
            }
            
            # 更新文件修改时间
            self.last_mtime = current_mtime
            
            total_count = sum(len(v) for v in self.hotwords_cache.values())
            
            # 只在文件真正变化时打印详细日志
            if force or self.last_mtime != 0:
                logger.info(f"🔄 热词已更新: {len(self.hotwords_cache)} 个类别, 共 {total_count} 个词")
                # 打印各类别数量
                for category, words in self.hotwords_cache.items():
                    logger.info(f"  - {category}: {len(words)} 个")
            else:
                logger.info(f"✅ 成功加载热词配置: {len(self.hotwords_cache)} 个类别, 共 {total_count} 个词")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ 热词配置文件格式错误: {e}")
            self.hotwords_cache = {}
        except Exception as e:
            logger.error(f"❌ 加载热词配置失败: {e}")
            self.hotwords_cache = {}
    
    def _create_default_config(self) -> None:
        """创建默认配置文件"""
        default_config = {
            "人名": ["张三", "李四", "王五"],
            "项目名": ["智能办公", "数据中台"],
            "技术词汇": ["机器学习", "深度学习", "大语言模型"],
            "说明": "这是FunASR服务的热词配置文件，可以随时修改。修改后需要重新加载热词。"
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 已创建默认热词配置: {self.config_path}")
            self.hotwords_cache = {
                k: v for k, v in default_config.items() 
                if isinstance(v, list)
            }
        except Exception as e:
            logger.error(f"❌ 创建默认配置失败: {e}")
    
    def get_all_hotwords(self) -> List[str]:
        """
        获取所有热词（合并所有类别）
        
        Returns:
            热词列表（去重）
        """
        all_words: Set[str] = set()
        for words in self.hotwords_cache.values():
            all_words.update(words)
        return list(all_words)
    
    def get_hotwords_by_category(self, category: str) -> List[str]:
        """
        获取指定类别的热词
        
        Args:
            category: 类别名称（如"人名"、"项目名"）
        
        Returns:
            该类别的热词列表
        """
        return self.hotwords_cache.get(category, [])
    
    def get_hotwords_string(self, separator: str = " ") -> str:
        """
        获取热词字符串（用于传递给ASR模型）
        
        Args:
            separator: 分隔符（默认空格）
        
        Returns:
            热词字符串，如："张三 李四 智能办公 数据中台"
        """
        # 自动检测文件变化并重新加载
        if self.auto_reload:
            self._load_hotwords()
        
        return separator.join(self.get_all_hotwords())
    
    def reload(self) -> bool:
        """
        重新加载热词配置（用于动态更新）
        
        Returns:
            是否成功重载
        """
        try:
            logger.info("🔄 重新加载热词配置...")
            old_count = sum(len(v) for v in self.hotwords_cache.values())
            self._load_hotwords()
            new_count = sum(len(v) for v in self.hotwords_cache.values())
            logger.info(f"✅ 热词重载完成: {old_count} → {new_count} 个词")
            return True
        except Exception as e:
            logger.error(f"❌ 重载热词失败: {e}")
            return False
    
    def add_hotwords(self, category: str, words: List[str]) -> bool:
        """
        添加热词到指定类别
        
        Args:
            category: 类别名称
            words: 要添加的热词列表
        
        Returns:
            是否成功添加
        """
        try:
            if category not in self.hotwords_cache:
                self.hotwords_cache[category] = []
            
            # 去重并添加
            existing = set(self.hotwords_cache[category])
            new_words = [w for w in words if w not in existing]
            
            if new_words:
                self.hotwords_cache[category].extend(new_words)
                self._save_to_file()
                logger.info(f"✅ 已添加 {len(new_words)} 个热词到 [{category}]")
                return True
            else:
                logger.info(f"ℹ️ 所有词已存在于 [{category}]")
                return True
                
        except Exception as e:
            logger.error(f"❌ 添加热词失败: {e}")
            return False
    
    def remove_hotwords(self, category: str, words: List[str]) -> bool:
        """
        从指定类别删除热词
        
        Args:
            category: 类别名称
            words: 要删除的热词列表
        
        Returns:
            是否成功删除
        """
        try:
            if category not in self.hotwords_cache:
                logger.warning(f"⚠️ 类别不存在: {category}")
                return False
            
            # 删除指定词
            original_count = len(self.hotwords_cache[category])
            self.hotwords_cache[category] = [
                w for w in self.hotwords_cache[category] 
                if w not in words
            ]
            removed_count = original_count - len(self.hotwords_cache[category])
            
            if removed_count > 0:
                self._save_to_file()
                logger.info(f"✅ 已从 [{category}] 删除 {removed_count} 个热词")
                return True
            else:
                logger.info(f"ℹ️ 没有找到要删除的热词")
                return False
                
        except Exception as e:
            logger.error(f"❌ 删除热词失败: {e}")
            return False
    
    def _save_to_file(self) -> None:
        """保存热词配置到文件"""
        try:
            # 添加说明字段
            data = dict(self.hotwords_cache)
            data["说明"] = "这是FunASR服务的热词配置文件，可以随时修改。修改后可通过API重新加载。"
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 热词配置已保存到: {self.config_path}")
        except Exception as e:
            logger.error(f"❌ 保存热词配置失败: {e}")
    
    def get_categories(self) -> List[str]:
        """获取所有类别名称"""
        return list(self.hotwords_cache.keys())
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取热词统计信息
        
        Returns:
            各类别的词数统计
        """
        return {
            category: len(words) 
            for category, words in self.hotwords_cache.items()
        }


# 全局单例（在FunASR服务启动时初始化）
hotword_service = None

def get_hotword_service() -> HotwordService:
    """获取热词服务单例"""
    global hotword_service
    if hotword_service is None:
        hotword_service = HotwordService()
    return hotword_service
