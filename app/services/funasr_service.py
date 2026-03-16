"""
FunASR语音识别服务
支持两种模式：
1. HTTP 模式（推荐）：调用独立的 FunASR 服务
2. 本地模式：直接加载模型（需要安装 funasr）
"""
import time
import uuid
import requests
from typing import Dict, Any
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import ASRServiceException

class FunASRService:
    """FunASR语音识别服务类"""
    
    def __init__(self):
        """初始化服务"""
        # 检查是否配置了 FunASR 服务 URL
        self.service_url = getattr(settings, "FUNASR_SERVICE_URL", None)
        
        if self.service_url:
            # HTTP 模式
            self.mode = "http"
            logger.info(f"🌐 FunASR 服务模式: HTTP ({self.service_url})")
            self._check_service_health()
        else:
            # 本地模式（需要安装 funasr）
            self.mode = "local"
            logger.info("💻 FunASR 服务模式: 本地加载")
            self._init_local_model()
    
    def _check_service_health(self):
        """检查远程服务健康状态"""
        try:
            health_url = f"{self.service_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ FunASR 服务连接成功: {data.get('device', 'unknown')}")
            else:
                logger.warning(f"⚠️ FunASR 服务响应异常: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ FunASR 服务连接失败: {e}")
            raise ASRServiceException(f"无法连接到 FunASR 服务 ({self.service_url}): {str(e)}")
    
    def _init_local_model(self):
        """初始化本地模型（如果需要）"""
        try:
            from funasr import AutoModel
            logger.info("🚀 正在加载本地 FunASR 模型...")
            
            self.model = AutoModel(
                model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                model_revision="v2.0.4",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
                device=getattr(settings, "FUNASR_DEVICE", "cuda"),
                ncpu=getattr(settings, "FUNASR_NCPU", 4),
                disable_update=True,
                disable_log=True
            )
            logger.info("✅ 本地 FunASR 模型加载成功")
        except ImportError:
            raise ASRServiceException(
                "FunASR 未安装。请选择：\n"
                "1. 安装 funasr: pip install funasr modelscope torch\n"
                "2. 或配置 FUNASR_SERVICE_URL 使用独立服务"
            )
        except Exception as e:
            raise ASRServiceException(f"FunASR 本地模型加载失败: {str(e)}")
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """
        执行语音识别
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            {
                "text": "完整文本",
                "transcript": [
                    {
                        "text": "句子",
                        "start_time": 0.0,
                        "end_time": 1.5,
                        "speaker_id": "1"
                    }
                ]
            }
        """
        if self.mode == "http":
            return self._transcribe_http(file_path)
        else:
            return self._transcribe_local(file_path)
    
    def _transcribe_http(self, file_path: str) -> Dict[str, Any]:
        """通过 HTTP 调用独立服务"""
        try:
            logger.info(f"🎤 [HTTP模式] 开始识别: {file_path}")
            start_time = time.time()

            # 发送请求到独立服务
            url = f"{self.service_url}/transcribe"

            # 注意：热词现在由FunASR服务自动管理，无需在这里传递

            # 1) URL 模式：如果 file_path 是 http(s)，走 audio_url 分支（不做本地存在性检查）
            if str(file_path).startswith(("http://", "https://")):
                # 检查是否配置了 Pyannote 服务
                try:
                    from app.services.pyannote_service import get_pyannote_service
                    pyannote_service = get_pyannote_service()
                    enable_diarization = not pyannote_service.is_available()
                except Exception:
                    enable_diarization = True

                data = {
                    "audio_url": file_path,
                    "enable_punc": True,
                    "enable_vad": True,
                    "enable_speaker_diarization": enable_diarization,
                }
                response = requests.post(
                    url,
                    data=data,
                    timeout=getattr(settings, "ASR_TIMEOUT", 600)
                )
            else:
                # 2) 本地文件模式：检查文件是否存在，并以上传文件形式传给独立服务
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    raise ASRServiceException(f"音频文件不存在: {file_path}")

                with open(file_path_obj, "rb") as f:
                    files = {"file": (file_path_obj.name, f, "audio/mpeg")}
                    # 检查是否配置了 Pyannote 服务
                    try:
                        from app.services.pyannote_service import get_pyannote_service
                        pyannote_service = get_pyannote_service()
                        enable_diarization = not pyannote_service.is_available()
                    except Exception:
                        # 如果导入失败，默认启用说话人分离
                        enable_diarization = True

                    data = {
                        "enable_punc": True,
                        "enable_vad": True,
                        # 如果配置了 Pyannote 服务，禁用 FunASR 内部的说话人分离
                        "enable_speaker_diarization": enable_diarization
                    }

                    response = requests.post(
                        url,
                        files=files,
                        data=data,
                        timeout=getattr(settings, "ASR_TIMEOUT", 600)
                    )
            
            if response.status_code != 200:
                raise ASRServiceException(f"FunASR 服务返回错误: {response.status_code} - {response.text}")
            
            response_data = response.json()
            elapsed = time.time() - start_time
            
            # FunASR独立服务返回格式: {"code": 0, "msg": "success", "data": {"text": "...", "transcript": [...]}}
            # 需要提取 data 部分
            if "data" in response_data:
                result = response_data["data"]
            else:
                # 兼容直接返回 text 和 transcript 的格式
                result = response_data
            
            # 提取流水号（如果三方服务返回了task_id或request_id）
            task_id = None
            if "task_id" in response_data:
                task_id = str(response_data["task_id"])
            elif "request_id" in response_data:
                task_id = str(response_data["request_id"])
            elif "data" in response_data and isinstance(response_data["data"], dict):
                if "task_id" in response_data["data"]:
                    task_id = str(response_data["data"]["task_id"])
                elif "request_id" in response_data["data"]:
                    task_id = str(response_data["data"]["request_id"])
            
            # 如果三方服务没有返回流水号，生成一个唯一标识
            if not task_id:
                task_id = str(uuid.uuid4())
                logger.info(f"📝 本地FunASR生成唯一标识: {task_id}")
            else:
                logger.info(f"📝 FunASR服务返回流水号: {task_id}")
            
            # 将流水号添加到结果中
            if isinstance(result, dict):
                result["task_id"] = task_id
            else:
                result = {"text": result.get("text", ""), "transcript": result.get("transcript", []), "task_id": task_id}
            
            text_length = len(result.get('text', ''))
            logger.info(f"✅ [HTTP模式] 识别完成 | 耗时:{elapsed:.2f}s | 字数:{text_length} | 流水号:{task_id}")
            
            return result
            
        except requests.exceptions.Timeout:
            raise ASRServiceException("FunASR 服务请求超时")
        except requests.exceptions.ConnectionError as e:
            raise ASRServiceException(f"无法连接到 FunASR 服务: {str(e)}")
        except Exception as e:
            logger.error(f"❌ [HTTP模式] 识别失败: {e}")
            raise ASRServiceException(f"识别失败: {str(e)}")
    
    def _transcribe_local(self, file_path: str) -> Dict[str, Any]:
        """本地模型识别"""
        try:
            logger.info(f"🎤 [本地模式] 开始识别: {file_path}")
            start_time = time.time()
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise ASRServiceException(f"音频文件不存在: {file_path}")
            
            # 调用本地模型（如果需要热词，应该在funasr_standalone服务中配置）
            res = self.model.generate(
                input=str(file_path_obj),
                batch_size_s=300,
                sentence_timestamp=True,
                vad_kwargs={
                    "speech_noise_thres": 0.3,
                    "max_single_segment_time": 60000,
                    "vad_tol": 300
                }
            )
            
            elapsed = time.time() - start_time
            
            # 处理结果
            transcript_data = []
            full_text = ""
            
            if res:
                raw_sentences = res[0].get("sentence_info", [])
                
                for s in raw_sentences:
                    text = s.get('text', '').strip()
                    if text in ["", "，", "。", "？"]:
                        continue
                    
                    full_text += text
                    
                    if 'timestamp' in s and len(s['timestamp']) > 0:
                        start_ms = s['timestamp'][0][0]
                        end_ms = s['timestamp'][-1][1]
                        
                        transcript_data.append({
                            "text": text,
                            "start_time": round(start_ms / 1000.0, 2),
                            "end_time": round(end_ms / 1000.0, 2),
                            "speaker_id": str(s.get('spk', '1'))
                        })
            
            # 本地模式生成唯一标识
            task_id = str(uuid.uuid4())
            logger.info(f"📝 本地FunASR生成唯一标识: {task_id}")
            logger.info(f"✅ [本地模式] 识别完成 | 耗时:{elapsed:.2f}s | 字数:{len(full_text)} | 流水号:{task_id}")
            
            return {
                "text": full_text,
                "transcript": transcript_data,
                "task_id": task_id
            }
            
        except Exception as e:
            logger.error(f"❌ [本地模式] 识别失败: {e}")
            raise ASRServiceException(f"识别失败: {str(e)}")


# 单例获取方法
_funasr_service_instance = None

def get_funasr_service():
    """获取 FunASR 服务实例（单例）"""
    global _funasr_service_instance
    if _funasr_service_instance is None:
        _funasr_service_instance = FunASRService()
    return _funasr_service_instance
