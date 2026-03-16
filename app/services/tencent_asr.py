"""
腾讯云ASR语音识别服务
支持文件识别，带重试机制、超时控制、错误处理
"""
import json
import re
import time
import os
from typing import Optional, Dict, Any, List
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.asr.v20190614 import asr_client, models
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import ASRServiceException
from app.core.utils import retry_with_backoff, safe_json_parse, truncate_text


class TencentASRService:
    """腾讯云ASR服务类"""
    
    def __init__(self):
        """初始化腾讯云ASR客户端"""
        try:
            # 验证配置
            if not settings.is_tencent_asr_available():
                raise ASRServiceException(
                    "腾讯云ASR配置不完整，请检查 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY"
                )
            
            # 初始化凭证
            cred = credential.Credential(
                settings.TENCENT_SECRET_ID,
                settings.TENCENT_SECRET_KEY
            )
            
            # 实例化HTTP配置
            http_profile = HttpProfile()
            http_profile.endpoint = "asr.tencentcloudapi.com"
            http_profile.reqTimeout = settings.ASR_TIMEOUT
            
            # 实例化客户端配置
            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            client_profile.signMethod = "TC3-HMAC-SHA256"
            
            # 实例化ASR客户端
            self.client = asr_client.AsrClient(cred, settings.TENCENT_REGION, client_profile)
            
            logger.info(f"✅ 腾讯云ASR客户端初始化成功 (区域: {settings.TENCENT_REGION})")
            
        except TencentCloudSDKException as e:
            logger.error(f"❌ 腾讯云ASR客户端初始化失败: {e}")
            raise ASRServiceException(f"腾讯云ASR客户端初始化失败: {str(e)}", {"error": str(e)})
        except Exception as e:
            logger.error(f"❌ 初始化异常: {e}")
            raise ASRServiceException(f"初始化异常: {str(e)}", {"error": str(e)})
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """
        识别音频文件
        """
        try:
            logger.info(f"🎤 开始识别音频文件: {file_path}")
            
            # 1. 处理路径/URL
            if file_path.startswith(("http://", "https://")):
                file_url = file_path
            else:
                if not os.path.exists(file_path):
                    raise ASRServiceException(f"音频文件不存在: {file_path}")
                file_url = file_path
            
            # 2. 调用识别 
            # result 已经是 {'text': '...', 'transcript': [...]} 的格式了
            result = self._call_create_rec_task(file_url)
            
            # 3. 记录日志
            text_len = len(result.get("text", ""))
            transcript_count = len(result.get("transcript", []))
            logger.info(f"✅ 识别流程结束，文本长度: {text_len}，逐字稿段落数: {transcript_count}")
            
            return result
            
        except ASRServiceException:
            raise
        except Exception as e:
            logger.error(f"❌ 识别失败: {e}")
            raise ASRServiceException(f"识别失败: {str(e)}", {"file_path": file_path, "error": str(e)})
    
    @retry_with_backoff(max_attempts=3, initial_wait=1.0, max_wait=5.0)
    def _call_create_rec_task(self, file_url: str) -> Dict[str, Any]:
        """
        调用腾讯云创建录音文件识别任务
        
        Args:
            file_url: 音频文件URL或路径
        
        Returns:
            识别结果字典
        """
        try:
            # 创建请求对象
            req = models.CreateRecTaskRequest()
            
            # 设置引擎模型类型（16k_zh: 中文普通话，16k_en: 英文等）
            req.EngineModelType = settings.TENCENT_ASR_ENGINE_MODEL_TYPE
            
            # 腾讯云ASR的CreateRecTask API只接受URL，不支持本地文件路径
            # 如果是本地路径，需要提示错误
            if file_url.startswith(("http://", "https://")):
                req.Url = file_url
            else:
                # 本地文件路径不被支持，需要先上传到可公网访问的URL
                raise ASRServiceException(
                    "腾讯云录音文件识别API要求音频文件必须是可公网访问的URL。"
                    "本地文件路径不被支持。请将文件上传到云存储（如COS、OSS）后提供URL，"
                    "或使用文件上传方式（file参数）。",
                    {"file_path": file_url}
                )
            
            # 设置识别频道数（1表示单声道，2表示双声道）
            req.ChannelNum = 1

            # 0：识别结果按句展示；1：识别结果按词展示。
            req.ResTextFormat = 0 
    
            # 0：语音 URL；1：语音数据（post body）。
            req.SourceType = 0
            
            # 设置是否开启说话人分离（0关闭，1开启）
            req.SpeakerDiarization = 1
            
            # 设置说话人人数（开启说话人分离时有效，0表示自动识别）
            req.SpeakerNumber = 0
            
            # 设置是否返回词级别时间戳（0关闭，1开启）
            req.FilterDirty = 0
            req.FilterModal = 0
            req.FilterPunc = 0
            req.ConvertNumMode = 1
            
            logger.info(f"📤 提交识别任务: {req.Url[:100]}...")
            
            # 调用API
            resp = self.client.CreateRecTask(req)
            
            # 检查响应
            if not resp or not resp.Data:
                raise ASRServiceException("创建识别任务失败：响应为空")
            
            task_id = resp.Data.TaskId
            
            if not task_id:
                raise ASRServiceException("创建识别任务失败：未返回TaskId")
            
            logger.info(f"✅ 任务已提交，TaskId: {task_id}")
            
            # 轮询等待结果
            return self._poll_task_result(task_id)
            
        except TencentCloudSDKException as e:
            logger.error(f"❌ 腾讯云API调用失败: {e}")
            raise ASRServiceException(f"API调用失败: {e.get_code()} - {e.get_message()}", {
                "code": e.get_code(),
                "message": e.get_message()
            })
    
    def _poll_task_result(self, task_id: int, max_wait_seconds: int = 300) -> Dict[str, Any]:
        """
        [最终版] 轮询任务结果
        """
        start_time = time.time()
        poll_interval = 3
        
        logger.info(f"🔄 开始轮询任务: {task_id}")
        
        while True:
            if time.time() - start_time > max_wait_seconds:
                raise ASRServiceException(f"识别任务超时", {"task_id": task_id})
            
            try:
                req = models.DescribeTaskStatusRequest()
                req.TaskId = task_id
                resp = self.client.DescribeTaskStatus(req)
                
                status = resp.Data.Status
                
                if status == 2:  # 识别完成
                    logger.info(f"✅ 腾讯云识别成功 (TaskId: {task_id})")
                    
                    # 这里的 Result 可能是 JSON 字符串，也可能是带时间戳的普通文本
                    result_data = resp.Data.Result
                    
                    # 调用增强版解析器
                    final_result = self._extract_transcript_from_result(result_data)
                    
                    return {
                        "text": "".join([i['text'] for i in final_result]), 
                        "transcript": final_result,
                        "task_id": str(task_id)  # 返回腾讯云的TaskId作为流水号
                    }
                
                elif status == 3:
                    error_msg = getattr(resp.Data, 'ErrorMsg', '未知错误')
                    raise ASRServiceException(f"识别失败: {error_msg}")
                
                time.sleep(poll_interval)
                
            except TencentCloudSDKException as e:
                logger.error(f"SDK异常: {e}")
                time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"轮询未知异常: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(poll_interval)
    
    import re  # 记得在文件头部导入 re

    def _extract_transcript_from_result(self, result_data: Any) -> List[Dict[str, Any]]:
        """
        [最终版] 提取逐字稿
        兼容：JSON List, JSON String, 以及腾讯云特殊的 [time] text 字符串格式
        """
        transcript = []
        
        try:
            if not result_data:
                return []

            # --- 情况 A: 已经是列表 ---
            if isinstance(result_data, list):
                source_list = result_data
            
            # --- 情况 B: 是字符串，尝试解析 ---
            elif isinstance(result_data, str):
                # 1. 尝试当做 JSON 解析
                try:
                    parsed = json.loads(result_data)
                    if isinstance(parsed, list):
                        source_list = parsed
                    elif isinstance(parsed, dict) and "Result" in parsed:
                        # 递归处理
                        return self._extract_transcript_from_result(parsed["Result"])
                    else:
                        raise ValueError("Not a standard JSON list")
                except Exception:
                    # 2. JSON 解析失败，说明是 "特殊文本格式"
                    # 格式示例: [0:0.040,0:4.220,0]  那个还是按正常的流程...
                    logger.info("⚠️ 识别结果非JSON格式，尝试使用正则解析文本流...")
                    return self._parse_text_stream(result_data)
            else:
                logger.warning(f"未知结果类型: {type(result_data)}")
                return []

            # --- 处理标准的 JSON List ---
            for item in source_list:
                if isinstance(item, dict):
                    transcript.append({
                        "text": item.get("Text", ""),
                        "start_time": float(item.get("StartTime", 0)) / 1000.0,
                        "end_time": float(item.get("EndTime", 0)) / 1000.0,
                        "speaker_id": item.get("SpeakerId")
                    })
            
            return transcript

        except Exception as e:
            logger.error(f"解析结果失败: {e}")
            return []

    def _parse_text_stream(self, text_stream: str) -> List[Dict[str, Any]]:
        """
        解析腾讯云特殊的文本流格式
        格式: [开始分:秒,结束分:秒,声道] 文本
        示例: [0:0.040,0:4.220,0] 文本内容
        """
        results = []
        # 正则表达式匹配一行
        # 匹配: [0:0.040,0:4.220,0] 内容
        pattern = re.compile(r"\[(\d+):(\d+\.\d+),(\d+):(\d+\.\d+),(\d+)\]\s*(.*)")
        
        lines = text_stream.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = pattern.match(line)
            if match:
                s_min, s_sec, e_min, e_sec, channel, content = match.groups()
                
                # 时间转换 (分:秒 -> 秒)
                start_time = int(s_min) * 60 + float(s_sec)
                end_time = int(e_min) * 60 + float(e_sec)
                
                results.append({
                    "text": content,
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "speaker_id": channel
                })
        
        logger.info(f"📝 正则解析完成，提取到 {len(results)} 条记录")
        return results

# 创建单例实例
try:
    asr_service = TencentASRService()
except ASRServiceException as e:
    logger.warning(f"⚠️ ASR服务初始化失败，将在首次使用时重试: {e}")
    asr_service = None

