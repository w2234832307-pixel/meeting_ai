#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyannote 说话人分离模块
使用专业的 Pyannote.audio 模型进行说话人分离
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("⚠️ Pyannote.audio 未安装，说话人分离功能将不可用")
    logger.warning("   安装命令: pip install pyannote.audio")


# 全局 pipeline 缓存（避免重复加载）
_pipeline_cache = None


def get_pyannote_pipeline(use_auth_token: Optional[str] = None):
    """
    获取 Pyannote pipeline（带缓存）
    
    Args:
        use_auth_token: HuggingFace token
    
    Returns:
        Pipeline 对象，失败返回 None
    """
    global _pipeline_cache
    
    if _pipeline_cache is not None:
        return _pipeline_cache
    
    if not PYANNOTE_AVAILABLE:
        return None
    
    try:
        import os
        from pathlib import Path
        
        hf_token = use_auth_token or os.getenv("HF_TOKEN")
        
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        local_model_path = project_root / "models" / "pyannote_diarization"
        
        pipeline = None
        use_local_model = False
        
        # 检查本地模型
        if local_model_path.exists() and (local_model_path / "config.yaml").exists():
            logger.info(f"✅ 检测到项目本地模型: {local_model_path}")
            
            local_segmentation_path = project_root / "models" / "pyannote_segmentation"
            local_embedding_path = project_root / "models" / "pyannote_wespeaker"
            
            has_local_segmentation = local_segmentation_path.exists() and (local_segmentation_path / "config.yaml").exists()
            has_local_embedding = local_embedding_path.exists() and (local_embedding_path / "config.yaml").exists()
            
            if has_local_segmentation and has_local_embedding:
                try:
                    import yaml
                    import shutil
                    
                    config_file = local_model_path / "config.yaml"
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    if 'pipeline' in config and 'params' in config['pipeline']:
                        config['pipeline']['params']['segmentation'] = str(local_segmentation_path.resolve())
                        config['pipeline']['params']['embedding'] = str(local_embedding_path.resolve())
                    
                    temp_config_file = local_model_path / "config.yaml.local"
                    with open(temp_config_file, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    
                    original_config_file = local_model_path / "config.yaml.original"
                    if not original_config_file.exists():
                        shutil.copy2(config_file, original_config_file)
                    
                    shutil.copy2(temp_config_file, config_file)
                    
                    try:
                        pipeline = Pipeline.from_pretrained(str(local_model_path), local_files_only=True)
                        logger.info("✅ 成功从项目本地路径加载 Pyannote 模型")
                        use_local_model = True
                    finally:
                        if original_config_file.exists():
                            shutil.copy2(original_config_file, config_file)
                        if temp_config_file.exists():
                            temp_config_file.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ 从本地路径加载失败: {e}")
        
        # 如果本地模型加载失败，尝试从 HuggingFace 加载
        if not use_local_model:
            cache_dirs = [
                Path.home() / ".cache" / "pyannote",
                Path.home() / ".cache" / "huggingface" / "hub",
            ]
            
            model_cached = False
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    model_path = cache_dir / "models--pyannote--speaker-diarization-3.1"
                    if model_path.exists():
                        model_cached = True
                        break
            
            try:
                if hf_token:
                    try:
                        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                    except TypeError:
                        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
                else:
                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                logger.info("✅ Pyannote pipeline 加载成功")
            except Exception as load_error:
                error_str = str(load_error).lower()
                if ("network" in error_str or "unreachable" in error_str) and model_cached:
                    try:
                        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", local_files_only=True)
                        logger.info("✅ 使用本地缓存加载 Pyannote pipeline")
                    except:
                        logger.error(f"❌ 无法使用本地缓存: {load_error}")
                        return None
                else:
                    logger.error(f"❌ 加载 Pyannote pipeline 失败: {load_error}")
                    return None
        
        _pipeline_cache = pipeline
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ 加载 Pyannote pipeline 失败: {e}")
        return None


def perform_pyannote_diarization(
    audio_path: str,
    transcript: List[Dict],
    use_auth_token: Optional[str] = None
) -> List[Dict]:
    """
    使用 Pyannote 进行说话人分离
    
    Args:
        audio_path: 音频文件路径
        transcript: ASR识别结果，包含text、start_time、end_time
        use_auth_token: HuggingFace token（如果需要访问私有模型）
    
    Returns:
        更新后的transcript，包含speaker_id字段
    """
    if not PYANNOTE_AVAILABLE:
        logger.error("❌ Pyannote.audio 未安装，无法使用说话人分离")
        logger.error("   请运行: pip install pyannote.audio")
        # 返回原始transcript，所有片段标记为speaker_id="0"
        for item in transcript:
            if 'speaker_id' not in item:
                item['speaker_id'] = "0"
        return transcript
    
    try:
        logger.info("🎤 使用 Pyannote.audio 进行说话人分离...")
        
        # 获取 pipeline（带缓存）
        pipeline = get_pyannote_pipeline(use_auth_token)
        if pipeline is None:
            logger.error("❌ 无法加载 Pyannote pipeline")
            for item in transcript:
                if 'speaker_id' not in item:
                    item['speaker_id'] = "0"
            return transcript
        
        # 处理音频
        # 优先使用项目中的本地模型路径
        try:
            import os
            from pathlib import Path
            
            # 优先使用传入的 token，其次从环境变量读取
            hf_token = use_auth_token or os.getenv("HF_TOKEN")
            
            # 1. 首先检查项目中的本地模型目录
            # 获取当前文件所在目录，然后找到项目根目录
            current_file = Path(__file__).resolve()
            # funasr_standalone/pyannote_diarization.py -> 项目根目录
            project_root = current_file.parent.parent
            local_model_path = project_root / "models" / "pyannote_diarization"
            
            pipeline = None
            use_local_model = False
            
            # 检查本地模型目录是否存在且包含 config.yaml
            if local_model_path.exists() and (local_model_path / "config.yaml").exists():
                logger.info(f"✅ 检测到项目本地模型: {local_model_path}")
                
                # 检查子模型是否也在本地
                local_segmentation_path = project_root / "models" / "pyannote_segmentation"
                local_embedding_path = project_root / "models" / "pyannote_wespeaker"
                
                has_local_segmentation = local_segmentation_path.exists() and (local_segmentation_path / "config.yaml").exists()
                has_local_embedding = local_embedding_path.exists() and (local_embedding_path / "config.yaml").exists()
                
                if has_local_segmentation:
                    logger.info(f"✅ 检测到本地分割模型: {local_segmentation_path}")
                else:
                    logger.warning(f"⚠️ 未检测到本地分割模型: {local_segmentation_path}")
                
                if has_local_embedding:
                    logger.info(f"✅ 检测到本地嵌入模型: {local_embedding_path}")
                else:
                    logger.warning(f"⚠️ 未检测到本地嵌入模型: {local_embedding_path}")
                
                # 如果所有子模型都在本地，修改 config.yaml 以使用本地路径
                if has_local_segmentation and has_local_embedding:
                    try:
                        try:
                            import yaml
                        except ImportError:
                            logger.error("❌ 缺少 PyYAML 库，无法修改配置文件")
                            logger.error("   请安装: pip install PyYAML")
                            raise ImportError("PyYAML is required to modify config.yaml")
                        
                        import shutil
                        
                        # 读取原始 config.yaml
                        config_file = local_model_path / "config.yaml"
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                        
                        # 修改子模型路径为本地路径（使用绝对路径）
                        if 'pipeline' in config and 'params' in config['pipeline']:
                            config['pipeline']['params']['segmentation'] = str(local_segmentation_path.resolve())
                            config['pipeline']['params']['embedding'] = str(local_embedding_path.resolve())
                            logger.info(f"   已更新配置：segmentation -> {local_segmentation_path}")
                            logger.info(f"   已更新配置：embedding -> {local_embedding_path}")
                        
                        # 创建临时配置文件
                        temp_config_file = local_model_path / "config.yaml.local"
                        with open(temp_config_file, 'w', encoding='utf-8') as f:
                            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                        
                        # 备份原始配置文件
                        original_config_file = local_model_path / "config.yaml.original"
                        if not original_config_file.exists():
                            shutil.copy2(config_file, original_config_file)
                        
                        # 使用临时配置文件
                        shutil.copy2(temp_config_file, config_file)
                        logger.info("   已临时修改 config.yaml 以使用本地子模型")
                        
                        try:
                            # 使用本地路径加载模型
                            pipeline = Pipeline.from_pretrained(str(local_model_path))
                            logger.info("✅ 成功从项目本地路径加载 Pyannote 模型（使用本地子模型）")
                            use_local_model = True
                        finally:
                            # 恢复原始配置文件
                            if original_config_file.exists():
                                shutil.copy2(original_config_file, config_file)
                                logger.info("   已恢复原始 config.yaml")
                            if temp_config_file.exists():
                                temp_config_file.unlink()
                    except ImportError:
                        logger.warning("   ⚠️ 缺少 yaml 库，无法修改配置文件，尝试直接加载...")
                        try:
                            pipeline = Pipeline.from_pretrained(str(local_model_path))
                            logger.info("✅ 成功从项目本地路径加载 Pyannote 模型")
                            use_local_model = True
                        except Exception as local_load_error:
                            logger.warning(f"⚠️ 从本地路径加载失败: {local_load_error}")
                            logger.info("   将尝试从 HuggingFace 或缓存加载...")
                    except Exception as config_error:
                        logger.warning(f"⚠️ 修改配置文件失败: {config_error}")
                        logger.info("   尝试直接加载模型...")
                        try:
                            pipeline = Pipeline.from_pretrained(str(local_model_path))
                            logger.info("✅ 成功从项目本地路径加载 Pyannote 模型")
                            use_local_model = True
                        except Exception as local_load_error:
                            logger.warning(f"⚠️ 从本地路径加载失败: {local_load_error}")
                            logger.info("   将尝试从 HuggingFace 或缓存加载...")
                else:
                    # 如果子模型不完整，尝试直接加载（可能会从网络下载缺失的）
                    try:
                        pipeline = Pipeline.from_pretrained(str(local_model_path))
                        logger.info("✅ 成功从项目本地路径加载 Pyannote 模型")
                        use_local_model = True
                    except Exception as local_load_error:
                        logger.warning(f"⚠️ 从本地路径加载失败: {local_load_error}")
                        logger.info("   将尝试从 HuggingFace 或缓存加载...")
            
            # 2. 如果本地模型加载失败，尝试从 HuggingFace 或缓存加载
            if not use_local_model:
                # 检查本地缓存目录（Pyannote 通常缓存到 ~/.cache/pyannote/ 或 ~/.cache/huggingface/）
                cache_dirs = [
                    Path.home() / ".cache" / "pyannote",
                    Path.home() / ".cache" / "huggingface" / "hub",
                ]
                
                model_cached = False
                for cache_dir in cache_dirs:
                    if cache_dir.exists():
                        # 检查是否有 speaker-diarization-3.1 的缓存
                        model_path = cache_dir / "models--pyannote--speaker-diarization-3.1"
                        if model_path.exists():
                            model_cached = True
                            logger.info(f"✅ 检测到本地模型缓存: {model_path}")
                            break
                
                # 尝试加载模型
                try:
                    if hf_token:
                        # 新版本的 transformers 使用 token 参数，而不是 use_auth_token
                        try:
                            pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                token=hf_token
                            )
                        except TypeError:
                            # 兼容旧版本，如果 token 参数不支持，尝试 use_auth_token
                            pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=hf_token
                            )
                    else:
                        # 尝试不使用token（如果模型是公开的或已缓存）
                        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                    logger.info("✅ Pyannote 模型加载成功（从 HuggingFace）")
                except Exception as load_error:
                    error_str = str(load_error).lower()
                    if "network" in error_str or "unreachable" in error_str or "connection" in error_str:
                        if model_cached:
                            logger.warning(f"⚠️ 网络不可达，但检测到本地缓存，尝试使用缓存...")
                            # 如果网络不可达但有缓存，尝试强制使用本地
                            try:
                                # 尝试从缓存目录直接加载
                                pipeline = Pipeline.from_pretrained(
                                    "pyannote/speaker-diarization-3.1",
                                    local_files_only=True  # 仅使用本地文件
                                )
                                logger.info("✅ 成功使用本地缓存加载模型")
                            except Exception as local_error:
                                logger.error(f"❌ 无法使用本地缓存: {local_error}")
                                raise load_error  # 抛出原始错误
                        else:
                            logger.error(f"❌ 网络不可达且无本地缓存: {load_error}")
                            logger.error("   解决方案:")
                            logger.error("   1. 确保网络可以访问 HuggingFace，或配置代理")
                            logger.error("   2. 手动下载模型到本地缓存:")
                            logger.error("      python -c \"from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token='YOUR_TOKEN')\"")
                            logger.error("   3. 或使用已下载的模型路径")
                            raise load_error
                    else:
                        raise load_error
                    
        except Exception as e:
            logger.error(f"❌ 加载 Pyannote 模型失败: {e}")
            logger.error("   请确保:")
            logger.error("   1. 已安装 pyannote.audio: pip install pyannote.audio")
            logger.error("   2. 在 HuggingFace 上接受模型使用协议: https://huggingface.co/pyannote/speaker-diarization-3.1")
            logger.error("   3. 如果网络不可达，请先在有网络的机器上下载模型，然后复制缓存目录")
            logger.error("   4. 缓存目录通常在: ~/.cache/pyannote/ 或 ~/.cache/huggingface/hub/")
            # 降级：返回原始transcript
            for item in transcript:
                if 'speaker_id' not in item:
                    item['speaker_id'] = "0"
            return transcript
        
        # 处理音频
        logger.info(f"📂 处理音频文件: {audio_path}")
        diarization = pipeline(audio_path)
        
        # 构建说话人时间映射
        # diarization格式: (start, end, speaker_label)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start_time': turn.start,
                'end_time': turn.end,
                'speaker_id': speaker
            })
        
        logger.info(f"✅ Pyannote 识别出 {len(set(s['speaker_id'] for s in speaker_segments))} 个说话人")
        logger.info(f"   共 {len(speaker_segments)} 个说话人片段")
        
        # 将说话人信息映射到transcript
        # 对于每个transcript片段，找到时间重叠的说话人片段
        for item in transcript:
            item_start = item.get('start_time', 0)
            item_end = item.get('end_time', 0)
            
            # 找到时间重叠的说话人片段
            matched_speaker = None
            max_overlap = 0
            
            for seg in speaker_segments:
                seg_start = seg['start_time']
                seg_end = seg['end_time']
                
                # 计算重叠时间
                overlap_start = max(item_start, seg_start)
                overlap_end = min(item_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                # 如果重叠时间超过片段长度的50%，认为是匹配的
                item_duration = item_end - item_start
                if item_duration > 0 and overlap / item_duration > 0.5:
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_speaker = seg['speaker_id']
            
            # 如果找到匹配的说话人，使用它；否则使用最近的说话人
            if matched_speaker:
                item['speaker_id'] = matched_speaker
            else:
                # 找到最近的说话人片段
                min_distance = float('inf')
                nearest_speaker = None
                
                for seg in speaker_segments:
                    seg_start = seg['start_time']
                    seg_end = seg['end_time']
                    seg_center = (seg_start + seg_end) / 2
                    item_center = (item_start + item_end) / 2
                    
                    distance = abs(item_center - seg_center)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_speaker = seg['speaker_id']
                
                item['speaker_id'] = nearest_speaker if nearest_speaker else "SPEAKER_00"
        
        # 规范化说话人ID（从SPEAKER_00, SPEAKER_01... 转换为 0, 1, 2...）
        speaker_id_map = {}
        speaker_counter = 0
        
        for item in transcript:
            original_id = item.get('speaker_id', 'SPEAKER_00')
            if original_id not in speaker_id_map:
                speaker_id_map[original_id] = str(speaker_counter)
                speaker_counter += 1
            item['speaker_id'] = speaker_id_map[original_id]
        
        logger.info(f"✅ 说话人分离完成，共识别出 {len(speaker_id_map)} 个说话人")
        
        return transcript
        
    except Exception as e:
        logger.error(f"❌ Pyannote 说话人分离失败: {e}", exc_info=True)
        # 降级：返回原始transcript，所有片段标记为speaker_id="0"
        for item in transcript:
            if 'speaker_id' not in item:
                item['speaker_id'] = "0"
        return transcript
