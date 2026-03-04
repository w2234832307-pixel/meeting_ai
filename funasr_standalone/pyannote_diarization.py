#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyannote 说话人分离模块
使用专业的 Pyannote.audio 模型进行说话人分离
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
import os
import shutil
import tempfile
import torch
import soundfile as sf
import subprocess
import torch
import torchaudio
import logging
from pathlib import Path
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("⚠️ Pyannote.audio 未安装，说话人分离功能将不可用")


# 全局 pipeline 缓存（避免重复加载）
_pipeline_cache = None


def _extract_annotation(diarization):
    """兼容不同版本 Pyannote，提取 Annotation 对象"""
    if hasattr(diarization, "itertracks"):
        return diarization
    
    # 尝试通过属性访问
    for attr in ["annotation", "speaker", "output", "result", "diarization"]:
        try:
            ann = getattr(diarization, attr, None)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
        except:
            continue
    
    # 尝试通过 __dict__ 访问
    try:
        obj_dict = vars(diarization) if hasattr(diarization, "__dict__") else {}
        for value in obj_dict.values():
            if value is not None and hasattr(value, "itertracks"):
                return value
    except:
        pass
    
    # 如果是 dataclass，尝试所有字段
    if hasattr(diarization, "__dataclass_fields__"):
        for field_name in diarization.__dataclass_fields__.keys():
            try:
                ann = getattr(diarization, field_name, None)
                if ann is not None and hasattr(ann, "itertracks"):
                    return ann
            except:
                continue
    
    # 如果是 dict
    if isinstance(diarization, dict):
        for key in ["annotation", "speaker", "output", "result", "diarization"]:
            ann = diarization.get(key)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
    
    raise TypeError(f"无法从 {type(diarization)} 提取 Annotation 对象")


logger = logging.getLogger(__name__)

# 全局模型缓存，避免每次切分音频都重新加载
_silero_model = None
_silero_utils = None

def _get_silero_vad_model():
    """本地离线加载 Silero VAD 模型"""
    global _silero_model, _silero_utils
    
    if _silero_model is not None:
        return _silero_model, _silero_utils
        
    try:
        # 自动定位到项目 models/silero-vad 目录 (与原代码中的本地寻址逻辑保持一致)
        project_root = Path(__file__).resolve().parent.parent
        silero_dir = project_root / "models" / "silero-vad"
        
        if not silero_dir.exists():
            logger.error(f"❌ 未找到本地 Silero VAD 目录: {silero_dir}")
            return None, None

        logger.info(f"⏳ 正在离线加载本地 Silero VAD: {silero_dir}")
        
        # 使用 torch.hub 本地加载模式 (source='local')
        model, utils = torch.hub.load(
            repo_or_dir=str(silero_dir),
            model='silero_vad',
            source='local',
            force_reload=False
        )
        
        # 将模型设为评估模式，并尽量利用 GPU 加速
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _silero_model = model.to(device).eval()
        _silero_utils = utils
        logger.info(f"✅ 本地 Silero VAD 加载完成 (Device: {device})")
        
    except Exception as e:
        logger.error(f"❌ 加载本地 Silero VAD 失败: {e}", exc_info=True)
        return None, None
        
    return _silero_model, _silero_utils


def detect_long_silence_with_vad(waveform: torch.Tensor, sample_rate: int, min_silence_duration: float = 2.0) -> list:
    """
    使用神经网络级别的 Silero VAD 替换原有的基础能量检测。
    精准提取高可靠性的长静音段（>2秒），为后续 Pyannote 切分提供干净的切口。
    """
    logger.info(f"🔍 使用 Silero VAD 扫描静音点（>={min_silence_duration}秒）...")
    
    try:
        model, utils = _get_silero_vad_model()
        if model is None:
            logger.warning("⚠️ Silero VAD 模型未就绪，退回空静音段")
            return []
            
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        device = next(model.parameters()).device
        
        # 1. 数据预处理
        # 确保音频与模型在同一个设备上
        if waveform.device != device:
            waveform = waveform.to(device)
            
        # Silero VAD 强烈依赖 16000Hz 采样率才能保证高准确率
        vad_sr = 16000
        if sample_rate != vad_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=vad_sr).to(device)
            vad_waveform = resampler(waveform)
        else:
            vad_waveform = waveform
            
        # 降频/混音为单声道 (Silero 需要 1D tensor)
        if vad_waveform.ndim > 1:
            if vad_waveform.shape[0] > 1:
                vad_waveform = torch.mean(vad_waveform, dim=0, keepdim=True)
            vad_waveform = vad_waveform.squeeze(0)
            
        # 2. 核心推理：获取所有讲话的时间戳
        # threshold=0.5 过滤底噪效果很好，min_speech_duration_ms 避免短促噪音被识别为语音
        speech_timestamps = get_speech_timestamps(
            vad_waveform, 
            model, 
            sampling_rate=vad_sr,
            threshold=0.5,
            min_speech_duration_ms=250 
        )
        
        # 3. 反转逻辑：从“讲话段”推导“静音段”
        silence_segments = []
        audio_duration = vad_waveform.shape[-1] / vad_sr
        current_time = 0.0
        
        for ts in speech_timestamps:
            speech_start = ts['start'] / vad_sr
            speech_end = ts['end'] / vad_sr
            
            # 计算两段讲话之间的空白时长
            silence_duration = speech_start - current_time
            if silence_duration >= min_silence_duration:
                silence_segments.append((current_time, speech_start))
            
            current_time = speech_end
            
        # 检查最后一段讲话到音频结尾的剩余部分
        if audio_duration - current_time >= min_silence_duration:
            silence_segments.append((current_time, audio_duration))
            
        logger.info(f"✅ Silero VAD 成功检测到 {len(silence_segments)} 个高质量切分点")
        return silence_segments
        
    except Exception as e:
        logger.error(f"❌ Silero VAD 运行时发生异常: {e}", exc_info=True)
        # 即使失败也返回空列表，让外层逻辑回退到硬切分，保证系统不崩溃
        return []

def split_audio_by_silence(waveform: torch.Tensor, sample_rate: int, 
                           silence_segments: List[tuple],
                           min_chunk_duration: float = 600.0,
                           max_chunk_duration: float = 1200.0) -> List[tuple]:
    """根据静音点将音频切分为 10-20 分钟的片段"""
    audio_duration = waveform.shape[-1] / sample_rate
    chunks = []
    
    if not silence_segments or len(silence_segments) > 1000:
        logger.info(f"⚠️ 静音段数量异常，使用固定时长切分")
        num_chunks = int(audio_duration / max_chunk_duration) + 1
        for i in range(num_chunks):
            start = i * max_chunk_duration
            end = min((i + 1) * max_chunk_duration, audio_duration)
            chunks.append((start, end))
        return chunks
    
    # 只选择足够长的静音段（>3秒）作为切分点
    valid_silence_points = []
    for silence_start, silence_end in silence_segments:
        if silence_end - silence_start >= 3.0:
            valid_silence_points.append((silence_start + silence_end) / 2)
    
    current_start = 0.0
    for cut_point in valid_silence_points:
        chunk_duration = cut_point - current_start
        if chunk_duration >= min_chunk_duration:
            chunks.append((current_start, cut_point))
            current_start = cut_point
        if chunk_duration >= max_chunk_duration:
            chunks.append((current_start, cut_point))
            current_start = cut_point
    
    if audio_duration - current_start > 0:
        chunks.append((current_start, audio_duration))
    
    # 如果切分结果不理想，回退到固定切分
    if len(chunks) == 0 or (len(chunks) == 1 and chunks[0][1] - chunks[0][0] > max_chunk_duration * 2):
        logger.info("⚠️ 切分结果不理想，回退到固定时长切分")
        chunks = []
        num_chunks = int(audio_duration / max_chunk_duration) + 1
        for i in range(num_chunks):
            start = i * max_chunk_duration
            end = min((i + 1) * max_chunk_duration, audio_duration)
            chunks.append((start, end))
    
    logger.info(f"✂️ 切分为 {len(chunks)} 个片段")
    return chunks


def extract_speaker_embeddings(pipeline, waveform: torch.Tensor, sample_rate: int, 
                               annotation) -> Dict[str, torch.Tensor]:
    """加载本地独立声纹模型"""
    import torch
    from pyannote.audio import Model
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)
    speaker_embeddings = {}

    try:
        # 1. 绝对定位你的本地 WeSpeaker 模型
        project_root = Path(__file__).resolve().parent.parent
        local_emb_dir = project_root / "models" / "pyannote_wespeaker"

        if not local_emb_dir.exists():
            logger.error(f"❌ 找不到本地声纹模型目录: {local_emb_dir}")
            return {}

        # 2. 独立加载声纹模型 (绕过 Pipeline 的黑盒)
        embedding_model = Model.from_pretrained(str(local_emb_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = embedding_model.to(device)
        embedding_model.eval()

        # 3. 提取特征
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if not speaker: continue
            
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            
            # 防止超长片段里的抢话污染特征
            duration_samples = end_sample - start_sample
            max_duration = int(sample_rate * 4.0) # 最多只取 4 秒提取声纹
            
            if duration_samples > max_duration:
                mid_point = start_sample + (duration_samples // 2)
                start_sample = mid_point - (max_duration // 2)
                end_sample = mid_point + (max_duration // 2)
            
            segment = waveform[:, start_sample:end_sample]
            
            if segment.shape[-1] < sample_rate * 1.5:
                continue

            segment = segment.to(device)

            with torch.no_grad():
                # 适配 Pyannote Model 期望的 (batch, channel, samples) 维度
                if segment.ndim == 1:
                    segment = segment.unsqueeze(0).unsqueeze(0)
                elif segment.ndim == 2:
                    segment = segment.unsqueeze(0)

                emb = embedding_model(segment)
                emb = emb.squeeze().cpu() # 移回内存
                
                if speaker not in speaker_embeddings:
                    speaker_embeddings[speaker] = []
                speaker_embeddings[speaker].append(emb)

        # 4. 剔除异常值后对声纹取均值，得到最纯净的“指纹”
        final_embeddings = {}
        for speaker, embs in speaker_embeddings.items():
            if not embs:
                continue
                
            emb_tensor = torch.stack(embs) # 形状: [N, D]
            
            # 如果只有1-2段声音，样本太少，直接取均值
            if len(emb_tensor) <= 2:
                final_embeddings[speaker] = torch.mean(emb_tensor, dim=0)
                continue
                
            # 第一步：计算初步的“质心”（粗略均值）
            centroid = torch.mean(emb_tensor, dim=0, keepdim=True)
            
            # 第二步：计算每段声纹与质心的余弦相似度
            sims = F.cosine_similarity(emb_tensor, centroid)
            
            # 第三步：抛弃掉偏离度最大的 20% 脏数据（孤立点/杂音）
            # 计算需要保留的数量 (至少保留前 80%)
            keep_count = max(int(len(emb_tensor) * 0.8), 1)
            
            # 获取相似度最高的前 keep_count 个索引
            top_indices = torch.topk(sims, keep_count).indices
            
            # 第四步：只用最纯净的那些特征，重新计算最终均值
            clean_embs = emb_tensor[top_indices]
            final_embeddings[speaker] = torch.mean(clean_embs, dim=0)

        logger.info(f"✅ 成功提取了 {len(final_embeddings)} 个说话人的全局声纹特征")
        return final_embeddings

    except Exception as e:
        # 使用 error 级别确保任何崩溃都会在控制台红色打印！
        logger.error(f"❌ 声纹提取发生崩溃: {e}", exc_info=True)
        return {}


def global_speaker_calibration(all_chunk_results: List[Dict], 
                               global_embeddings: Dict[str, torch.Tensor],
                               threshold: float = 0.60) -> Dict[str, str]:
    """全局声纹校准：对跨片段的 speaker ID 进行聚类校准"""
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        logger.warning("⚠️ sklearn 未安装，无法进行全局校准")
        return {}
    
    if not global_embeddings:
        logger.warning("⚠️ 没有 embedding，跳过全局校准")
        return {}
    
    # 准备数据
    speaker_ids = list(global_embeddings.keys())
    embeddings_matrix = torch.stack([global_embeddings[sid] for sid in speaker_ids]).cpu().numpy()
    
    # 计算相似度矩阵并聚类
    similarity_matrix = cosine_similarity(embeddings_matrix)
    distance_matrix = 1 - similarity_matrix
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        linkage='average',
        metric='precomputed'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # 创建映射
    id_mapping = {}
    cluster_to_global_id = {}
    global_id_counter = 0
    
    for i, speaker_id in enumerate(speaker_ids):
        cluster_id = cluster_labels[i]
        if cluster_id not in cluster_to_global_id:
            global_speaker_id = f"SPEAKER_{global_id_counter:02d}"
            cluster_to_global_id[cluster_id] = global_speaker_id
            global_id_counter += 1
        id_mapping[speaker_id] = cluster_to_global_id[cluster_id]
    
    logger.info(f"✅ 全局校准完成：{len(speaker_ids)} 个片段 speaker -> {global_id_counter} 个全局 speaker")
    return id_mapping


def _process_chunks(pipeline, waveform: torch.Tensor, sample_rate: int, 
                    chunk_boundaries: List[tuple], device: torch.device) -> tuple:
    """处理音频片段并提取 embedding"""
    from pyannote.core import Annotation, Segment
    
    all_chunk_results = []
    global_embeddings = {}
    
    for chunk_idx, (chunk_start_time, chunk_end_time) in enumerate(chunk_boundaries):
        start_sample = int(chunk_start_time * sample_rate)
        end_sample = int(chunk_end_time * sample_rate)
        chunk_waveform = waveform[:, start_sample:end_sample].clone()
        
        if chunk_waveform.shape[-1] < sample_rate * 0.5:
            continue
        
        if chunk_waveform.device != device:
            chunk_waveform = chunk_waveform.to(device)
        
        if chunk_idx % 5 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        
        chunk_diarization = pipeline({"waveform": chunk_waveform, "sample_rate": sample_rate})
        chunk_annotation = _extract_annotation(chunk_diarization)
        
        chunk_embeddings = {}
        try:
            chunk_embeddings = extract_speaker_embeddings(pipeline, chunk_waveform, sample_rate, chunk_annotation)
        except Exception as e:
            logger.error(f"⚠️ 片段 {chunk_idx + 1} embedding 提取失败: {e}", exc_info=True)
        
        all_chunk_results.append({
            "chunk_idx": chunk_idx,
            "start_time": chunk_start_time,
            "end_time": chunk_end_time,
            "annotation": chunk_annotation,
            "embeddings": chunk_embeddings
        })
        
        for speaker_id, embedding in chunk_embeddings.items():
            global_key = f"chunk_{chunk_idx}_{speaker_id}"
            global_embeddings[global_key] = embedding.cpu() if embedding.is_cuda else embedding
        
        del chunk_waveform, chunk_diarization, chunk_annotation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    return all_chunk_results, global_embeddings


def _merge_chunks(all_chunk_results: List[Dict], id_mapping: Dict[str, str] = None) -> 'Annotation':
    """合并所有片段"""
    from pyannote.core import Annotation, Segment
    
    all_segments = Annotation()
    for chunk_result in all_chunk_results:
        chunk_idx = chunk_result["chunk_idx"]
        chunk_start_time = chunk_result["start_time"]
        chunk_annotation = chunk_result["annotation"]
        
        for turn, _, speaker in chunk_annotation.itertracks(yield_label=True):
            if speaker is None or speaker == "":
                speaker = f"SPEAKER_{chunk_idx:02d}"
            
            if id_mapping:
                chunk_speaker_key = f"chunk_{chunk_idx}_{speaker}"
                global_speaker_id = id_mapping.get(chunk_speaker_key, speaker)
            else:
                global_speaker_id = speaker
            
            shifted_segment = Segment(
                turn.start + chunk_start_time,
                turn.end + chunk_start_time
            )
            all_segments[shifted_segment, global_speaker_id] = global_speaker_id
    
    return all_segments


def process_audio_with_pipeline(pipeline, waveform: torch.Tensor, sample_rate: int, 
                                max_chunk_duration: int = 300,
                                use_vad_smart_chunking: bool = True):
    """使用 Pyannote pipeline 处理音频（支持长音频分段处理）"""
    audio_duration = waveform.shape[-1] / sample_rate
    
    # 获取设备信息
    if hasattr(pipeline, "device"):
        device = pipeline.device
    elif hasattr(pipeline, "_device"):
        device = pipeline._device
    else:
        device = waveform.device if hasattr(waveform, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda" and torch.cuda.is_available():
        logger.info(f"🖥️ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"🖥️ 使用 CPU")
    
    if waveform.device != device:
        waveform = waveform.to(device)
    
    # 短音频直接处理
    if audio_duration <= max_chunk_duration:
        logger.info(f"⏱️ 音频时长 {audio_duration:.1f}秒，直接处理")
        return pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    # 长音频处理
    chunk_boundaries = None
    
    # 尝试 VAD 智能切分（仅对 <1小时 的音频）
    if use_vad_smart_chunking and audio_duration < 3600:
        try:
            silence_segments = detect_long_silence_with_vad(waveform, sample_rate, min_silence_duration=2.0)
            chunk_boundaries = split_audio_by_silence(waveform, sample_rate, silence_segments, 600.0, 1200.0)
            
            if not chunk_boundaries or len(chunk_boundaries) == 1 or len(chunk_boundaries) > 50:
                chunk_boundaries = None
            else:
                logger.info(f"✅ VAD 切分成功，共 {len(chunk_boundaries)} 个片段")
        except Exception as e:
            logger.warning(f"⚠️ VAD 切分失败: {e}")
            chunk_boundaries = None
    
    # 如果没有 VAD 切分结果，使用固定切分
    if not chunk_boundaries:
        optimized_chunk_duration = min(max_chunk_duration, 900.0)  # 最大 15 分钟
        num_chunks = int(audio_duration / optimized_chunk_duration) + 1
        chunk_boundaries = [
            (i * optimized_chunk_duration, min((i + 1) * optimized_chunk_duration, audio_duration))
            for i in range(num_chunks)
        ]
        logger.info(f"✂️ 固定切分为 {len(chunk_boundaries)} 个片段（每段约 {optimized_chunk_duration:.0f}秒）")
    
    # 处理所有片段
    logger.info(f"🚀 开始处理 {len(chunk_boundaries)} 个片段...")
    all_chunk_results, global_embeddings = _process_chunks(pipeline, waveform, sample_rate, chunk_boundaries, device)
    
    # 全局声纹校准
    if len(all_chunk_results) > 1 and global_embeddings:
        logger.info(f"🔗 开始全局声纹校准（共 {len(global_embeddings)} 个 speaker embedding）...")
        id_mapping = global_speaker_calibration(all_chunk_results, global_embeddings, threshold=0.60)
        all_segments = _merge_chunks(all_chunk_results, id_mapping)
        logger.info(f"✅ 全局校准完成，合并 {len(all_chunk_results)} 个片段")
        return all_segments
    else:
        all_segments = _merge_chunks(all_chunk_results)
        logger.info(f"✅ 合并完成，共 {len(all_segments)} 个说话人片段")
        return all_segments


def get_pyannote_pipeline(use_auth_token: Optional[str] = None):
    """获取 Pyannote pipeline（优先使用项目本地 models/ 目录，强制离线）"""
    global _pipeline_cache
    
    if _pipeline_cache is not None:
        return _pipeline_cache
    
    if not PYANNOTE_AVAILABLE:
        return None
    
    hf_token = use_auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    
    # 强制离线
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    def _safe_from_pretrained(model_ref: str):
        """兼容不同版本参数名"""
        if Path(model_ref).exists():
            return Pipeline.from_pretrained(model_ref)
        try:
            if hf_token:
                return Pipeline.from_pretrained(model_ref, token=hf_token)
            return Pipeline.from_pretrained(model_ref)
        except TypeError as e:
            msg = str(e)
            if hf_token and ("token" in msg or "unexpected keyword argument" in msg):
                return Pipeline.from_pretrained(model_ref, use_auth_token=hf_token)
            raise
    
    try:
        project_root = Path(__file__).resolve().parent.parent
        local_diar_dir = project_root / "models" / "pyannote_diarization"
        local_cfg = local_diar_dir / "config.yaml"
        local_offline_cfg = local_diar_dir / "offline_config.yaml"
        local_seg_dir = project_root / "models" / "pyannote_segmentation"
        local_emb_dir = project_root / "models" / "pyannote_wespeaker"
        
        if not (local_diar_dir.exists() and local_cfg.exists()):
            logger.error(f"❌ 未找到本地 Pyannote 模型目录: {local_diar_dir}")
            return None
        
        if not local_seg_dir.exists() or not local_emb_dir.exists():
            logger.error(f"❌ 缺少本地模型目录: segmentation={local_seg_dir.exists()}, embedding={local_emb_dir.exists()}")
            return None
        
        cfg_source = local_offline_cfg if local_offline_cfg.exists() else local_cfg
        logger.info(f"✅ 使用本地 Pyannote 配置: {cfg_source}")
        
        try:
            import yaml
        except Exception as e:
            logger.error(f"❌ 缺少 PyYAML: {e}")
            return None
        
        original_cfg_bytes = local_cfg.read_bytes()
        try:
            cfg_obj = yaml.safe_load(cfg_source.read_text(encoding="utf-8"))
            if not isinstance(cfg_obj, dict):
                raise ValueError("config.yaml 解析结果不是 dict")
            
            pipeline_section = cfg_obj.setdefault("pipeline", {})
            params = pipeline_section.setdefault("params", {})
            
            # 移除 dict 类型的 clustering 配置
            if isinstance(params.get("clustering"), dict):
                params.pop("clustering", None)
            
            # 查找本地 PLDA 资源
            local_plda_candidates = [
                project_root / "models" / "pyannote_speaker_diarization_community_1",
                project_root / "models" / "speaker-diarization-community-1",
                project_root / "models" / "pyannote_plda",
            ]
            plda_ref = None
            for cand in local_plda_candidates:
                xvec_root = cand / "xvec_transform.npz"
                plda_root = cand / "plda.npz"
                xvec_sub = cand / "plda" / "xvec_transform.npz"
                plda_sub = cand / "plda" / "plda.npz"
                
                if xvec_root.exists() and plda_root.exists():
                    plda_ref = str(cand.resolve())
                    break
                if xvec_sub.exists() and plda_sub.exists():
                    flat_dir = project_root / "models" / "_pyannote_plda_flat"
                    flat_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(xvec_sub, flat_dir / "xvec_transform.npz")
                        shutil.copy2(plda_sub, flat_dir / "plda.npz")
                        plda_ref = str(flat_dir.resolve())
                        break
                    except Exception as e:
                        logger.error(f"❌ 展开 PLDA 文件失败: {e}")
                        return None
            
            if plda_ref:
                params["plda"] = plda_ref
            else:
                params["plda"] = "pyannote/speaker-diarization-community-1"
                logger.warning("⚠️ 未找到本地 PLDA 资源，将使用 HF cache")
            
            # 指向本地目录
            params["segmentation"] = str(local_seg_dir.resolve())
            params["embedding"] = str(local_emb_dir.resolve())
            
            # 写入临时配置
            local_cfg.write_text(yaml.safe_dump(cfg_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
            
            pipeline = _safe_from_pretrained(str(local_diar_dir))
            
            # 移动到 GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                pipeline = pipeline.to(device)
                logger.info(f"✅ Pipeline 已移动到 GPU: {device}")
            else:
                logger.warning("⚠️ CUDA 不可用，使用 CPU")
            
            # 尝试设置性能优化参数
            try:
                if hasattr(pipeline, "segmentation"):
                    seg = pipeline.segmentation
                    
                    # min_duration_on: 最短发音多久才算是一句话（可以保持0.5，过滤零碎咳嗽声）
                    # min_duration_off: 最短停顿多久就算一句话结束（从 0.5 降到 0.25，防止抢话连粘）
                    for attr_name, attr_value in [("min_duration_on", 0.5), ("min_duration_off", 0.25)]:
                        if hasattr(seg, attr_name):
                            setattr(seg, attr_name, attr_value)
                        elif hasattr(seg, "_segmentation") and hasattr(seg._segmentation, attr_name):
                            setattr(seg._segmentation, attr_name, attr_value)
                        elif hasattr(seg, "params") and isinstance(seg.params, dict):
                            seg.params[attr_name] = attr_value
                
                if hasattr(pipeline, "clustering") and hasattr(pipeline.clustering, "threshold"):
                    pipeline.clustering.threshold = 0.72
                    
            except Exception as e:
                logger.debug(f"⚠️ 设置 pipeline 参数失败: {e}")
            
            _pipeline_cache = pipeline
            logger.info("✅ Pyannote pipeline 加载成功")
            return pipeline
        except Exception as e:
            logger.error(f"❌ 本地 Pyannote pipeline 加载失败: {e}", exc_info=True)
            return None
        finally:
            try:
                local_cfg.write_bytes(original_cfg_bytes)
            except:
                pass
    
    except Exception as e:
        logger.error(f"❌ Pyannote 初始化错误: {e}", exc_info=True)
        return None


def perform_pyannote_diarization(
    audio_path: str,
    transcript: List[Dict],
    use_auth_token: Optional[str] = None
) -> List[Dict]:
    """使用 Pyannote 进行说话人分离"""
    if not PYANNOTE_AVAILABLE:
        logger.error("❌ Pyannote.audio 未安装")
        return transcript
    
    try:
        logger.info("🎤 使用 Pyannote.audio 进行说话人分离...")
        
        pipeline = get_pyannote_pipeline(use_auth_token)
        if pipeline is None:
            logger.error("❌ 无法加载 Pyannote pipeline")
            for item in transcript:
                if 'speaker_id' not in item:
                    item['speaker_id'] = "0"
            return transcript
        
        # 处理音频（支持 URL）
        tmp_path = None
        converted_audio_path = None
        try:
            if isinstance(audio_path, str) and audio_path.startswith(("http://", "https://")):
                import requests
                logger.info(f"🔗 下载音频 URL: {audio_path}")
    
                resp = requests.get(audio_path, timeout=(10, 30), stream=True)
                resp.raise_for_status()
                suffix = Path(audio_path).suffix or ".mp3"
    
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    
                    for chunk in resp.iter_content(chunk_size=131072):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name
                audio_path = tmp_path
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # 读取音频（如果不支持格式，使用 ffmpeg 转换）
            try:
                data, sample_rate = sf.read(audio_path)
            except Exception as e:
                logger.info(f"⚠️ soundfile 不支持该格式，使用 ffmpeg 转换")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    converted_audio_path = tmp_wav.name
                
                cmd = ["ffmpeg", "-i", audio_path, "-ac", "1", "-ar", "16000", "-f", "wav", "-y", converted_audio_path]
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                data, sample_rate = sf.read(converted_audio_path)
            
            if data.ndim == 1:
                data = data[None, :]
            else:
                data = data.T
            
            waveform = torch.tensor(data, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if waveform.device != device:
                waveform = waveform.to(device)
            
            diarization = process_audio_with_pipeline(pipeline, waveform, sample_rate)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            if converted_audio_path and os.path.exists(converted_audio_path):
                try:
                    os.remove(converted_audio_path)
                except:
                    pass
        
        # 提取 Annotation
        annotation = _extract_annotation(diarization)
        
        # 构建说话人时间映射
        speaker_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                'start_time': turn.start,
                'end_time': turn.end,
                'speaker_id': speaker
            })
        
        logger.info(f"✅ Pyannote 识别出 {len(set(s['speaker_id'] for s in speaker_segments))} 个说话人")
        
        # 将说话人信息映射到 transcript
        for item in transcript:
            item_start = item.get('start_time', 0)
            item_end = item.get('end_time', 0)
            
            # 用于统计这句话里，每个说话人累计出现了多长时间
            speaker_overlap_durations = {}
            
            for seg in speaker_segments:
                seg_start = seg['start_time']
                seg_end = seg['end_time']
                
                # 计算这段声音和这句话的重叠时间
                overlap_start = max(item_start, seg_start)
                overlap_end = min(item_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    spk_id = seg['speaker_id']
                    # 累加该说话人的重叠时间
                    speaker_overlap_durations[spk_id] = speaker_overlap_durations.get(spk_id, 0) + overlap
            
            # 找出重叠时间最长的那个说话人
            if speaker_overlap_durations:
                # 按照累计重叠时间降序排序，取第一名的 key
                matched_speaker = max(speaker_overlap_durations.items(), key=lambda x: x[1])[0]
            else:
                matched_speaker = "SPEAKER_00"
                
            item['speaker_id'] = matched_speaker
        
        # 规范化 ID
        speaker_id_map = {}
        speaker_counter = 0
        for item in transcript:
            original_id = item.get('speaker_id', 'SPEAKER_00')
            if original_id not in speaker_id_map:
                speaker_id_map[original_id] = str(speaker_counter)
                speaker_counter += 1
            item['speaker_id'] = speaker_id_map[original_id]
        
        return transcript
        
    except Exception as e:
        logger.error(f"❌ Pyannote 说话人分离失败: {e}", exc_info=True)
        for item in transcript:
            item['speaker_id'] = "0"
        return transcript
