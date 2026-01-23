"""
历史会议处理服务
支持检索模式和总结模式两种处理方式
"""
import asyncio
from typing import List, Dict, Optional, Any
from app.core.logger import logger
from app.services.vector import vector_service
from app.services.llm_factory import get_llm_service_by_name


class MeetingHistoryService:
    """历史会议处理服务"""
    
    @staticmethod
    def determine_mode(
        meeting_ids: List[str],
        user_requirement: Optional[str],
        history_mode: str = "auto",
        threshold: int = 5
    ) -> str:
        """
        判断历史会议处理模式
        
        Args:
            meeting_ids: 会议ID列表
            user_requirement: 用户需求
            history_mode: 用户指定的模式（auto/retrieval/summary）
            threshold: 检索模式的会议数量阈值
        
        Returns:
            模式名称（retrieval/summary）
        """
        # 用户手动指定
        if history_mode in ["retrieval", "summary"]:
            logger.info(f"🎯 用户指定模式: {history_mode}")
            return history_mode
        
        # 自动判断
        meeting_count = len(meeting_ids)
        has_requirement = user_requirement and len(user_requirement.strip()) > 10
        
        # 场景1：会议少 + 有需求 → 检索模式（精确）
        if meeting_count <= threshold and has_requirement:
            logger.info(
                f"🔍 自动选择检索模式 "
                f"(会议数: {meeting_count}, 有需求: {has_requirement})"
            )
            return "retrieval"
        
        # 场景2：会议多 或 无需求 → 总结模式（宏观）
        logger.info(
            f"📝 自动选择总结模式 "
            f"(会议数: {meeting_count}, 有需求: {has_requirement})"
        )
        return "summary"
    
    @staticmethod
    async def process_by_retrieval(
        meeting_ids: List[str],
        user_requirement: Optional[str],
        current_transcript: str,
        top_k: int = 10,
        llm_model: str = "auto"
    ) -> Dict[str, Any]:
        """
        检索模式：从历史会议中精确检索相关信息
        
        适用场景：会议数量少（<= 5）且有明确需求
        
        Args:
            meeting_ids: 历史会议ID列表
            user_requirement: 用户需求
            current_transcript: 当前会议转录
            top_k: 检索数量
            llm_model: LLM模型名称
        
        Returns:
            {
                "mode": "retrieval",
                "relevant_segments": [...],
                "summary": "...",
                "meeting_count": N
            }
        """
        logger.info(f"🔍 检索模式: 从 {len(meeting_ids)} 个会议中检索相关内容")
        
        if not vector_service or not vector_service.is_available():
            logger.warning("⚠️ 向量服务不可用，返回空结果")
            return {
                "mode": "retrieval",
                "relevant_segments": [],
                "summary": "向量服务不可用",
                "meeting_count": len(meeting_ids)
            }
        
        # 构建检索查询
        # 优先使用 user_requirement，否则使用当前会议的关键内容
        query = user_requirement if user_requirement else current_transcript[:500]
        
        # 从向量库检索（带过滤）
        try:
            # 注意：这里需要向量服务支持 filters 参数
            # 如果你的 vector_service 不支持，需要修改 search_similar 方法
            search_results = await MeetingHistoryService._search_with_filter(
                query=query,
                meeting_ids=meeting_ids,
                top_k=top_k
            )
            
            if not search_results:
                return {
                    "mode": "retrieval",
                    "relevant_segments": [],
                    "summary": "未在历史会议中找到相关内容",
                    "meeting_count": len(meeting_ids)
                }
            
            # 构建相关片段列表
            relevant_segments = [
                {
                    "meeting_id": result.get("metadata", {}).get("meeting_id", "未知"),
                    "text": result.get("text", ""),
                    "speaker": result.get("metadata", {}).get("speaker", "未知"),
                    "timestamp": result.get("metadata", {}).get("timestamp", ""),
                    "relevance_score": result.get("score", 0.0)
                }
                for result in search_results
            ]
            
            # 用 LLM 生成简要总结
            segments_text = "\n\n".join([
                f"[{seg['meeting_id']} - {seg['speaker']} - {seg['timestamp']}]\n"
                f"{seg['text']}"
                for seg in relevant_segments
            ])
            
            llm_service = get_llm_service_by_name(llm_model)
            
            prompt = f"""
以下是从 {len(meeting_ids)} 个历史会议中检索到的相关片段：

{segments_text}

请生成一份简要总结（150字以内），提取关键信息。
"""
            
            summary = await asyncio.to_thread(llm_service.chat, prompt)
            
            return {
                "mode": "retrieval",
                "relevant_segments": relevant_segments,
                "summary": summary,
                "meeting_count": len(meeting_ids)
            }
            
        except Exception as e:
            logger.error(f"❌ 检索模式处理失败: {e}")
            return {
                "mode": "retrieval",
                "relevant_segments": [],
                "summary": f"检索失败: {str(e)}",
                "meeting_count": len(meeting_ids)
            }
    
    @staticmethod
    async def process_by_summary(
        meeting_ids: List[str],
        user_requirement: Optional[str],
        llm_model: str = "auto"
    ) -> Dict[str, Any]:
        """
        总结模式：对大量历史会议进行分块总结（Map-Reduce）
        
        适用场景：会议数量多（> 5）或无明确需求
        
        Args:
            meeting_ids: 历史会议ID列表
            user_requirement: 用户需求
            llm_model: LLM模型名称
        
        Returns:
            {
                "mode": "summary",
                "meeting_summaries": [...],
                "overall_summary": "...",
                "key_themes": [...],
                "processed_count": N,
                "total_count": N
            }
        """
        logger.info(f"📝 总结模式: 对 {len(meeting_ids)} 个会议进行分块总结")
        
        llm_service = get_llm_service_by_name(llm_model)
        
        # === Map 阶段：并行生成单会议摘要 ===
        async def summarize_single_meeting(meeting_id: str) -> Dict[str, Any]:
            """总结单个会议"""
            try:
                # 从向量库获取会议内容
                meeting_content = await MeetingHistoryService._get_meeting_content(
                    meeting_id
                )
                
                if not meeting_content:
                    return {
                        "meeting_id": meeting_id,
                        "summary": "无法获取会议内容",
                        "status": "failed"
                    }
                
                prompt = f"""
请总结以下会议的关键信息（150字以内）：

【会议内容】
{meeting_content}

要求：
1. 主要讨论议题
2. 重要决策和行动项
3. 关键参与人员的观点
"""
                
                summary = await asyncio.to_thread(llm_service.chat, prompt)
                
                return {
                    "meeting_id": meeting_id,
                    "summary": summary,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"❌ 总结会议 {meeting_id} 失败: {str(e)}")
                return {
                    "meeting_id": meeting_id,
                    "summary": f"总结失败: {str(e)}",
                    "status": "failed"
                }
        
        # 并行处理所有会议（提速）
        meeting_summaries = await asyncio.gather(*[
            summarize_single_meeting(mid) for mid in meeting_ids
        ])
        
        # 过滤成功的摘要
        successful_summaries = [
            s for s in meeting_summaries 
            if s["status"] == "success"
        ]
        
        if not successful_summaries:
            return {
                "mode": "summary",
                "meeting_summaries": [],
                "overall_summary": "无法生成总结：所有会议处理失败",
                "key_themes": [],
                "processed_count": 0,
                "total_count": len(meeting_ids)
            }
        
        # === Reduce 阶段：汇总生成整体总结 ===
        combined_text = "\n\n---\n\n".join([
            f"【会议 {i+1}: {s['meeting_id']}】\n{s['summary']}"
            for i, s in enumerate(successful_summaries)
        ])
        
        # 根据是否有用户需求，调整 prompt
        if user_requirement:
            final_prompt = f"""
基于以下 {len(successful_summaries)} 个历史会议的摘要，结合用户需求生成综合总结。

【历史会议摘要】
{combined_text}

【用户需求】
{user_requirement}

要求：
1. 重点关注与用户需求相关的内容
2. 总结跨会议的主要主题和趋势
3. 提取关键决策和行动项
4. 控制在 300 字以内
"""
        else:
            final_prompt = f"""
基于以下 {len(successful_summaries)} 个历史会议的摘要，生成综合总结。

【历史会议摘要】
{combined_text}

要求：
1. 总结整体讨论的主要主题（按重要性排序）
2. 提取跨会议的关键决策和行动项
3. 识别重复讨论的议题和趋势
4. 控制在 300 字以内
"""
        
        overall_summary = await asyncio.to_thread(llm_service.chat, final_prompt)
        
        # 提取主要主题
        key_themes = await MeetingHistoryService._extract_key_themes(
            successful_summaries,
            llm_service
        )
        
        return {
            "mode": "summary",
            "meeting_summaries": successful_summaries,
            "overall_summary": overall_summary,
            "key_themes": key_themes,
            "processed_count": len(successful_summaries),
            "total_count": len(meeting_ids)
        }
    
    @staticmethod
    async def _search_with_filter(
        query: str,
        meeting_ids: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        带过滤的向量检索
        
        注意：这是一个简化实现，实际需要向量库支持metadata过滤
        如果你的 vector_service 不支持，这里会检索所有结果再过滤
        
        Args:
            query: 查询文本
            meeting_ids: 要检索的会议ID列表
            top_k: 检索数量
        
        Returns:
            检索结果列表
        """
        try:
            # TODO: 这里需要你的 vector_service 支持 filters 参数
            # 如果不支持，需要修改 vector.py 的 search_similar 方法
            
            # 临时方案：调用标准检索，然后过滤
            # 这不是最优的，因为会检索很多不相关的结果
            
            # 获取向量
            query_vec = vector_service.get_embedding(query)
            if not query_vec:
                return []
            
            # 检索（这里假设你的 collection.query 支持过滤）
            # 如果不支持，需要检索更多结果再手动过滤
            results = vector_service.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k * 2,  # 多检索一些，因为要过滤
                include=["documents", "metadatas", "distances"]
            )
            
            # 过滤只保留指定会议的结果
            filtered_results = []
            
            if results and results.get("documents"):
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    meeting_id = metadata.get("meeting_id", "")
                    
                    # 只保留指定会议的结果
                    if meeting_id in meeting_ids or str(metadata.get("source_id", "")) in meeting_ids:
                        distance = distances[i] if i < len(distances) else float('inf')
                        similarity = 1 / (1 + distance)
                        
                        filtered_results.append({
                            "text": doc,
                            "metadata": metadata,
                            "score": similarity
                        })
                    
                    # 达到数量限制就停止
                    if len(filtered_results) >= top_k:
                        break
            
            logger.info(f"🔍 检索到 {len(filtered_results)} 条相关历史记录")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []
    
    @staticmethod
    async def _get_meeting_content(meeting_id: str) -> str:
        """
        获取会议完整内容
        
        优先级：
        1. 从数据库获取结构化数据（如果有）
        2. 从向量库获取所有片段并拼接
        
        Args:
            meeting_id: 会议ID
        
        Returns:
            会议内容文本
        """
        try:
            if not vector_service or not vector_service.is_available():
                return ""
            
            # 从向量库获取该会议的所有片段
            # 使用空查询或特定查询获取所有片段
            results = vector_service.collection.get(
                where={"source_id": int(meeting_id)} if meeting_id.isdigit() else {"meeting_id": meeting_id},
                limit=100  # 最多获取100个片段
            )
            
            if not results or not results.get("documents"):
                logger.warning(f"⚠️ 未找到会议 {meeting_id} 的内容")
                return ""
            
            # 拼接所有片段
            documents = results["documents"]
            content = "\n".join(documents)
            
            logger.info(f"✅ 获取会议 {meeting_id} 内容，共 {len(documents)} 个片段")
            return content
            
        except Exception as e:
            logger.error(f"❌ 获取会议内容失败: {e}")
            return ""
    
    @staticmethod
    async def _extract_key_themes(
        summaries: List[Dict[str, Any]],
        llm_service
    ) -> List[str]:
        """
        从多个会议摘要中提取主要主题
        
        Args:
            summaries: 会议摘要列表
            llm_service: LLM服务实例
        
        Returns:
            主题关键词列表
        """
        try:
            combined = " ".join([s["summary"] for s in summaries])
            
            prompt = f"""
从以下会议摘要中提取 5 个最主要的讨论主题（关键词），用逗号分隔：

{combined}

只输出主题关键词，不要其他内容。
"""
            
            themes_text = await asyncio.to_thread(llm_service.chat, prompt)
            themes = [t.strip() for t in themes_text.split(",")]
            
            return themes[:5]  # 最多返回5个
            
        except Exception as e:
            logger.error(f"❌ 提取主题失败: {e}")
            return []


# 创建单例实例
meeting_history_service = MeetingHistoryService()
