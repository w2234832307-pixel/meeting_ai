"""
查看 ChromaDB 中存储的会议纪要数据
"""
import sys
import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 确保控制台能正确显示中文
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def view_chroma_data():
    """查看 ChromaDB 数据"""
    
    # 连接配置（从 .env 读取）
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "meeting_knowledge")
    
    print(f"🔗 正在连接 ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"📚 集合名称: {COLLECTION_NAME}\n")
    
    try:
        # 连接到 Chroma
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # 测试连接
        client.heartbeat()
        print("✅ ChromaDB 连接成功！\n")
        
        # 获取集合
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"❌ 集合 '{COLLECTION_NAME}' 不存在或无法访问: {e}")
            return
        
        # 获取集合统计信息
        count = collection.count()
        print(f"📊 集合统计:")
        print(f"   总记录数: {count}\n")
        
        if count == 0:
            print("ℹ️  集合为空，尚未归档任何会议纪要")
            return
        
        # 获取所有数据（如果数据量大，可以限制数量）
        print("=" * 80)
        print("📄 存储的数据详情:")
        print("=" * 80)
        
        # 查询所有数据
        results = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        # 按 source_id 分组显示
        data_by_source = {}
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
            document = results['documents'][i] if i < len(results['documents']) else ""
            embedding = results['embeddings'][i] if i < len(results['embeddings']) else []
            
            source_id = metadata.get('source_id', 'unknown')
            
            if source_id not in data_by_source:
                data_by_source[source_id] = []
            
            data_by_source[source_id].append({
                'id': doc_id,
                'metadata': metadata,
                'document': document,
                'embedding_dim': len(embedding) if embedding else 0
            })
        
        # 显示每个会议纪要的数据
        for source_id, chunks in sorted(data_by_source.items()):
            print(f"\n{'='*80}")
            print(f"📋 会议纪要 ID: {source_id}")
            print(f"{'='*80}")
            print(f"切片数量: {len(chunks)}")
            
            # 获取用户ID（如果有）
            user_id = chunks[0]['metadata'].get('user_id', 'N/A') if chunks else 'N/A'
            print(f"用户ID: {user_id}")
            
            print(f"\n{'─'*80}")
            for chunk_data in chunks:
                chunk_index = chunk_data['metadata'].get('chunk_index', '?')
                doc_text = chunk_data['document']
                
                print(f"\n  Chunk #{chunk_index} (ID: {chunk_data['id']})")
                print(f"  向量维度: {chunk_data['embedding_dim']}")
                print(f"  内容预览 (前200字符):")
                print(f"  {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
                print(f"  {'-'*76}")
        
        print(f"\n{'='*80}")
        print(f"✅ 数据查看完成！共 {len(data_by_source)} 个会议纪要，{count} 个切片")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()

def view_specific_meeting(minutes_id: int):
    """查看特定会议纪要的数据"""
    
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "meeting_knowledge")
    
    print(f"🔍 查询会议纪要 ID: {minutes_id}\n")
    
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # 查询特定 source_id 的所有切片
        results = collection.get(
            where={"source_id": minutes_id},
            include=["documents", "metadatas"]
        )
        
        if not results['ids']:
            print(f"❌ 未找到会议纪要 ID: {minutes_id}")
            return
        
        print(f"✅ 找到 {len(results['ids'])} 个切片\n")
        print("=" * 80)
        
        # 按 chunk_index 排序
        sorted_data = sorted(
            zip(results['ids'], results['documents'], results['metadatas']),
            key=lambda x: x[2].get('chunk_index', 0)
        )
        
        for doc_id, document, metadata in sorted_data:
            chunk_index = metadata.get('chunk_index', '?')
            print(f"\n切片 #{chunk_index} (ID: {doc_id})")
            print(f"{'-'*80}")
            print(document)
            print(f"{'-'*80}")
        
        print(f"\n{'='*80}")
        print("✅ 查询完成")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")

def search_content(query_text: str, top_k: int = 5):
    """语义搜索"""
    
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "meeting_knowledge")
    
    print(f"🔍 语义搜索: \"{query_text}\"")
    print(f"   Top-{top_k} 结果\n")
    
    try:
        # 需要先初始化 embedding 服务
        from app.services.embedding_factory import get_embedding_service
        
        embedding_service = get_embedding_service()
        query_vec = embedding_service.get_embedding(query_text)
        
        if not query_vec:
            print("❌ 向量化失败")
            return
        
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # 语义搜索
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'][0]:
            print("❌ 未找到相关结果")
            return
        
        print("=" * 80)
        for i, doc_id in enumerate(results['ids'][0]):
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # 计算相似度（L2距离转相似度）
            similarity = 1 / (1 + distance)
            
            print(f"\n结果 #{i+1} (相似度: {similarity:.3f})")
            print(f"会议纪要ID: {metadata.get('source_id', 'N/A')}")
            print(f"切片索引: {metadata.get('chunk_index', 'N/A')}")
            print(f"{'-'*80}")
            print(document[:300] + ('...' if len(document) > 300 else ''))
            print(f"{'-'*80}")
        
        print(f"\n✅ 搜索完成")
        
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()

def delete_meeting(minutes_id: int):
    """删除特定会议纪要的所有切片"""
    
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "meeting_knowledge")
    
    print(f"🗑️  准备删除会议纪要 ID: {minutes_id}")
    
    confirm = input("⚠️  确认删除？(yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 已取消删除")
        return
    
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # 查询所有相关切片的ID
        results = collection.get(
            where={"source_id": minutes_id},
            include=[]
        )
        
        if not results['ids']:
            print(f"❌ 未找到会议纪要 ID: {minutes_id}")
            return
        
        # 批量删除
        collection.delete(ids=results['ids'])
        
        print(f"✅ 已删除 {len(results['ids'])} 个切片")
        
    except Exception as e:
        print(f"❌ 删除失败: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="查看和管理 ChromaDB 中的会议纪要数据")
    parser.add_argument("--view", action="store_true", help="查看所有数据")
    parser.add_argument("--meeting", type=int, help="查看特定会议纪要 (提供 minutes_id)")
    parser.add_argument("--search", type=str, help="语义搜索 (提供查询文本)")
    parser.add_argument("--top-k", type=int, default=5, help="搜索结果数量 (默认5)")
    parser.add_argument("--delete", type=int, help="删除特定会议纪要 (提供 minutes_id)")
    
    args = parser.parse_args()
    
    if args.view:
        view_chroma_data()
    elif args.meeting:
        view_specific_meeting(args.meeting)
    elif args.search:
        search_content(args.search, args.top_k)
    elif args.delete:
        delete_meeting(args.delete)
    else:
        # 默认显示所有数据
        print("=" * 80)
        print("📚 ChromaDB 数据查看工具")
        print("=" * 80)
        print("  python view_chroma_data.py --view              # 查看所有数据")
        print("  python view_chroma_data.py --meeting 1001      # 查看特定会议")
        print("  python view_chroma_data.py --search '项目进展'  # 语义搜索")
        print("  python view_chroma_data.py --delete 1001       # 删除特定会议")
        print("\n默认执行 --view\n")
        view_chroma_data()
