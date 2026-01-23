"""
数据库服务 - 用于获取音频文件信息
"""
import pymysql
from typing import Optional, Dict, Any
from app.core.config import settings
from app.core.logger import logger


class DatabaseService:
    """数据库服务类"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.connection = None
    
    def _get_connection(self):
        """获取数据库连接"""
        if self.connection is None or not self.connection.open:
            try:
                self.connection = pymysql.connect(
                    host=settings.MYSQL_HOST,
                    port=settings.MYSQL_PORT,
                    user=settings.MYSQL_USER,
                    password=settings.MYSQL_PASSWORD,
                    database=settings.MYSQL_DB,
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
                logger.info(f"✅ MySQL连接成功: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}")
            except Exception as e:
                logger.error(f"❌ MySQL连接失败: {e}")
                raise
        
        return self.connection
    
    def get_audio_info(self, audio_id: int) -> Optional[Dict[str, Any]]:
        """
        从数据库获取音频文件信息
        
        Args:
            audio_id: 音频ID
        
        Returns:
            音频信息字典，包含：
            - id: 音频ID
            - file_url: 音频文件URL
            - file_path: 音频文件路径（如果有）
            - duration: 音频时长（秒）
            - file_size: 文件大小（字节）
            - format: 文件格式
            如果不存在则返回None
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                # 假设表名为 audio_files，字段可能需要根据实际表结构调整
                # 这里提供一个通用的查询示例
                sql = """
                    SELECT 
                        id,
                        file_url,
                        file_path,
                        duration,
                        file_size,
                        format,
                        created_at
                    FROM audio_files 
                    WHERE id = %s
                """
                cursor.execute(sql, (audio_id,))
                result = cursor.fetchone()
                
                if result:
                    logger.info(f"📋 从数据库获取音频信息: ID={audio_id}, URL={result.get('file_url', 'N/A')}")
                    return dict(result)
                else:
                    logger.warning(f"⚠️ 音频ID {audio_id} 在数据库中不存在")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 查询音频信息失败: {e}")
            return None
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.open:
            self.connection.close()
            logger.info("🔌 MySQL连接已关闭")


# 创建单例实例
database_service = DatabaseService()