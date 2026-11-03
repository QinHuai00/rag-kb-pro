
md5_path = "./md5.text"

# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# 相似度K值
similarity_threshold =1     # 检索返回匹配的文档数量

import os

embedding_model_name=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
# 原项目默认 qwen3-max，实际账号无该模型权限，默认改为 qwen-plus，可用环境变量覆盖
chat_model_name=os.getenv("RAG_CHAT_MODEL", "qwen-plus")

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}