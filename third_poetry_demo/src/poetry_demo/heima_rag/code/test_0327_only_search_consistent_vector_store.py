from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

# Chroma 向量数据库（轻量级的）
# 确保 langchain-chroma chromadb 这两个库安装了的

# 创建内存向量存储对象（内存数据库）
vector_store = Chroma(
    collection_name="test", # 类似于数据库表名
    embedding_function=DashScopeEmbeddings(), # 提供嵌入模型
    persist_directory="./chroma_db", # 指定数据存放的文件夹
)

# 删除保存文档到chroma向量数据库的代码，仅保留检索代码，如下。因为文档嵌入后已经被持久化到chroma向量数据库中。

# 检索
print("\n\n\n========== 检索 ")
result = vector_store.similarity_search(
    "python是不是简单易学",
    3, # 检索出几条最相似的结果
    filter={"source":"百度"} # 或有，设置过滤条件
)
print(result)