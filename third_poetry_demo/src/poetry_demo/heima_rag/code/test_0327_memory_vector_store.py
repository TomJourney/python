from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# 创建内存向量存储对象（内存数据库）
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(),
)

loader = CSVLoader(
    file_path="../data/info.csv",
    encoding="utf-8",
    source_column="source", # 指定本条数据的来源
)

documents = loader.load()
print(documents[0])
print(documents[1])
# page_content='source: 百度
# info: python是世界上最好的编程语言' metadata={'source': '百度', 'row': 0}
# page_content='source: 必应
# info: python学起来很简单' metadata={'source': '必应', 'row': 1}

# 向量存储的 新增，删除，检索
vector_store.add_documents(
    documents=documents, # 被添加的文档，类型：list[Document]
    ids=["id" + str(i) for i in range(1, len(documents)+1)] # 给添加的文档提供id（字符串） list[str]
)

# 删除 传入[id, id...]
vector_store.delete(["id1", "id2"])

# 检索
print("\n\n\n========== 检索 ")
result = vector_store.similarity_search(
    "python是不是简单易学",
    3, # 检索出几条最相似的结果
)
print(result)