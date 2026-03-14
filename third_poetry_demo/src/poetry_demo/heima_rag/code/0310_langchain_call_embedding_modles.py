# 调用嵌入模型

from langchain_community.embeddings import DashScopeEmbeddings

# 创建模型对象， 不传入model，默认使用的是 text-embeddings-v1
model = DashScopeEmbeddings()

# 不用invoke ， stream
# 使用 embed_query, embed_documents
print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))