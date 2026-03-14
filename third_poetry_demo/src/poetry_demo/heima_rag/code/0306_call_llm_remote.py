# 调用大模型
from langchain_community.llms.tongyi import Tongyi

# qwen3-max是聊天模型， qwen-max是大语言模型
model = Tongyi(model="qwen-max")

# 调用invoke向模型提问
result = model.invoke(input="你是谁呀，能做什么？")
print(result)

