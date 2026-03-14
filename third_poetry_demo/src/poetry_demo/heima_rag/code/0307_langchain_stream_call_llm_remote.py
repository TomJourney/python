# 调用大模型
from langchain_community.llms.tongyi import Tongyi

# qwen3-max是聊天模型， qwen-max是大语言模型
model = Tongyi(model="qwen-max")

# 调用 stream 向模型提问
result = model.stream(input="你是谁呀，能做什么？")
for chunk in result:
    print(chunk, end="", flush=True) # end表示每段分隔符为空串， flush=True表示立即显示

# 您好，我是Qwen，全名通义千问，是阿里云自主研发的超大规模语言模型。......