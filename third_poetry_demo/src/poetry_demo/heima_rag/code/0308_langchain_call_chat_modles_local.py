# 调用聊天模型

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# 得到模型对象，qwen3-max就是聊天模型
model = ChatOllama(model="qwen3:4b")

# 准备消息列表
messages = [
    SystemMessage(content="你是一个边塞诗人。"), # 或有
    HumanMessage(content="按照以下格式，写一首唐诗"),
    AIMessage(content="助禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦"),  # 给出示例
]

# 调用stream流式执行
result = model.stream(input=messages)

# for循环迭代打印输出，通过.content来获取内容
for chunk in result:
    print(chunk.content, end="", flush=True)