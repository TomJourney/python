# 调用聊天模型的消息简写形式

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# 得到模型对象，qwen3-max就是聊天模型
model = ChatTongyi(model="qwen3-max")

# 准备消息列表 (简写形式)
messages = [
    # (角色, 内容)  角色：只有3个选项，博阿凯system/human/ai
    ('system', "你是一个边塞诗人。"),
    ('human', "按照以下格式，写一首唐诗"),
    ('ai', "助禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦")
]

# 调用stream流式执行
result = model.stream(input=messages)

# for循环迭代打印输出，通过.content来获取内容
for chunk in result:
    print(chunk.content, end="", flush=True)

# 消息简写形式：支持内部填充{变量}占位
messages = [
    ('system', "今天的天气是{weather}"),
    ('human', "我的名字是：{name}"),
    ('ai', "欢迎{lastname}先生")
]