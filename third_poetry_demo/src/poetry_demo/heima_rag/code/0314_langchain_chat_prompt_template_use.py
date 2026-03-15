from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

# 创建聊天提示词模版
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗"),

    ]
)

history_data = [
    ("human", "你来写一首唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一首"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦")
]

# StringPromptValue  to_string()
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
print(prompt_text)
# System: 你是一个边塞诗人，可以作诗
# Human: 你来写一首唐诗
# AI: 床前明月光，疑是地上霜，举头望明月，低头思故乡
# Human: 好诗再来一首
# AI: 锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦
# Human: 请再来一首唐诗

# 请求大模型
model = ChatTongyi(model="qwen3-max")
result = model.invoke(prompt_text)
print("====== 大模型回复内容：\n ")
print(result)
print(type(result))

# <class 'langchain_core.messages.ai.AIMessage'>

# 获取llm回复的字符串
print("====== 大模型回复的字符串类型的结果 \n")
print(result.content)
# 黄沙百战穿金甲，
# 不破楼兰终不还。
# 孤城落日连烽火，
# 铁马西风卷玉关。
#
# ——边塞戍卒志
