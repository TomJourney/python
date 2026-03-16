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

# ========== 调用大模型方式1： 先生成提示词，然后调大模型获取结果
print("\n调用大模型方式1： 先生成提示词，然后调大模型获取结果")
prompt_value = chat_prompt_template.invoke({"history": history_data}).to_string()
print(prompt_value)
model = ChatTongyi(model="qwen3-max")
result = model.invoke(prompt_value)
print(result)

print("\n========== 调用大模型方式2： 基于chain调用大模型")
# ==========  调用大模型方式2： 基于chain调用大模型
# 组成链 : 要求每一个组件都是Runnable接口的子类
chain = chat_prompt_template | model
# 方式2： 通过链调用invoke或stream
result = chain.invoke({"history": history_data})
print(result.content)
# 方式2： 通过链调用stream，并通过stream流式输出
print("\n========== 方式2： 通过链调用stream，并通过stream流式输出")
for chunk in chain.stream({"history":history_data}):
    print(chunk.content, end="", flush=True)

