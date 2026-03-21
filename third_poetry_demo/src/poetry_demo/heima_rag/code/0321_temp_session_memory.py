# 临时会话记忆
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# RunnableWithMessageHistory 帮助创建一个带有历史消息的新链
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate

model = ChatTongyi(model="qwen3-max")

# 方式1： 通用提示词模板
# prompt = PromptTemplate.from_template(
#     "你需要根据会话历史回应用户问题。对话历史：{chat_history}，用户提问：{input}，请回答"
# )
# chat_history 是函数get_history通过session_id获取的InMemoryChatMessageHistory类实例，并注入的

# 方式2： 聊天提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回答用户问题。会话历史如下："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答如下问题: {input}")
    ]
)

str_parser = StrOutputParser()

# 打印提示词
def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

# 创建基础链
base_chain = prompt | print_prompt | model | str_parser

# 创建一个字典，key是session_id， value就是 InMemoryChatMessageHistory 类对象
story = {}
# 实现通过会话id获取 InMemoryChatMessageHistory 类对象
def get_history(session_id):
    if session_id not in story:
        story[session_id] = InMemoryChatMessageHistory()
    return story[session_id]

# 创建一个新链(会话链)： 对基础链增强功能：自动附加历史消息
conversation_chain = RunnableWithMessageHistory(
    base_chain, # 被增强的chain
    get_history, # 通过会话id获取 InMemoryChatMessageHistory 类对象
    input_messages_key="input", # 表示用户输入在模板中的占位符
    history_messages_key="chat_history" # 表示用户输入在模板中的占位符
)

if __name__ == "__main__":
    # 固定格式，添加langchain配置，为当前程序配置所属的session_id
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }
    result = conversation_chain.invoke({"input":"小明有2只猫"}, session_config)
    print("第1次执行", result)

    result = conversation_chain.invoke({"input": "小刚有1只狗"}, session_config)
    print("第2次执行", result)

    result = conversation_chain.invoke({"input": "总共有几只宠物"}, session_config)
    print("第3次执行", result)