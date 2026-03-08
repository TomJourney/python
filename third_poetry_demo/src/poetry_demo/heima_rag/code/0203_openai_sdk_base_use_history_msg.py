# 1 获取client对象
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2 调用模型
response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role":"system", "content":"你是一个AI助理，简单回答"}
        , {"role":"user", "content":"小明有2条宠物狗"}
        , {"role":"assistant", "content":"好的"}
        , {"role":"user", "content":"小红有3条宠物猫"}
        , {"role":"assistant", "content":"好的"}
        , {"role":"user", "content":"总共有几只宠物？"}
    ],
    stream=True # 开启流式输出
)

# print(response.choices[0].message.content)
# 3 处理流式的响应结果
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True) # end=" "表示每段以空格分隔； flush=True表示立刻刷新缓冲区

# ========== 大模型回复内容：
# 小明有2条狗，小红有3只猫，所以总共有：
# 2 + 3 = **5只宠物**。

