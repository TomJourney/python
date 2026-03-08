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
        {"role":"system", "content":"你是一个python编程专家，并且不说废话，简单回答"}
        , {"role":"assistant", "content":"我是编程专家，并且话不多，你要问什么？"}
        , {"role":"user", "content":"使用python代码，输出1-10数字"}
    ],
    stream=True # 开启流式输出
)

# print(response.choices[0].message.content)
# 3 处理流式的响应结果
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end=" ", flush=True) # end=" "表示每段以空格分隔； flush=True表示立刻刷新缓冲区

