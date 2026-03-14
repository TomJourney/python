# 调用大模型-ollama本地模型
from langchain_ollama import OllamaLLM

# qwen3-max是聊天模型， qwen-max是大语言模型
model = OllamaLLM(model="qwen3:4b")

# 调用invoke向模型提问
result = model.invoke(input="你是谁呀，能做什么？")
print(result)

