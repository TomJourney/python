# 【README】

1. 课程名称：黑马程序员大模型RAG与Agent智能体项目实战教程，基于主流的LangChain技术从大模型提示词到实战项目;

2. <font color=red>课程目录：</font>
   1. 提示词优化：开发基础，基础到进阶，写精准提示词，提示词优化；
   2. Langchain1.2核心技术学习：核心组件，工作原理，零基础都能懂；
   3. rag实战案例：项目案例，文档处理，向量数据库搭建，向量数据检索匹配；
   4. agent智能体实战：需求拆解，agent定义，工具调用，流程控制，实战开发 ；

3. 本课程是大模型高级课程；
   1. 基础课程参见： 黑马程序员python+AI大模型零基础到项目实战； 

---

# 【1】前置准备

## 【1.1】通义千问大模型接入

步骤1：进入[阿里云百炼平台](https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29619931.J_SEsSjsNv72yRuRFS2VknO.2.42b910d7ouC9i4&tab=demohouse#/experience/llm)；

步骤2：创建API Key；

---

## 【1.2】代码调用云端大模型

1. 代码调用云端大模型：
   1. 创建api-key；
   2. 安装python的OpenAI库； poetry add openai
   3. 编码代码测试（从百炼平台复制代码，直接执行）

---

## 【1.3】使用环境变量保护APIKEY

1. 有两类APIKEY

   1. OPENAI_API_KEY：用于openapi库；
   2. DASHSCOPE_API_KEY: 用于langchain库； 

2. 配置方法：

   1. vim ~/.bash_profile ； 配置 API_KEY ;

   2. export OPENAI_API_KEY='sk-1bc2de9b0af1413b88538dbec5d6317f'

      export DASHSCOPE_API_KEY='sk-1bc2de9b0af1413b88538dbec5d6317f'

---

## 【1.4】Ollama简介

1. 背景： 基于Ollama部署本地模型，供代码调用； 
2. Ollama定义：一款旨在简化大模型本地部署和运行的开源软件； 
   1. 通过Ollama，开发者可以导入和定制自己的模型，无需关注复杂的底层细节；网址： [Ollama官网](https://ollama.com)

3. Ollama模型库： [Ollama模型库](https://ollama.com/library)

---

## 【1.5】windows与macOS系统部署Ollama

1. 安装Ollama步骤：
   1. 步骤1：下载Ollama；
   2. 步骤2：ollama run 模型名称；即可运行对应模型，并在命令行内做交互；
2. ollama安装的是蒸馏模型；
   1. <font color=red>蒸馏模型：可以理解为是标准大模型的学生，它学习了标准大模型的核心功能，但没有标准大模型强大；因为标准大模型的运行对硬件性能要求很高； </font>
   2. <font color=red>为了满足在个人pc机上运行大模型，蒸馏模型应运而生</font>；

---

## 【1.6】代码调用ollama的本地模型

1. 代码改动：
   1. 步骤1：修改openAI中的baseUrl为 localhost:11434/v1
   2. 步骤2：把模型修改为本地模型名称，如qwen3:4b

```python
client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    base_url="http://localhost:11434/v1",
)

messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model="qwen3:4b",  # 本地部署模型名称为qwen3:4b
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
```

<br>

---

# 【2】Python OpenAI库基础使用

## 【2.1】OpenAI库的基础使用

1. OpenAI SDK（OpenAI 库）定义：是OpenAI官方推出的python sdk，核心作用是让开发者能够简单，高效调用OpenAI的各类API（如GPT聊天），无需手动处理HTTP请求，身份验证等细节；
   1. 由于其发布较早且简单易用， 现如今许多模型服务商（如阿里云百炼平台）均兼容OpenAI SDK调用；
2. OpenAI SDK的使用步骤：
   1. 步骤1：获取客户端对象；
   2. 步骤2：调用模型； 
   3. 步骤3：处理结果；

---

### 【2.1.1】获取客户端对象

```python
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

### 【2.1.2】调用模型

```python
messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model="qwen3-max",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
```

1. 主要是2个参数：
   1. model： 模型名称；
   2. messages：提供给模型的消息：
      1. 类型为list，可以包含多个字典消息；
         1. 每个字典消息包含2个key，role-角色，content-内容； 
      2. 角色列表：
         1. system：设定助手的整体行为，角色和规则，为对话提供上下文框架（如指定助手身份，回答风格，核心要求），是全局的背景设置，影响后续所有交互；【例】 {"role":"system", "content":"你是一个python编程专家"}
         2. assistant：代表AI助手的回答，可以在代码中设定；【例】 {"role":"assistant", "content":"我是一个python编程专家，请问有什么可以帮助你的吗"}
         3. user角色：代表用户，发送问题，指令或需求；【例】{"role":"user", "content":"for循环输出1到5的数字"}

### 【2.1.3】处理结果

response变量：就是ChatCompletion对象，包含的信息如下所示。

```python
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
    ]
)

# 3 处理结果
print(response)
print(response.choices[0].message.content) # 打印处理结果 
```

【运行结果】

````c++
ChatCompletion(id='chatcmpl-f6344428-fe8e-9a2e-988f-3535ded6dd09', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='```python\nfor i in range(1, 11):\n    print(i)\n```', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1772930094, model='qwen3-max', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=19, prompt_tokens=57, total_tokens=76, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0)))
```python
for i in range(1, 11):
    print(i)
```
````

### 【总结】使用OpenAI的3个流程

1. 创建客户端对象（OpenAI类对象）
2. 和模型对话，可以提供3个角色使用：
   1. system: 设定模型的行为和规则；
   2. assistant: 设定模型的回答，由用户设定；
   3. user：用户的提问；

3. 处理结果： response.choices[0].message.content

---

## 【2.2】OpenAI库的流式输出

1. 可以设定结果输出为stream模式（流式输出），获得更好用户体验；
2. 开启流式输出主要分2步：
   1. 步骤1：在client.chat.completions.create()调用模型的时候设定参数，stream=True；
   2. 步骤2：for 循环response对象，并在循环内输出内容 ；

```python
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
```

【响应结果】

``` c++
for  i in range (1,  11):
    print (i)
```

---

【2.3】









































