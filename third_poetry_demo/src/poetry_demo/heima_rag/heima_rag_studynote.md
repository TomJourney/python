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

## 【2.3】OpenAI库附带历史消息调用模型 

1. 调用模型传入的参数messages，其要求的是list对象，即表明其支持非常多的消息在内；
   1. 我们可以基于此，将历史消息填入，让模型知晓对话的上下文，更好的回答；
   2. <font color=red>也就是说，可以在messages设定多个角色的语境上下文；</font>
   3. <font color=red> messages: 就是历史消息列表 </font>；

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
        {"role":"system", "content":"你是一个AI助理，简单回答"}
        , {"role":"user", "content":"小明有2条宠物狗"}
        , {"role":"assistant", "content":"好的"}
        , {"role":"user", "content":"小红有3条宠物猫"}
        , {"role":"assistant", "content":"好的"}
        , {"role":"user", "content":"总共有几只宠物？"}
    ],
    stream=True # 开启流式输出
)
# messages 就是历史消息列表；

# print(response.choices[0].message.content)
# 3 处理流式的响应结果
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True) # end=" "表示每段以空格分隔； flush=True表示立刻刷新缓冲区

# ========== 大模型回复内容：
# 小明有2条狗，小红有3只猫，所以总共有：
# 2 + 3 = **5只宠物**。
```

### 【小结】

1. 在messages的list内，组织历史消息提供给模型； 
2. 当前的历史消息是一次性的， 如果是生产系统可以将消息保存到文件，数据库等持久化工具内，需要的时候提取使用；
3. 后续学习 Langchain库， 会学习短期记忆和长期记忆的使用方法；

---

# 【2】提示词工程（Prompt Engineering）

## 【2.1】大模型prompt提示词工程指南 

1. 提示词工程定义： Prompt engineering， 也称为in-context prompt，<font color=red>指在不更新模型权重的情况下如何与大模型交互以引导其行为以获得所需结果的方法 </font>；
   1. 提示词工程：指包含与大语言模型交互和研发的各种技能和技术。提示工程在实现和大语言模型交互、对接，以及理解大语言模型能力方面都起着重要作用；

2. 人工智能领域，prompt指的是用户给大模型发出的指令；
   1. 如， 讲个笑话，用python编个贪吃蛇游戏，写封情书等；
   2. 虽然看似简单，但实际上，prompt的设计对于模型的结果影响很大；
   3. 因为如何设计prompt， 进而与模型更好的交互， 是研究人员必备的必不可少的技能（提示工程）；

---

### 【2.1.1】提示词技巧

1. 技巧1：详细的描述；
   1. 例-简单：写一封情书；
   2. 例-详细：用一些温柔的话语写一封情书，来表达我对你的仰慕和思念。 最后，我要求书写字体数要不低于500个字；
2. 技巧2：让模型充当某个角色；
   1. 例：我需要你充当一个AI算法面试官的角色，要求你自主的对我进行AI面试过程中常考的面试题，你可以一次说一个问题，然后我回答完成，你再出第二道题；
3. 技巧3：使用分隔符标明输入的不同部分；
   1. 中括号，xml标签，三引号等分隔符可以帮助划分要区别对待的文本；也可以帮助模型更好的理解文本内容。常用''' '''把内容框起来；
   2. 例：用20个字符总结由三引号分隔的文本。 ''' 在此插入文本 '''
      1. 提问：用20个字符总结由三引号分隔的文本。 ''' 今天的天气真好，我想去成都天府新区兴隆湖划皮划艇，吃个烧烤，放松放松 '''；
      2. 大模型回答：今日天气好，想去成都兴隆湖划艇烧烤放松。
4. <font color=red>技巧4：对任务指定步骤；</font> 对于可以拆分的任务可以尽量拆开，最后能够为其指定一系列步骤，明确步骤可以让模型更容易实现它们；
   1. 例：利用下面的分步情况来响应用户的输入。
      1. 步骤1： '''用户输入文本'''，用一句话总结这段文本，并加上前缀'Summary'
      2. 步骤2： 将步骤1中的摘要翻译成英语，并添加前缀"翻译："
      3. 例：
         1. 问题：利用下面的分步情况来响应用户的输入。步骤1： ''今天的天气真好，我想去成都天府新区兴隆湖划皮划艇，吃个烧烤，放松放松'''，用一句话总结这段文本，并加上前缀'Summary'。步骤2： 将步骤1中的摘要翻译成英语，并添加前缀"翻译："
         2. llm回复：Summary 今天天气好，想去成都兴隆湖划艇、吃烧烤放松。 翻译：The weather is nice today, and I want to go to Chengdu Xinglong Lake to kayak, have BBQ, and relax.
5. 技巧5：提供例子；本质类似于few-shot learning。先扔给大模型举例，然后让模型按照例子来输出。
   1. 按照这句评论文本的格式：'''用户输入文本''', 帮我创造新的样本；
   2. 例：
      1. 问题：'''今天的天气真好，我想去成都天府新区兴隆湖划皮划艇，吃个烧烤，放松放松'''，帮我创造新的样本；
      2. llm回答：基于原文本的结构和意图，为您创造了以下 5 个新样本：
         1. **同义改写**：天气不错，打算去兴隆湖划船吃烧烤，放松一下。
         2. **地点变换**：今天天气真好，我想去杭州西湖划船，吃个龙井虾仁，放松放松。
         3. **活动变换**：今天的天气真好，我想去成都天府新区兴隆湖露营，煮个火锅，放松放松。
         4. **风格正式**：鉴于今日气候宜人，拟前往成都天府新区兴隆湖开展皮划艇及烧烤活动，以求放松。
         5. **风格口语**：天儿真棒，想去兴隆湖划划艇，整点烧烤，歇会儿。
6. 技巧6：基于文本文档，辅助大模型回答，<font color=red>降级模型幻觉（一本正经的胡说八道）问题</font>。
   1. 使用参考文本作答 ； <font color=red>经典的知识库用法，让大模型使用我们提供的信息来组成答案</font>
   2. 例：
      1. 问题：根据下文中三重引号引起来的文章来回答问题。如果在文章中找不到答案，请写“我找不到答案”，不要自己造答案。'''<在此插入文档>''' '''<在此插入文档>'''  问题：<在此插入问题> 
      2. llm回答：张三与李四一共有 7 只小狗。

### 总结

1. 提示词工程就是更好的向模型提问的技巧； 
2. 大模型本身是一种简单结构： 即用户输入，模型输出；
   1. 用更详细，更清晰，有逻辑，有参考的提问，获得期望中的回答效果；
   2. 不管是rag还是agent智能体，或者是其他围绕模型的各类复杂的开发工作，本质上都可以简单总结为在提示词上下功夫；

---

<br>

## 【2.2】提示词优化案例和零样本少样本思想 

1. 术语介绍：
   1. 零样本： zero-shot 
   2. 少样本： few-shot

---

### 【2.2.1】实战案例背景

1. 背景描述：当前金融领域信息化发展的时代，金融数据大量激增，许多投资者和研究者试图通过对这些数据进行深度分析而获得一些有效的决策和帮助， 尽可能减少决策失误带来的损失； 
   1. 所以，针对金融数据的分析方法研究是目前十分有益且热门的话题；

2. 当前案例主要有三大业务场景实现：
   1. 基于大模型完成：金融文本分类；
   2. 基于大模型完成：金融文本信息抽取；
   3. 基于大模型完成：金融文本匹配；
3. <font color=red>采用方法：基于few-shot + zero-shot的思想（基于零样本与少样本的思想）</font>， 设计prompt提示词，进而应用大模型完成相应的任务；

---

### 【2.2.2】基于zero-shot思想

1. <font color=red>zero-shot（零样本学习）</font>：指在训练阶段不存在与测试阶段完全相同的类别，但模型可以使用训练过的知识来推广到测试集中的新类别上；这种能力被称为零样本学习， 因为模型在训练时从未见过测试集中的新类别，在模型训练和提示词优化中均有体现；
   1. <font color=red>零样本学习能力总结：模型对某个新事物完全不认识，但模型可以在已知的知识中抽取特征，然后做属性迁移，从而识别新类别</font>；
   2. 例： 如已知马是四脚兽，虎有条纹，熊猫是黑白色的特征；但没有训练斑马的数据；
      1. 告知模型： 斑马是四脚兽，有黑白色的条纹； 模型可以在已知数据中进行推理， 从而识别斑马；

3. <font color=red>提示词中利用零样本zero-shot的思想</font>：zero-shot思想基于已训练的能力，不提供任何示例，仅通过语言取描述任务的要求，目标和约束，让模型直接生成结果；
   1. <font color=red>简单来说：用语言定义任务，解放（信任）模型的预训练知识</font>；
   2. 例：
      1. 问题：请分析用户评论中的情感倾向，反馈 正面 或 负面”。用户评论如下：'''这款鸡胸肉饱腹感很强，吃起来不柴，很推荐'''。
      2. llm回复：正面；

---

### 【2.2.3】基于few-shot思想（少样本思想）

1. <font color=red>Few-shot学习(少样本学习)：</font>指模型在学习了一定类别的大量数据后，对于新类别，只需要少量样本就能够快速学习，对应的有one-shot learning，单样本学习，也算样本少到为一的情况下的一种few-shot learning。
2. 在模型训练中（相似度判断方法）： 基于少量企鹅样本并结合相似度判断，推论未知图片含有企鹅；
3. 同理，在提示词优化中： few-shot主要用于基于少量示例，让模型参考示例回答；
   1. 简单来说，少样本学习就是“用示例定义任务，在模型的预训练知识的基础上，提升模型回答的对齐精度（比如参考示例的格式）”
   2. 例：
      1. 请抽取产品名称和核心卖点2个字段，格式为json，有2个示例如下。示例1：MacBookPro高效节能，性能强大，适合牛马工作使用；输出：{"产品名称":"MacBookPro","产品卖点":"高效节能，性能强大"} 。示例2：联想笔记本拥有RTX4060独立显卡，畅玩游戏，丝滑流畅；输出：{"产品名称":"联想笔记本", "产品卖点":"畅玩游戏，丝滑流畅"} 。 请处理：华为MatepadPro，高清大屏，长效续航，你的好帮手。
      2. llm的回复： {"产品名称":"华为 MatepadPro","产品卖点":"高清大屏，长效续航"}

---

【2.2.4】总结

1. 提示词工程实战的业务需求是在金融业务内完成：
   1. 文本分类；
   2. 信息抽取；
   3. 文本匹配；

<font color=red>共上述3个任务目标，通过zero-shot， few-shot思想设计提示词完成</font>；

2. 在模型训练层面：
   1. zero-shot：零样本， 基于模型训练阶段学习的属性/语义关联，迁移到未知的新类别； 
   2. few-shot： 少样本， 基于少量样本，快速泛化识别新样本；
3. 在提示词优化层面：
   1. zero-shot： 无提示，语言描述任务，依赖模型预训练知识问答； 
   2. few-shot：少量样本，给于模型少量示例，引导模型对齐示例输出结果；

<br>

---

## 【2.3】提示词优化案例-金融文本分类任务

### 【2.3.1】few-shot少样本下的prompt提示词的设计



### 【2.3.2】利用大模型实现文本分类代码

















































