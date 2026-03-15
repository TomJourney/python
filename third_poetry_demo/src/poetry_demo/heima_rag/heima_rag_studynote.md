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

   2. export OPENAI_API_KEY='XXX'

      export DASHSCOPE_API_KEY='XXXX'

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

#### 【2.3.1.1】llm分类任务介绍：

1. 下面几段文本来自某平台发布的金融领域文本：

```
1. 今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。
2. ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。本次交易是ABC公司在扩大业务范围，加强市场竞争力方面的重要举措。据悉，此次收购将进一步巩固ABC公司在行业中的地位，并为未来业务发展提供更广阔的发展空间。详情参见公司官网公告栏。
3. 公司资产负债表显示， 公司偿债能力强劲，现金流充足，为未来投资和扩张提供了坚实的财务基础。
4. 最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长，投资者应该关注这一领域的投资机会。
```

2. 我们的目标是期望模型能够帮助我们识别出这4段话中，每一句话描述的是一个什么类型的报告。即期望的输出结果为：['新闻报道', '公司公告', '财务公告', '分析师报告']

---

#### 【2.3.1.2】prompt提示词设计

1. <font color=red>对于llm， prompt的设计非常重要， 一个明确的prompt能够帮助我们更好从大模型中获得我们想要的结果</font>。
2. <font color=red>在该任务的prompt设计中，需要考虑2点</font>：
   1. 需要向模型解释什么叫做文本分类任务；
   2. 需要让模型按照我们指定的格式输出；
3. 为了让模型知道什么叫做文本分类，我们借用few-shot（少样本学习）的方式，给模型展示一些正确的例子：
   1. 例子：
      1. User：今日，股市经历一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。以上描述是['新闻报道', '公司公告', '财务公告', '分析师报告']里的什么类别？ 
      2. Bot：新闻报道
   2. 其中， User代表我们输入给模型的句子，Bot代表模型回复内容， 模型应当作出类似Bot的回答； 

---

<br>

### 【2.3.2】利用大模型实现文本分类代码

【0204_prompt_optimize_fin_text_classify.py】

```python
# 提示词优化案例_金融文本分类

# 1 获取client对象
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 示例数据
example_data = {
    '新闻报道':'今日，股市经历一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。',
    '财务报告':'本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康发展奠定了坚实基础。',
    '公司公告':'本公司高兴宣布成功完成最新一轮并购交易，收购了一家在人工智能领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力。',
    '分析师报告':'最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算，人工智能和数字化被认为是引领行业发展的关键因素。投资者应关注这些领域的发展态势。'
}

# 分类列表
example_types = ['新闻报道', '公司公告', '财务公告', '分析师报告']

# 提问数据
questions = [
    '今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。',
    'ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。本次交易是ABC公司在扩大业务范围，加强市场竞争力方面的重要举措。据悉，此次收购将进一步巩固ABC公司在行业中的地位，并为未来业务发展提供更广阔的发展空间。详情参见公司官网公告栏。',
    '公司资产负债表显示， 公司偿债能力强劲，现金流充足，为未来投资和扩张提供了坚实的财务基础。',
    '最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长，投资者应该关注这一领域的投资机会。',
    '小明喜欢小新哟'
]

# 注释：附带历史消息调用模型，附带多个角色
"""
[
    {"role":"system", "content":"你是金融专家，将文本分类为['新闻报道', '公司公告', '财务公告', '分析师报告'], 不清楚的分类为'不清楚类别'。下面有示例： "}
    
    {"role":"user", "content":"今日，央行发布公告宣布降息.........."},
    {"role":"assistant", "content":"新闻报道"},
    {"role":"user", "content":"ABC公司金融发布公告称，已成功完成对XYZ公司股..........."},
    {"role":"assistant", "content":"财务报告"},
    {"role":"user", "content":"公司资产负债表显示， 公司偿债能力强劲，现金流充足......"},
    {"role":"assistant", "content":"公司公告"},
    {"role":"user", "content":"最新的分析报告指出，可再生能源.........."},
    {"role":"assistant", "content":"分析师报告"},
    
    {"role":"user", "content":"要提问的问题"},
]
"""

messages = [
    {"role":"system", "content":"你是金融专家，将文本分类为['新闻报道', '公司公告', '财务公告', '分析师报告'], 不清楚的分类为'不清楚类别'。下面有示例： "}
]

# 追加到messages， 形成附带历史消息及多个角色调用模型
for key, value in example_data.items():
    messages.append({"role":"user", "content":value})
    messages.append({"role": "assistant", "content": key})

# print(messages)

# 向模型提问
for q in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages + [{"role":"user", "content":f"按照示例，回答这段文本的分类类别：{q}"}]
    )
    print(response.choices[0].message.content)
# 新闻报道
# 公司公告
# 财务公告
# 分析师报告
# 不清楚类别
```

<br>

---

## 【2.4】提示词工程-json数据格式

1. json对象与python对象对比：
   1. json对象 ：对应于python字典；
   2. json数组： python列表中含有多个字典；
   3. json在python中，就是字典和列表嵌套字典的字符串表现形式；

<br>

---

### 【2.4.1】python中使用json

1. python中使用json主要完成：
   1. <font color=red>（正向转换）python字典及列表：转换为json字符串</font>；
   2. <font color=red>（反向转换）json字符串：转换为python字典或列表</font>； 
2. 主要使用python内置的json库：
   1. json.dumps(字段或列表，ensure_ascii=False)： 将字典或列表转换为json字符串； 
      1. ensure_ascii： 确保中文能够正常显示； 
      2. 返回值： json字符串； 
   2. json.loads(json字符串) : 将json字符串转换为python字典或列表； 
      1. 返回值： python字典 或 puthon列表； 

<br>

---

### 【2.4.2】代码实现python字典与json的转换 

【0205_dict_json_convert_eachother.py】

```python
# python字典与json相互转换
import json

dict = {
    "name":"张三",
    "age":"11",
    "gender":"男"
}

print("\n========== python字典或字典列表转为json对象或json数组 \n")
# 转为json字符串
json_str = json.dumps(dict, ensure_ascii=False)
print(json_str) # {"name": "张三", "age": "11", "gender": "男"}

# 【2】 python字典列表转为json字符串
arr_dict = [
    {
    "name":"张三",
    "age":"3",
    "gender":"男"
    },
    {
    "name":"李四",
    "age":"4",
    "gender":"女"
    }
]
# 转为json数组
json_arr = json.dumps(arr_dict, ensure_ascii=False)
print(json_arr)
# [{"name": "张三", "age": "3", "gender": "男"}, {"name": "李四", "age": "4", "gender": "女"}]

# ========== json对象或数组转为python字典或列表
print("\n\n==========json对象或数组转为python字典或列表\n")
temp_json_str = '{"name": "张三", "age": "11", "gender": "男"}'
temp_json_arr = '[{"name": "张三", "age": "3", "gender": "男"}, {"name": "李四", "age": "4", "gender": "女"}]'

# 转为字典或数组
temp_dict = json.loads(temp_json_str)
temp_dict_arr = json.loads(temp_json_arr)

print(temp_dict, type(temp_dict))
# {'name': '张三', 'age': '11', 'gender': '男'} <class 'dict'>
print(temp_dict_arr, type(temp_dict_arr))
# [{'name': '张三', 'age': '3', 'gender': '男'}, {'name': '李四', 'age': '4', 'gender': '女'}] <class 'list'>
```

<br>

---

## 【2.5】提示词优化案例_金融文本信息抽取

### 【2.5.1】llm信息抽取任务介绍

1. 首先，我们定义信息抽取的schema： 

```python
schema={
  '金融':['日期', '股票名称', '开盘价', '收盘价', '成交量']
}
```

下面几段文本来自某平台发布的股票信息：

```c++
1. '2023-02-15, 寓意吉祥的节日，股票百度[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微副增加至460,000，投资者情况较为平稳。'， 
2. '2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，投资者保持观望态度。'
```

我们目的是：期望模型能够帮助我们识别出这2段话中的SPO三元组信息；

<br>

---

### 【2.5.2】prompt设计

1. 在该任务的prompt设计中， 我们主要考虑2点：
   1. 需要向模型解释什么叫做“信息抽取任务”； 
   2. 需要让模型按照我们指定的格式(JSON)输出； 
2. 为了让模型知道什么叫做信息抽取， 我们借用few-shot的方式，先给模型展示几个正确的例子：
   1. 例：
      1. User：'2023-01-10， 股市震荡。股票谷歌-D[EOOE]美股今日开盘价100美元，一度飙升到105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。' 提取上述句子中“金融('日期', '股票名称', '开盘价', '收盘价', '成交量')类型的实体”，并按照json格式输出，上述句子中没有的信息用['原文中未提及']来表示，多个值之间用',' 分隔； 
      2. Bot： {'日期':'2023-01-10', '股票名称':'谷歌-D[EOOE]美股',  '开盘价':'100美元', '收盘价':'102美元', '成交量':'520000'}
3. 

【代码实现-0206_prompt_optimize_fin_info_abstract.py】

```python
# 提示词优化案例——金融信息抽取


# 1 获取client对象
from openai import OpenAI
import json
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 元数据： 数据结构
schema = ['日期', '股票名称', '开盘价', '收盘价', '成交量']

# 样本数据
example_data = [
    {
        "content":"2023-01-10， 股市震荡。强大科技A股今日开盘价100人民币，一度飙升到105人民币，随后回落至98美元，最终以102美元收盘，成交量达到520000。",
        "answers":{
            "日期":"2023-01-10",
            "股票名称":"强大科技A股",
            "开盘价":"100人民币",
            "收盘价":"102人民币",
            "成交量":"520000"
        }
    },
    {
        "content": "2024-05-16， 股市利好。英伟达美股今日开盘价105美元，一度飙升到109美元，随后回落至100美元，最终以116美元收盘，成交量达到3560000。",
        "answers": {
            "日期": "2024-05-16",
            "股票名称": "英伟达美股",
            "开盘价": "105美元",
            "收盘价": "116没有",
            "成交量": "3560000"
        }
    }
]
# 问题列表
questions = [
    "2026-02-25, 股市利好。股票传智播客A股开盘价66人民币，一度飙升到77人民币，随后回落至65人民币，最终以68人民币收盘，成交量达到1230000。",
    "2026-02-26, 股市利好。股票黑马程序员A股开盘价200人民币，一度飙升到211人民币，随后回落至201人民币，最终以206人民币收盘。"
]

# 提示词设计
"""
[
    {"role":"system", "content":f"你帮我完成信息抽取，我给你句子，你抽取{schema}信息，按json字符串输出，如果某些信息部存在，用'原文未提及'表示，请参考如下示例："}, 
    
    {"role":"user", "content":"2023-01-10， 股市震荡。强大科技A股今日开盘价100人民币，一度飙升到105人民币，随后回落至98美元，最终以102美元收盘，成交量达到520000。"},
    {"role":"assistant", "{'日期':'2023-01-10', '股票名称':'强大科技A股',  '开盘价':'100人民币', '收盘价':'102人民币', '成交量':'520000'}"},
    
    {"role":"user", "content":"2024-05-16， 股市利好。英伟达美股今日开盘价105美元，一度飙升到109美元，随后回落至100美元，最终以116美元收盘，成交量达到3560000。"},
    {"role":"assistant", "{'日期':'2024-05-16', '股票名称':'英伟达美股',  '开盘价':'105美元', '收盘价':'116美元', '成交量':'3560000'}"},
    
     {"role":"user", "content":"按照上述例子，现在抽取这个句子的信息：{要抽取的句子文本}"},
]
"""

# 构建提示词
messages = [
    {"role":"system", "content":f"你帮我完成信息抽取，我给你句子，你抽取{schema}信息，按json字符串输出，如果某些信息部存在，用'原文未提及'表示，请参考如下示例："}
]
for example in example_data:
    messages.append(
        {"role":"user", "content":example["content"]}
    )
    messages.append(
        {"role": "assistant", "content": json.dumps(example["answers"], ensure_ascii=False)}
    )

for x in messages:
    print(x)

print("\n\n========== 追加用户的提问，并发送给大模型")
# 追加用户的提问
for question in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages + [{"role":"user", "content": f"按照上述示例，抽取下面这个句子的信息：{question}"}]
    )
    # 打印模型回复信息：
    print("\n大模型回复：", response.choices[0].message.content)

# 大模型回复： {"日期": "2026-02-25", "股票名称": "传智播客A股", "开盘价": "66人民币", "收盘价": "68人民币", "成交量": "1230000"}
# 大模型回复： {"日期": "2026-02-26", "股票名称": "黑马程序员A股", "开盘价": "200人民币", "收盘价": "206人民币", "成交量": "原文未提及"}


```

<br>

---

## 【2.6】提示词优化案例_金融文本匹配

### 【2.6.1】LLM信息文本匹配任务介绍

首先，我们构造几个短文本对：

```
1. ('股票市场今日大涨，投资者乐观', '持续上涨的市场让投资者感到满意')
2. ('油价大幅下跌，能源公司面临挑战', '未来智能城市的建设趋势愈发明显')
3. ('利率上升，影响房地产市场。', '高利率对房地产有一定冲击')
```

我们期望模型能够帮助我们识别出成对的句子，2句话是否有关联。 

我们期望的模型输出的结果为：

```
['是', '不是', '是']
```

<br>

---

### 【2.6.2】prompt设计

1. <font color=red>在该任务的prompt设计中，我们考虑以下2点：</font>
   1. 需要向模型解释什么叫做文本匹配任务；
   2. 需要让模型按照我们指定的格式输出；
2. 为了让模型知道文本匹配任务是什么，我们借用 few-shot方式，先给模型展示几个例子：
   1. 例子1：
      1. User: 
         1. 句子1：公司ABC发布了季度财报，显示盈利增长。
         2. 句子2：财报披露，公司ABC利润上升。
      2. Bot：是。
   2. 例子2：
      1. User：
         1. 句子1：黄金价格下跌，投资者抛售。 
         2. 句子2：外汇市场交易额创下新高。
      2. Bot：不是。

<br>

---

### 【2.6.3】提示词优化案例_金融文本匹配-代码实现

【0207_prompt_optimize_fin_text_match.py】

```python
# 1 获取client对象
from openai import OpenAI
import json
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

example_data = {
    "是":[
        ("公司ABC发布了季度财报，显示盈利增长。", "财报披露，公司ABC利润上升")
        , ("公司ITCAST发布了年度财报，显示盈利大幅度增长。", "财报披露，公司ITCAST更赚钱了。")
    ],
    "不是":[
        ("黄金价格下跌，投资者抛售。", "外汇市场交易额创下新高。")
        , ("央行降息，刺激经济增长。", "新能源技术的创新。")
    ]
}

# 问题清单
questions = [
    ("利率上升，影响房地产市场。", "高利率对房地产有一定的冲击"),
    ("油价大幅度下降，能源公司面临挑战", "未来只能城市的建设趋势越加明显。"),
    ("股票市场今日大涨，投资者乐观。", "持续上涨的市场让投资者感到满意。")

]

"""
    {"role":"system", "content":f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："}
    
    {"role":"user", "content":"句子1：[公司ABC发布了季度财报，显示盈利增长。] 句子2:[财报披露，公司ABC利润上升。]"}
    {"role":"assistant", "content":"是"} 
    
    {"role":"user", "content":"句子1：[公司ITCAST发布了年度财报，显示盈利大幅度增长。] 句子2:[财报披露，公司ITCAST更赚钱了。]"}
    {"role":"assistant", "content":"是"} 
    
    {"role":"user", "content":"句子1：[黄金价格下跌，投资者抛售。] 句子2:[外汇市场交易额创下新高。]"}
    {"role":"assistant", "content":"不是"} 
    
    {"role":"user", "content":"句子1：[央行降息，刺激经济增长。] 句子2:[新能源技术的创新。]"}
    {"role":"assistant", "content":"不是"} 
    
    {"role":"user", "content": f"按照上述示例，回答这2个句子的情况。句子1:[...], 句子2:[...]"}
"""

# 整理提示词
print("\n==========整理提示词\n")
messages = [
    {"role":"system", "content":f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："}
]

for key, value in example_data.items():
    for t in value:
        messages.append(
            {"role":"user", "content":f"句子1：[{t[0]}], 句子2：[{t[1]}]"}
        )
        messages.append(
            {"role": "assistant", "content": key}
        )
for x in messages:
    print(x)

print("\n==========整理问题，并向大模型提问\n")
# 整理问题，并向大模型提问
for question in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages = messages + [{"role":"user", "content":f"句子1：[{question[0]}], 句子2：[{question[1]}]"}]
    )
    print(response.choices[0].message.content)

# 是
# 不是
# 是
```

<br>

---

# 【3】rag开发

## 【3.1】LangChain简介 

1. langchain定义：为各种大模型实现通用接口，把大模型相关的组件链接在一起，简化大模型应用的开发难度，方便开发者快速开发复杂的大模型应用；
2. <font color=red>langchain主要功能（提供开发API，如提示词优化的API）：</font>
   1. Prompts：优化提示词（提示词工程）
   2. models： 调用各类模型； 
   3. History： 管理会话历史记录（记忆）
   4. Indexes： 管理和分析各类文档；
   5. Chains： 构建功能的执行链条； 
   6. Agent： 构建智能体； 

【补充】<font color=red>LangChain是后续学习RAG开发的主力框架</font>；

<br>

---

## 【3.2】LangChain环境部署

1. langchain安装：
   1. langchain：核心包；
   2. langchain-community： 社区支持包，提供更多的第三方模型调用（我们用的阿里云千问模型就需要这个包）
   3. langchain-ollama： ollama支持包，支持调用ollama托管部署的本地模型； 
   4. dashscope：阿里云通义千问的python sdk
   5. chromadb： 轻量级向量数据库；
2. 执行命令

```c++
poetry add langchain langchain-community langchain-ollama dashscope chromadb
```

<br>

---

## 【3.3】rag介绍

### 【3.3.1】rag定义

1. 通用基础大模型存在的问题：
   1. 问题1： 大模型的知识不是实时的，模型训练后不具备自动更新知识的能力，<font color=red>导致部分信息滞后</font>；
   2. 问题2：<font color=red>大模型领域知识是缺乏的</font>，大模型训练数据来自互联网和开源数据集，无法覆盖特定领域或高度专业化的内部知识；
   3. 问题3：<font color=red>幻觉问题</font>，大模型有时会在回答中生成看似合理但实际上是错误的信息；
   4. 问题4：<font color=red>数据安全性</font>； 
2. <font color=red>大模型存在的问题总结</font>：
   1. 领域知识匮乏；
   2. 过时； 
   3. 幻觉； 
   4. 安全；

<br>

---

### 【3.3.2】rag解决什么问题

1. <font color=red>rag：检索增强生成技术，解决大模型存在的问题； 利用检索外部文档提升生成结果质量</font>； 
   1. 领域知识和私有数据；
   2. 实时数据；
   3. 减少生成不确定性； 
   4. 增强数据安全； 
2. rag检索增强生成技术：为大模型提供了从特定数据源检索到的信息，以此来修正和补充生成的答案。
   1. <font color=red>可以总结为一个公式： RAG = 检索技术 + LLM提示 </font>;

<br>

---

### 【3.3.3】理解rag的工作流程

工作流图解：

![image-20260308211439997](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_workflowpng.png)

rag标准流程：

![image-20260308210933665](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_intro.png)

<br>

---

#### 【3.3.3.1】rag的工作原理

1. rag分为2个流程：
   1. 离线流程：知识库预处理；
   2. 在线流程：用户问题与检索生成；

![image-20260308211836568](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_online_offline.png)

<br>

#### 【3.3.3.2】rag标准流程总结（非常重要）

1. <font color=red>rag标准流程由索引index， 检索retrieve和生成generation三个核心阶段组成</font>。
   1. 索引阶段：通过处理多种来源多种格式的文档提取其中文本，将其切分为标准长度的文本块（chunk），并进行嵌入向量化Embedding， 向量存储在向量数据库（vector database）中。
      1. 加载文件；
      2. 内容提取；
      3. 文本分割，形成chunk
      4. 文本向量化；
      5. 存向量数据库；
   2. 检索阶段： 用户输入的问题或查询（query）被转化为向量表示，通过相似度匹配从向量数据库中检索出最相关的文本块；
      1. 问题或query向量化；
      2. 在文本向量中匹配出与问句向量相似的top_k个； 
   3. 生成阶段： 检索到的相关文本与原始查询（问题）共同构成提示词（prompt），输入大模型，生成精确且具备上下文关联的回答。
      1. 匹配出的文本作为上下文和问题一起添加到prompt中；
      2. 提交给大模型生成答案； 

<br>

---

### 【3.3.4】rag总结

1. 模型本质就是用户输入， 模型给出输出，用户能做到就是在输入上做功夫； 
2. rag就是在向模型提问前基于已有的知识库或文档内容做检索，确保向模型提问的内容更加精准以及包含足够的信息量用以提供给模型；
3. <font color=red>rag的核心工作是2个流程，包括离线流程-知识库预处理，在线流程-用户提问与检索生成</font>；

4. <font color=red>rag的核心价值：</font>
   1. 解决知识时效性问题：rag可以介入最新文档（如公式财报，政策文件），让模型输出与时俱进；
   2. 降级模型幻觉：模型的回答依赖检索到的事实性资料，而非纯靠自身记忆，大幅减少编造信息的概率；
   3. 无需重新训练模型：相比微调fine-turning， rag只需要更新知识库，成本更低，效率更高；

<br>

---

## 【3.4】向量的基础概念

1. rag流程中，向量库是一个重要节点：
   1. 离线流程：知识和信息 -> 向量嵌入（向量化） -> 存入向量库<font color=red>（向量数据库） </font>
   2. 在线流程： 用户的提问 -> 向量嵌入（向量化） -> 在向量库中匹配； 

### 【3.4.1】向量的基本概念

1. 向量Vector就是文本的数学身份证： 它把一段文字的语义信息，转换成一串固定长度的数字列表， 让计算机能够看懂文字的含义并做相似度计算；
   1. 简单来说， 就是让计算机更方便的理解不同的文本内容，是否表述的是一个意思；

### 【3.4.2】文本嵌入模型

1. 文本嵌入模型：如text-embedding-v1，通过深度学习等技术，从文本提取语义特征并映射为固定长度的数字序列； 
2. 文本嵌入过程：我们一般选用合适的文本嵌入模型来完成，如text-embedding-v1 ；

### 【3.4.3】向量匹配

1. 在向量匹配的过程中，如何识别2段文本是否表述相似的含义，<font color=red>主要可以通过如余弦相似度算法来完成</font>。

2. 如（下面案例的向量并非实际向量，仅描述概念）
   1. A：如何快速学打篮球 -> [0.2, 0.5, 0.8] 
   2. B：打篮球怎么学的快  -> [0.18, 0.52, 0.79] 
   3. C：运动后吃什么好呢？  -> [0.9, 0.1, 0.2] 
3. 通过余弦相似度算法可以得到： A和B相似度0.999789， A和C相似度0.361446 ；
   1. 由此可以通过精确的数学计算，取匹配2段文本是否描述为同一个意思，提高语义匹配的效率和精度；

#### 【3.4.3.1】精准的语义匹配

1. 如何更为精准的语义匹配，生成向量的维度是一个很重要的指标；
   1. 如text-embedding-v1模型，可以生成1536维的向量（<font color=red>一段文本固定得到1536个数字序列</font>），比较实用；
   2. 1536个数字表示，这段文本在1536个主题（抽象的语义特征）方向上的得分（强度）; 

2. 生成向量的维度越多，就更好记录文本的语义特征，做语义匹配会更加精准；
   1. 更多的向量会在计算，存储和匹配过程中，带来更大压力；
   2. <font color=red>选择合适的向量维度需要在精确和性能之间做平衡；一般选择1536维度是比较好的选择</font>；

![vector](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/vector.png)

<br>

---

### 【3.4.4】向量总结

1. 向量：也就是文本的数学身份证；它把一段文字的语义信息，转换为一串固定长度的数字列表，让计算机能够看懂文字的含义并做相似度计算；
   1. 向量的计算：是文本嵌入过程，可借助文本嵌入模型实现，如text-embedding-v1 ；
   2. 向量的匹配通过算法实现，如余弦相似度；
   3. 向量的维度表示一段文本在多个抽象语义特征方面的强度：
      1. 维度数代表模型用多少个抽象语义特征来描述文本；
      2. 维度越多，做语义匹配越精准；
      3. 性能压力也会增大；

<br>

---

## 【3.5】余弦相似度算法

### 【3.5.1】余弦相似度

1. 余弦相似度：向量的数字序列，共同决定了向量在高维空间中的方向和长度；
   1. 而余弦相似度主要就是撇除长度的影响，得到方向的夹角。夹角越小越相似，即方向相同；
2. 余弦相似度主要匹配的是：<font color=red>同向（无所谓长度）;</font>
3. 我们能够直接发现： [0.5, 0.5] 和 [0.7, 0.7] 是同向不同长度；
   1. 那这就需要依赖余弦相似度算法来计算相似度；

4. 余弦相似度定义：两个向量的点积 除以 两个向量模长的乘积；

$$
\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\|_2 \|B\|_2} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

5. 【例】余弦相似度计算：以 A(0.5, 0.5) , B(0.7, 0.7) 为例：

   1. 点积： 例如，计算两个向量点积时：$A \cdot B = 0.5*0.7 + 0.5*0.7 = 0.35 + 0.35 = 0.7$

   2. 模长：单个向量不同维度的平方之和开根号； 

   3. 对于向量 $A = (0.5, 0.5)$，其模长计算如下：
      $$
      \|A\|_2 = \sqrt{\sum_{i=1}^{2} A_i^2} = \sqrt{0.5^2 + 0.5^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} \approx 0.7071
      $$

6. 向量 $A=(0.5, 0.5)$，$B=(0.7, 0.7)$ 的余弦相似度为：

$$
\begin{aligned}
\cos(\theta) &= \frac{A \cdot B}{\|A\| \|B\|} \\
&= \frac{0.5 \times 0.7 + 0.5 \times 0.7}{\sqrt{0.5^2 + 0.5^2} \times \sqrt{0.7^2 + 0.7^2}} \\
&= \frac{0.35 + 0.35}{\sqrt{0.25 + 0.25} \times \sqrt{0.49 + 0.49}} \\
&= \frac{0.7}{\sqrt{0.5} \times \sqrt{0.98}} \\
&= \frac{0.7}{0.7071 \times 0.9899} \\
&= \frac{0.7}{0.7} = 1
\end{aligned}
$$

<br>

---

### 【3.5.2】余弦相似度计算python实现

【0305_cos_similarity.py】 

```python
# 计算余弦相似度
import numpy

# 计算点积
def get_dot(vec_a, vec_b):
    # return vec_a.dot(vec_b)  可以直接计算点积
    if len(vec_a) != len(vec_b):
        raise ValueError("2个向量的维度必须相同")

    dot_sum = 0
    for a, b in zip(vec_a, vec_b):
        dot_sum += a * b
    return dot_sum

# 计算模长
def get_norm(vector):
    sum_square = 0
    for element in vector:
        sum_square += element**2
    # 使用numpy sqrt开根号
    return numpy.sqrt(sum_square)

# 计算余弦相似度
def cos_similarity(vec_a, vec_b):
    result = get_dot(vec_a, vec_b) / ( get_norm(vec_a) * get_norm(vec_b) )
    print(result)

# 测试
if __name__ == '__main__':
    v1 = [0.5, 0.5]
    v2 = [0.7, 0.7]
    cos_similarity(v1, v2) # 1.0
```

<br>

---

## 【3.6】langchain调用大模型

1. langchain框架回顾：集成调用各种大模型的精简统一接口；
2. <font color=red>langchain目前支持3种类型的模型： 大模型LLMs，聊天模型Chat Models， 嵌入模型Embeddings Models</font>；
   1. 大模型：技术范畴统称；指基于大参数量，海量文本训练的Transformer架构模型，核心能力是理解和生成自然语言，<font color=red>主要服务于文本生成场景</font>；
   2. 聊天模型：应用范畴统称； 指转为对话场景优化的大模型，核心能力是模拟人类对话的轮次交互，<font color=red>主要服务于聊天场景</font>；
   3. 文本嵌入模型：文本嵌入模型接受文本作为输入，得到文本的向量；

3. LangChain支持的3类模型，它们是使用场景不同，输入和输出不同，开发者需要根据项目做对应选择；
   1. 补充：我们所用的阿里云千问系列主要来自于  langchain_community包； 

<br>

---

### 【3.6.1】langchain调用大模型示例

【0306_call_llm_remote.py】调用阿里云千问模型

```python
# 调用大模型
from langchain_community.llms.tongyi import Tongyi

# qwen3-max是聊天模型， qwen-max是大语言模型
model = Tongyi(model="qwen-max")

# 调用invoke向模型提问
result = model.invoke(input="你是谁呀，能做什么？")
print(result)
```

<br>

【0306_call_ollama_local.py】

```python
# 调用大模型-ollama本地模型
from langchain_ollama import OllamaLLM

# qwen3-max是聊天模型， qwen-max是大语言模型
model = OllamaLLM(model="qwen3:4b")

# 调用invoke向模型提问
result = model.invoke(input="你是谁呀，能做什么？")
print(result)
```

<br>

---

## 【3.7】langchain模型的流式输出

1. 如果需要流式输出结果，需要将模型的invoke方法修改为stream方法；
   1. invoke方法：一次性返回完整结果； 
   2. stream方法： 逐段返回结果， 流式输出

【0307_langchain_stream_call_llm_remote.py】langchain流式调用大模型

```python
# 调用大模型
from langchain_community.llms.tongyi import Tongyi

# qwen3-max是聊天模型， qwen-max是大语言模型
model = Tongyi(model="qwen-max")

# 调用 stream 向模型提问
result = model.stream(input="你是谁呀，能做什么？")
for chunk in result:
    print(chunk, end="", flush=True) # end表示每段分隔符为空串， flush=True表示立即显示

# 您好，我是Qwen，全名通义千问，是阿里云自主研发的超大规模语言模型。...... 
```

<br>

### 【3.7.1】总结

1. langchain有2个方法调用大模型：
   1. invoke：一次性返回完整结果； 
   2. stream，逐段流式输出结果； 
2. 这两个方法是新版langchain中基于Runnable接口的通用核心方法；
   1. <font color=red>绝大多数组件（如提示词模版，链，向量检索，工具调用等，后续学习）都支持这2个方法，这也是langchain设计的核心统一范式</font>； 

<br>

---

## 【3.8】langchain调用聊天模型

1. 聊天消息包含下面几种类型，使用时需要按照约定传入合适的值：
   1. AIMessage：AI输出的消息，可以是针对问题的答案。<font color=red>（openai库中的assistant角色）</font>
   2. HumanMessage：人类消息就是用户信息，由人给出的信息发送给LLMs的提示信息，比如“实现一个快速排序方法”。<font color=red>（openai库中的user角色）</font>
   3. SystemMessage：可以用于指定模型具体所处的环境和背景，如角色扮演等。你可以在这里给出具体的指示，比如“作为一个代码专家”，或者“返回json格式”。<font color=red>（openai库中的system角色）</font>

<br>

---

### 【3.8.1】使用不同角色调用聊天模型的python实现

【0308_langchain_call_chat_modles.py】调用聊天模型

```python
# 调用聊天模型

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# 得到模型对象，qwen3-max就是聊天模型
model = ChatTongyi(model="qwen3-max")

# 准备消息列表
messages = [
    SystemMessage(content="你是一个边塞诗人。"), # 或有
    HumanMessage(content="写一首唐诗")
]

# 调用stream流式执行
result = model.stream(input=messages)

# for循环迭代打印输出，通过.content来获取内容
for chunk in result:
    print(chunk.content, end="", flush=True)
```

【LLM回复结果】

```c++
《塞上曲》
朔风卷地裂寒旌，铁甲凝霜夜柝惊。
孤城落日驼铃碎，大漠连天雁字横。
血浸征袍埋骨处，春生野草牧羝声。
何须更觅封侯印，一剑能消万古兵。
```

<br>

---

【0308_langchain_call_chat_modles.py】 调用聊天模型

```python
# 调用聊天模型

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# 得到模型对象，qwen3-max就是聊天模型
model = ChatTongyi(model="qwen3-max")

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
```

【LLM回复】

```c++
戍楼月如钩，霜刃凝寒秋。  
谁怜征人骨，夜夜枕戈愁。
```

<br>

---

### 【3.8.2】使用不同角色调用本地聊天模型的python实现

【0308_langchain_call_chat_modles_local.py】调用本地ollama部署的模型

```python
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
```

【LLM回复结果】

```c++
《塞上曲》
朔风卷地白草折，胡马嘶风月似钩。
将军夜猎胡尘起，铁甲寒光泪满楼。
```

<br>

---

## 【3.9】langchain消息的简写形式

1. SystemMessage, HumanMessage, AIMessage 的第一种写法（非简写）

```python
# 准备消息列表
messages = [
    SystemMessage(content="你是一个边塞诗人。"), # 或有
    HumanMessage(content="按照以下格式，写一首唐诗"),
    AIMessage(content="助禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦"),  # 给出示例
]
```

2. SystemMessage, HumanMessage, AIMessage 的第二种写法（简写）

【0309_langchain_call_chat_modles_simple.py】消息简写形式

```python
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
```

【LLM回复结果】

```c++
戍楼月如钩，霜刃凝寒秋。  
谁怜征夫骨，寸寸尽离愁。
```

<br>

---

### 【3.9.1】langchain消息的简写形式总结 

1. 消息形式：
   1. 非简写（创建 SystemMessage, HumanMessage, AIMessage类对象）：是静态的，一步到位；
   2. 简写：（创建 SystemMessage, HumanMessage, AIMessage类对象）：是动态的，需要在运行时，由langchain内部机制转换为Message类对象；
2. <font color=red>langchain消息简写的好处</font>：
   1. 无需导入SystemMessage, HumanMessage, AIMessage包； 
   2. <font color=red>由于是动态的，需要转换步骤；所以简写形式支持内部填充{变量}占位符</font>；
      1. 可以在运行时填充具体值（后续学习提示词模版时用到）； 

```python
# 消息简写形式：支持内部填充{变量}占位
messages = [
    ('system', "今天的天气是{weather}"),
    ('human', "我的名字是：{name}"),
    ('ai', "欢迎{lastname}先生")
] 
```

<br>

---

## 【3.10】langchain调用嵌入模型

1. <font color=red>嵌入模型Embeddings Model的特点</font>：将字符串作为输入，返回一个浮点数的列表（向量）。在NLP中，Embedding的作用就是将数据进行文本向量化； 

【0310_langchain_call_embedding_modles.py】

```python
# 调用嵌入模型

from langchain_community.embeddings import DashScopeEmbeddings

# 创建模型对象， 不传入model，默认使用的是 text-embeddings-v1
model = DashScopeEmbeddings()

# 不用invoke ， stream
# 使用 embed_query, embed_documents
print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))
```

【LLM回复】

```c++
[-3.02587890625, 3.3109374046325684, 4.410546779632568, 0.4593261778354645, -4.43798828125, 0.844921886920929, ...]

```

<br>

---

### 【3.10.1】langchain调用3类模型的使用总结

![langchain_call_model_conclusion](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/langchain_call_model_conclusion.png)

<br>

---

## 【3.11】langchain通用提示词模版

### 【3.11.1】通用提示词模版

1. 提示词优化在模型应用中非常重要，langchain提供了PromptTemplate类，用来协助优化提示词；
2. <font color=red>PromptTemplate表示提示词模版</font>，可以构建一个自定义的基础提示词模版，支持变量的注入，最终生成所需的提示词；

### 【3.11.2】代码实现

【0311_langchain_prompt_template.py】langchain提示词模版实践

```python
# langchain通用提示词模版
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

# 提示词模版类PromptTemplate，是Runnable接口的实现类，它可以加入到langchain中的链条
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}， 刚生了{gender}, 你帮我起个名字，简单回答。"
)

# 调用 .format方法注入信息即可
prompt_text = prompt_template.format(lastname="张", gender="女儿")
print("prompt_text = " + prompt_text)
# 我的邻居姓张， 刚生了女儿, 你帮我起个名字，简单回答。

model = Tongyi(model="qwen-max")
result = model.invoke(input=prompt_text)
print(result) # 张家欣
```

<br>

【0311_langchain_chain_call_llm.py】<font color=red> langchain链对象</font>调用大模型

```python
# langchain通用提示词模版
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

# 提示词模版类PromptTemplate，是Runnable接口的实现类，它可以加入到langchain中的链条
# zero-shot 零样本学习
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}， 刚生了{gender}, 你帮我起个名字，简单回答。"
)

model = Tongyi(model="qwen-max")

# 创建链对象
chain = prompt_template | model
result = chain.invoke(input={"lastname":"张", "gender":"女儿"})
print(result)
# 张婉儿
```

<br>

### 【3.11.3】总结

1. 基于PromptTemplate类可以得到提示词模版，支持基于模板注入变量得到最终提示词； 
   1. zero-shot思想下， 可以基于PromptTemplate 直接完成；
   2. few-shot思想下， 需要更换为 FewShotPromptTemplate ；
2. <font color=red>使用PromptTemplate类的优点（为什么不使用拼接或格式化方式构建提示词）</font>
   1. 适应Template模板构建提示词，在大工程中更容易做标准化模版； 
   2. Template模板类，支持langchain框架的链式调用（Runnable接口）
      1. PromptTemplate  （zero-shot，零样本学习）
      2. FewShotPromptTemplate （few-shot， 少样本学习）
      3. ChatPromptTemplate （聊天提示词模版）

<br>

---

## 【3.12】langchain框架FewShotPromptTemplate的使用

### 【3.12.1】FewShotPromptTemplate-少样本提示词模板(带有示例的提示词)

【0312_langchain_fewshot_prompt_template.py】

```python
# langchain少样本提示词模版
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms.tongyi import Tongyi

example_prompt_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

example_data = [
    {"word":"大", "antonym":"小"},
    {"word":"上", "antonym":"下"}
]

# 少样本提示词模板FewShotPromptTemplate， 封装了通用模板对象PromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt_template, # 提示词模板
    examples=example_data,  # 示例数据 ，用于注入动态数据 ， list内套字典
    prefix="返回给定词的反义词，有如下示例：", # 示例数据之前的提示词
    suffix="基于示例告诉我，{input_word}的反义词是什么",  # 示例数据之后的提示词
    input_variables=['input_word']
)

# 获得最终提示词
prompt_text = few_shot_prompt.invoke(input={"input_word":"左"})
print(prompt_text.to_string())
# 单词：大, 反义词：小
# 单词：上, 反义词：下
# 基于示例告诉我，左的反义词是什么

# 调用大模型
model = Tongyi(model="qwen-max")
result = model.invoke(input=prompt_text)
print(result)  # 基于您给出的示例，"左" 的反义词是 "右"。
```

<br>

【3.12.2】FewShotPromptTemplate总结

1. FewShotPromptTemplate类对象构建需要5个核心参数：
   1. example_prompt： 示例数据的提示词模板 
   2. examples: 示例数据， list，内套字典 
   3. prefix： 组装提示词，在示例数据前的内容； 
   4. suffix： 组装提示词，在示例数据后的内容； 
   5. input_variables： 列表， 注入的变量列表 

```python
# 少样本提示词模板FewShotPromptTemplate， 封装了通用模板对象PromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt_template, # 提示词模板
    examples=example_data,  # 示例数据 ，用于注入动态数据 ， list内套字典
    prefix="返回给定词的反义词，有如下示例：", # 示例数据之前的提示词
    suffix="基于示例告诉我，{input_word}的反义词是什么",  # 示例数据之后的提示词
    input_variables=['input_word']
)

# 获得最终提示词
prompt_text = few_shot_prompt.invoke(input={"input_word":"左"})
print(prompt_text.to_string())
# 单词：大, 反义词：小
# 单词：上, 反义词：下
# 基于示例告诉我，左的反义词是什么
```

<br>

---

## 【3.13】langchain框架模板类的format和invoke方法

1. PropmptTemplate, FewShotPromptTemplate, ChatPromptTemplate 都拥有 format 和 invoke 这2类方法； 
2. 类继承结构如下：

![prompt_template](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/prompt_template.png)

3. format与invoke的区别：

| 区别   | format                               | invoke                                                       |
| ------ | ------------------------------------ | ------------------------------------------------------------ |
| 功能   | 纯字符串替换，解析占位符，生成提示词 | Runnable 接口标准方法，解析占位符生成提示词                  |
| 返回值 | 字符串                               | PromptValue类对象<br>（需要使用 .to_string()方法转为字符串） |
| 传参   | .format(k=v, k=v, ...)               | .invoke( {{ "k":v, "k":"v", ...... })                        |
| 解析   | 支持解析 {} 占位符                   | 支持解析 {} 占位符 和 MessagePlaceHolder <br> 结构化占位符 ； |

<br>

---

### 【3.13.1】format与invoke方法代码实现

【0313_langchain_prompt_template_format_invoke.py】

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

"""
PromptTemplate ->(extends) StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
"""

# 测试format
template = PromptTemplate.from_template("我的邻居是 {lastname}，最喜欢{hobby}")
result = template.format(lastname="张三", hobby="钓鱼")
print(result) # 我的邻居是 张三，最喜欢钓鱼
print(type(result)) # <class 'str'>

# 测试invoke
result2 = template.invoke({"lastname":"李四", "hobby":"唱歌"})
print(result2) # text='我的邻居是 李四，最喜欢唱歌'
print(type(result2)) # <class 'langchain_core.prompt_values.StringPromptValue'>
```

【<font color=red>format与invoke的应用场景</font>】

- 若需要使用langchain的链，则使用invoke返回的提示词是一个对象StringPromptValue；
- 若需要使用字符串类型的提示词，则使用format方法返回提示词； 

<br>

---



































