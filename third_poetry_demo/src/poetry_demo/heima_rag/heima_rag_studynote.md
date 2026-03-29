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

## 【3.14】ChatPromptTemplate（聊天提示词模版）的使用

1. PromptTemplate回顾：
   1. PromptTemplate： 通用提示词模板， 支持动态注入信息；
   2. FewShotPromptTemplate： 支持基于模板注入任意数量的示例信息； 
2. <font color=red>ChatPromptTemplate： 支持注入任意数量的历史会话信息</font>；

3. 通过from_messages方法，从列表中获取多轮次会话作为聊天的基础模板 ；
   1. 补充：前面 PromptTemplate类用的 from_template 仅能够接入一条消息，而 from_messages 可以接入一个list的消息；

<br>

4. ChatPromptTemplate优点： 支持动态注入；
   1. 历史会话信息并不是静态的（固定的），而是随着对话的进行不停积攒，即动态的；
      1. 所以历史会话信息需要支持动态注入；
   2. MessagePlaceHolder作为占位符：提供history作为占位的key；
   3. <font color=red>基于invoke动态注入历史会话记录， 必须是invoke，format无法注入</font>；

### 【3.14.1】聊天提示词模版代码实现

【0314_langchain_chat_prompt_template_use.py】聊天提示词模版代码实现

```python
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
```

<br>

---

## 【3.15】langchain框架chain链的基础使用

1. <font color=red>链定义：把组件串联，上一个组件的输出，作为下一个组件的输入（类似于linux管道命令），这是langchain链（尤其是 | 管道链）的核心工作原理，这也是链式调用的核心价值</font>； 
2. 实现数据的自动化流转与组件的协同工作，代码如下： chain = prompt_template | model 
3. 核心前提： 即Runnable子类对象才能入链（以及Callable，Mapping接口子类对象也可以加入）；
4. 我们目前所学的组件，均是Runnable接口的子类，继承关系如下；

![chain_runnable_class_extend](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/chain_runnable_class_extend.png)

<br>

---

### 【3.15.1】chain链 

1. 通过 |(符号) 链接提示词模板对象和模型对象；
2. 返回值chain对象是RunnableSerializable对象；
   1. 是Runnable接口的直接子类； 
   2. 也是绝大多数组件的父类；
3. 通过invoke或stream进行阻塞执行或流式执行； 
4. 组成的链在执行上有：上一个组件的输出作为下一个组件的输入的特性； 

![chain_steps](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/chain_steps.png)

---

【0315_langchain_chain_base_use.py】代码实现

```python
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
```

<br>

### 【3.15.2】langchain链总结

1. <font color=red>langchain中链是一种将各个组件串联在一起，按顺序执行，前一个组件的输出作为下一个组件的输入</font>；
   1. 可以通过 "|" 符号来让各个组件形成链； 
   2. 成链的各个组件，需要Runnable接口的子类； 
   3. 形成的链是 RunnableSerializable 对象（Runnable接口子类）
   4. 可以通过链调用invoke或stream触发整个链条的执行；  

<br>

---

## 【3.16】或运算符的重写

### 【3.16.1】运算符重载 

1. 前文代码中： chain = chat_prompt_template | model 
   1. <font color=red>在语法上使用了 "|" 运算符的重写</font>； 

2. 在python中，运算符（如 “+”， “|”）的行为由类的魔法方法决定。 例如：

```python
a + b 的本质调用是 a.__add__(b)
a | b 的本质调用是 a.__or__(b)
```

只需要自行实现类的 \_ _or\_ _方法， 即可对 "|" 符号的功能进行重写； 

3. 示例：

```python
让 a | b | c 的代码得到一个自定义的类对象（类似列表即 [a, b, c]）
调用run方法依次输出 a, b, c 
我们需要重写 | 即 __or__ 方法
```

![operator_overload](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/operator_overload.png)

<br>

---

### 【3.16.2】运算符重载python实现（扩展）

```python
class Test(object):扩展
    def __init__(self, name):
        self.name = name

    def __or__(self, other):
        return MySequence(self, other)

    def __str__(self):
        return self.name

class MySequence(object):
    def __init__(self, *args):
        self.sequence = []
        for arg in args:
            self.sequence.append(arg)

    def __or__(self, other):
        self.sequence.append(other)
        return self

    def run(self):
        for item in self.sequence:
            print(item)

if __name__ == "__main__":
    a = Test("a")
    b = Test("b")
    c = Test("c")

    d = a | b | c # a.__or__(b)
    d.run()
    print(type(d))

# a
# b
# c
# <class '__main__.MySequence'>
```

<br>

---

## 【3.17】简单理解langchain框架的Runnable接口 

### 【3.17.1】Runnable接口 

1. <font color=red>langchain链的基础架构</font>：
   1. langchain中绝大多数核心组件都继承Runnable抽象基类（位于 langchain_core.runnables.base）
   2. 代码： ``` chain = prompt | model```
   3. chain变量是 RunnableSequence(RunnableSerilizable子类)类型， 而得到这个类型的原因就是 Runnable基类内部对 \_ \_or\_ _ 魔术方法的改写； 
   4. 同时，在后面继续使用 "|" 添加新的组件，依旧会得到 RunnableSequence， 这就是langchain链的基础架构；

【langchain框架#Runnable#_ _or_ _方法源码】

```python
def __or__(
    self,
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]:
    """Runnable "or" operator.

    Compose this `Runnable` with another object to create a
    `RunnableSequence`.

    Args:
        other: Another `Runnable` or a `Runnable`-like object.

    Returns:
        A new `Runnable`.
    """
    return RunnableSequence(self, coerce_to_runnable(other))
```

【langchain类型回顾】

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt = PromptTemplate.from_template("你是一个AI助手")
model = Tongyi(model="qwen3-max")

chain = prompt | model
print(type(chain))
# <class 'langchain_core.runnables.base.RunnableSequence'>
```

<br>

---

## 【3.18】StrOutputParser字符串输出解析器

### 【3.18.1】碰到的问题

1. 有如下代码，把第1次llm返回的结果，再送入llm，但报错：

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，仅告知名字无需其他内容"
)

chain = prompt | model | model
response = chain.invoke({"lastname":"张", "gender":"女儿"})
# 报错：ValueError: Invalid input type <class 'langchain_core.messages.ai.AIMessage'>. Must be a PromptValue, str, or list of BaseMessages.

print(response.content)
```

【错误根因分析】

- prompt的结果是 PromptValue类型，输入给了model 
- model的输出结果是 AIMessage ，再送入第2个model就不符号要求了；

2. 模型（ChatTongyi）源码中关于invoke方法明确指定了input的类型：

【langchain框架#Runnable#invoke方法源码】

```python
def invoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
```

<br>

<font color=red>【解决方法】 使用StrOutputParser-字符串输出解析器</font> ;

需要做类型转换， 可以借助 langchain内置的解析器：StrOutputParser 字符串输出解析器； 

![str_output_parser](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/str_output_parser.png)

<br>

---

### 【3.18.2】StrOutputParser字符串输出解析器 

1. StrOutputParser是langchain内置的简单字符串解析器；
   1. 可以将 AIMessage解析为简单的字符串，符合模型invoke方法要求（可传入字符串，不接受AIMessage类型）
   2. 是Runnable接口的子类（可以加入链）

![str_output_parser_solution](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/str_output_parser_solution.png)

【代码实现】使用StrOutputParser解析模型输出

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，仅告知名字无需其他内容"
)

# chain = prompt | model | model
# response = chain.invoke({"lastname":"张", "gender":"女儿"})
# 报错：ValueError: Invalid input type <class 'langchain_core.messages.ai.AIMessage'>. Must be a PromptValue, str, or list of BaseMessages.

# 使用 StrOutputParser 转换第1个model的输出AIMessage，StrOutputParser表示AIMessage转为字符串后，作为第2个model的输入字符串
strOutputParser = StrOutputParser()
chain = prompt | model | strOutputParser | model
response = chain.invoke({"lastname":"张", "gender":"女儿"})

print(type(response)) # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content) # 你好！你提到“张若曦”，。。。。。。

# 方式2：不使用 response.content打印输出，而使用 StrOutputParser 做类型转换
print("\n========== 方式2：不使用 response.content打印输出，而使用 StrOutputParser 做类型转换 ")
chain2 = prompt | model | strOutputParser | model | strOutputParser
response2 = chain2.invoke({"lastname":"张", "gender":"女儿"})
print(type(response2)) # <class 'langchain_core.messages.base.TextAccessor'>
print(response2) # 你好！你提到“张若溪”。。。。。。
```

<br>

---

### 【总结】StrOutputParser

1. StrOutputParser 是 langchain内置的简单字符串解析器；
   1. 可以将AIMessage 类型转换为基础字符串；
   2. <font color=red>可以加入chain作为组件存在（因为StrOutputParser是Runnable接口的子类）</font>;

<br>

---

## 【3.19】JsonOutputParser和多模型执行链

### 【3.19.1】JsonOutputParser介绍

1. ```chain = prompt | model | strOutputParser | model``` 这行代码的处理并不常见； 
   1. 因为上一个模型的输出，没有被处理就输入给下一个模型；
2. 正常情况下我们应该有如下处理逻辑：
   1. invoke | stream 初始输入 -> 提示词模板 -> 模型 -> <font color=red>数据处理（新增） -> 提示词模板（新增）</font> -> 模型 -> 解析器 -> 结果  
   2. <font color=red>即上述伪代码的处理逻辑是： 上一个模型的输出，作为提示词模板的输入，构建下一个提示词，再把提示词作为第二个模型的输入</font>； 

<br>

### 【3.19.2】如何对模型输出结果做数据处理

1. 第1次model的输出类型为AIMessage， 而提示词模板的输入类型为dict
2. <font color=red>所以解决办法是：把模型输出的AIMessage类型转为字典dict，然后注入第2个提示词模板中，形成新的提示词（PromptValue对象）</font>;
   1. <font color=red>使用 JsonOutputParser 把AIMessage转为dict </font>；

【PromptTemplate-invoke方法源码】输入类型是dict

```python
def invoke(
    self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
) -> PromptValue:
```

<br>

【代码实现】基于 JsonOutputParser 构建langchain链

【0319_json_output_parser.py】

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max")

first_template = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，并封装到JSON格式返回给我，"
    "要求key是name，value是起的名字。请严格遵守格式要求"
)

second_template = PromptTemplate.from_template(
    "姓名{name}， 请帮我解析含义"
)

# 构建langchain链
chain = first_template | model | json_parser | second_template | model | str_parser

result = chain.invoke({"lastname":"张", "gender":"女儿"})
print(type(result)) # <class 'langchain_core.messages.base.TextAccessor'>
print(result)
# “张婉清”是一个富有诗意和文化内涵的中文姓名 。。。。。。
```

<br>

【0319_stream_json_output_parser.py】基于 JsonOutputParser 构建langchain链进行流式输出

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max")

first_template = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，并封装到JSON格式返回给我，"
    "要求key是name，value是起的名字。请严格遵守格式要求"
)

second_template = PromptTemplate.from_template(
    "姓名{name}， 请帮我解析含义"
)

# 构建langchain链
chain = first_template | model | json_parser | second_template | model | str_parser
# 流式输出调用llm
result = chain.stream({"lastname":"张", "gender":"女儿"})

for chunk in result:
    print(chunk, end="", flush=True)
```

<br>

---

### 【总结】JsonOutputParser

1. 在构建链的时候要注意整体兼容性， 注意前后组件的输入和输出要求； 
   1. 模型输入： PromptValue 或字符串或序列 （BaseMessage, list, tuple, str, dict）
   2. 模型输出：AIMessage； 
   3. 提示词模板输入： 要求是字典 
   4. 提示词模板输出： PromptValue 对象 
   5. StrOutputParser ： AIMessage输入， str输出 
   6. JsonOutputParser： AIMessage输入，dict输出 

![jsom_output_parser](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/jsom_output_parser.png)

<br>

---

## 【3.20】langchain框架RunnableLambda及把自定义函数加入链

### 【3.20.1】RunnableLambda类

1. 构建链代码： 

2. ```python
   # 构建langchain链
   chain = first_template | model | json_parser | second_template | model | str_parser
   ```

3. 问题：

   1. 上述代码我们使用json_parser做了数据处理；
   2. <font color=red>作为可选方案，数据处理还可以自定义函数来实现；自定义函数通过编写RunnableLambda函数来实现</font>； 
   3. <font color=red>优点：RunnableLambda自定义处理函数会更加灵活，格式不限；（而JsonOutputParser要求输入格式是json）</font>;

4. RunnableLambda类是langchain内置的， 将普通函数等转换为 Runnable接口实例， 方便自定义函数加入chain； 

   1. 语法： RunnableLambda(函数对象或lambda匿名函数)

【0319_stream_runnable_lambda_func.py】

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda

str_parser = StrOutputParser()

model = ChatTongyi(model="qwen3-max")

first_template = PromptTemplate.from_template(
    "我邻居姓{lastname}， 刚生了{gender}， 请起名，仅告诉我名字，不需要额外信息"
)

second_template = PromptTemplate.from_template(
    "姓名{name}， 请帮我解析含义"
)

# 使用RunnableLambda类创建自定义函数
my_func = RunnableLambda(lambda ai_msg : {"name":ai_msg.content})

# 基于RunnableLambda函数构建langchain链
chain = first_template | model | my_func | second_template | model | str_parser
# 流式输出调用llm
result = chain.stream({"lastname":"张", "gender":"女儿"})

for chunk in result:
    print(chunk, end="", flush=True) # 当然可以！我们来解析一下“张婉清”这个名字的含义。......
```

<br>

### 【3.20.2】自定义函数直接入链

1. 函数直接入链： 

2. ```python
   chain2 = (first_template | model | RunnableLambda(lambda ai_msg : {"name":ai_msg.content})
             | second_template | model | str_parser)
   ```

2. <font color=red>跳过 RunnableLambda类，直接让函数入链</font>； 
   1. 因为Runnable接口类在重写 _ _ or _ _ 函数时，支持Callable接口的实例； 而函数就是 Callable接口的实例；

【0319_stream_runnable_lambda_func.py】 <font color=red>本质是将函数自动转为 RunnableLambda </font>

```python
# 方式2 ： 直接把RunnableLambda自定义函数加入链
print("\n\n ========== 方式2： 直接把RunnableLambda自定义函数加入链: ")
chain2 = (first_template | model | RunnableLambda(lambda ai_msg : {"name":ai_msg.content})
          | second_template | model | str_parser)
result2 = chain2.invoke({"lastname":"张", "gender":"女儿"}) #
print(result2)
```

<br>

【 Runnable#_ _ or _ _ 函数源码】

```python
def __or__(
    self,
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]:
    """Runnable "or" operator.

    Compose this `Runnable` with another object to create a
    `RunnableSequence`.

    Args:
        other: Another `Runnable` or a `Runnable`-like object.

    Returns:
        A new `Runnable`.
    """
    return RunnableSequence(self, coerce_to_runnable(other))
```

---

### 【3.20.3】总结

1. 如果要在链中加入自定义函数，可以选择：
   1. <font color=red>将函数封装到 RunnableLambda类对象</font>， 实际是 Runnable接口实例， 可以直接入链； 
   2. 直接将函数入链， 函数会自动转换为 RunnableLambda ； 

<br>

---

## 【3.21】langchain框架Memory临时会话记忆

### 【3.21.1】临时记忆

1. 如果想要封装历史记录，除了自行维护历史消息外，也可以借助langchain内置的历史记录附加功能；
2. langchain提供了history功能，帮助模型在有历史记忆的情况下回答：
   1. 基于RunnableWithMessageHistory在原有链的基础上创建带有历史记录功能的新链（新Runnable实例）
   2. 基于InMemoryChatMessageHistory 为历史记录提供内存存储（临时用）

【代码实现-基于临时会话记忆调用大模型】0321_temp_session_memory.py

```python
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
```

【运行结果】

```c++
=================== System: 你需要根据会话历史回答用户问题。会话历史如下：
Human: 请回答如下问题: 小明有2只猫 ====================
第1次执行 小明有2只猫。
==================== System: 你需要根据会话历史回答用户问题。会话历史如下：
Human: 小明有2只猫
AI: 小明有2只猫。
Human: 请回答如下问题: 小刚有1只狗 ====================
第2次执行 小刚有1只狗。
==================== System: 你需要根据会话历史回答用户问题。会话历史如下：
Human: 小明有2只猫
AI: 小明有2只猫。
Human: 小刚有1只狗
AI: 小刚有1只狗。
Human: 请回答如下问题: 总共有几只宠物 ====================
第3次执行 小明有2只猫，小刚有1只狗，所以总共有：

2 + 1 = **3只宠物**。
```

<br>

### 【3.21.2】总结 

1. RunnableWithMessageHistory是langchain内Runnable接口的实现，主要用于：创建一个带有历史记忆功能的Runnable实例（链）； 
2. 它在创建的时候需要提供一个 BaseChatMessageHistory的具体实现（用来存储历史消息）
   1. InMemoryChatMessageHistory ： 可以实现在内存中存储历史；
3. 额外的，如果想要在invoke或stream执行链的同时，把提示词打印出来，可以在链中加入自定义函数，如上述代码的print_prompt 函数； 

<br>

---

## 【3.22】langchain框架Memory长期回话记忆

### 【3.22.1】memory长期回话记忆

1. 问题与解决方法：
   1. 问题： InMemoryChatMessageHistory仅可以在内存中临时保存会话记忆，一旦程序退出，则记忆消失；
      1. InMemoryChatMessageHistory 类继承自 BaseChatMessageHistory ；
   2. 解决方法：
      1. 在官方注释中给出了相关的实现指南， 并给出了<font color=red>基于文件的历史消息（FileChatMessageHistory）</font>存储示例代码； 
      2. 我们可以自定实现一个基于json格式和本地文件的会话数据保存；

<br>

### 【3.22.2】FileChatMessageHistory实现长期回话记忆

1. FileChatMessageHistory类实现的核心思路：基于文件存储会话记录，以 session_id 为文件名， 不同session_id 由不同文件存储消息； 
   1. FileChatMessageHistory继承了 BaseChatMessageHistory ； 
2. 继承 BaseChatMessageHistory 实现如下3个方法：
   1. add_messages: 同步模式， 添加消息； 
   2. messages: 同步模式， 获取消息；  
   3. clear： 同步模式， 清除消息； 

<br>

---

### 【3.22.3】代码实现

【module_0322_consistent_session_memory.py】长期记忆模板

```python
# 持久或长期会话记忆
import json
import os
from collections.abc import Sequence

from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
# message_to_dict: 单个消息对象 (BaseMessage类实例) -> 字典
# message_from_dict： [字典, 字典...] -> [消息, 消息...]
# AIMessage, HumanMessage, SystemMessage, 都是BaseMessage的子类

class DiyFileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id # 会话id
        self.storage_path = storage_path  # 不同会话id的存储文件， 所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_message(self, message:Sequence[BaseMessage]) -> None:
        # Sequence序列 ： 类似list, tuple
        all_messages = list(self.messages) # 已有的消息列表
        all_messages.extend(message) # 新的和已有的融合为一个list

        # 将数据同步写入到本地文件
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将 BaseMessage 消息转为字典（借助 json模块以json字符串写入文件 ）
        new_messages= [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property  # @property 把message方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内： list[字典]
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                message_data = json.load(f) # 返回值就是： list[字典]
                return messages_from_dict(message_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump([], f)
```

【0322_consistent_session_memory_test.py】长期记忆测试案例

```python
# 临时会话记忆
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# RunnableWithMessageHistory 帮助创建一个带有历史消息的新链
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate
from module_0322_consistent_session_memory import DiyFileChatMessageHistory

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

# 实现通过会话id获取 InMemoryChatMessageHistory 类对象
def get_history(session_id):
    return DiyFileChatMessageHistory(session_id, "./chat_history")

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
```

【测试步骤】

- 步骤1：注释第三次执行的代码；仅执行第1次与第2次执行的代码；<font color=red>执行后就有回话历史保存到文件</font>；
- 步骤2：注释第一次与第二次执行的代码；仅执行第3次执行的代码；<font color=red>执行时读取文件中的会话历史，接着执行第三次执行代码</font>；

【补充】代码运行失败，需要调试

<br>

---

## 【3.23】langchain框架组件-Document loaders：文档加载器

### 【3.23.1】Document loaders：文档加载器介绍 

1. <font color=red>文档加载器提供了一套标准接口，用于将不同来源（如csv，pdf或json等）的数据读取为langchain的文档格式</font>。这确保了无论数据来源如何，都能对齐进行一致性处理； 
2. 文档加载器（内置或自行实现）：需要实现BaseLoader接口； 
3. <font color=red>Class Document：是langchain内文档的统一载体， 所有文档加载器最终返回此类的实例</font>； 
4. 一个基础的document类实例， 基于如下代码创建：

```python
from langchain_core.documents import Document

document = Document(
    page_content = "hello world", metadata={"source":"https://example.com"}
)
```

5. 可以看到： Document类其核心记录了：
   1. page_content: 文档内容
   2. metadata： 文档元数据（字典）

<br>

### 【3.23.2】Document loaders加载文件的不同方法

1. 不同文档加载器可能定义了不同参数， 但其都实现了统一的接口（方法）：
   1. load()： 一次性加载全部文档； 
   2. lazy_load() : 延迟流式传输文档，对大型数据集很有用， 避免内存溢出； 

【例】CSVLoader的使用

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    ... # 初始化参数 
)

# 一次性加载全部文档
documents = loader.load()

# 对于大数据集， 分段返回文档 
for document in loader.lazy_load():
  print(document)
```

<br>

### 【3.23.3】CSVLoader-csv加载器

1. langchain内置了许多文档加载器，官方文档：[https://docs.langchain.com/oss/python/integrations/document_loaders](https://docs.langchain.com/oss/python/integrations/document_loaders)
2. 我们简单学习如下几个常用的文档加载器：
   1. CSVLoader
   2. JSONLoader
   3. PDFLoader

【例】CSVLoader测试代码

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    ...  # Integration-specific parameters here
)

# Load all documents
documents = loader.load()

# For large datasets, lazily load documents
for document in loader.lazy_load():
    print(document)

```

<br>

---

### 【3.23.4】langchain-csv加载器-代码实现

【test_0323_csvloader.py】

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="../data/stu.csv",
    csv_args={
        "delimiter" : ",", # 指定分隔符
        "quotechar": '"', # 指定带有分隔符文本的引号是单引号还是双引号
        "fieldnames": ['a', 'b', 'c', 'd'], # 或有，指定表头(但原文件的第一行的表头会被当做数据处理)
    },
    encoding="utf-8"  # 指定编码为utf-8
)

# 方式1：批量加载： .load() -> [Document, Document, ...]
documents = loader.load()

for document in documents:
    print("="*20)
    print(type(document), document)

# 方式2： 懒加载 .lazy_load()  迭代器[Document]
print("\n\n", "="*20, "方式2： 懒加载")
for document in loader.lazy_load():
    print("=" * 20)
    print(document)
```

【运行结果】

```c++
===================
<class 'langchain_core.documents.base.Document'> page_content='a: name
b: age
c: gender
d: hobby' metadata={'source': '../data/stu.csv', 'row': 0}
====================
<class 'langchain_core.documents.base.Document'> page_content='a: 张三01
b: 21
c: 男
d: 吃饭1,rap' metadata={'source': '../data/stu.csv', 'row': 1}
====================
<class 'langchain_core.documents.base.Document'> page_content='a: 张三02
b: 22
c: 男
d: 吃饭2,rap' metadata={'source': '../data/stu.csv', 'row': 2}
====================
<class 'langchain_core.documents.base.Document'> page_content='a: 张三03
b: 23
c: 女
d: 吃饭3,rap' metadata={'source': '../data/stu.csv', 'row': 3}


 ==================== 方式2： 懒加载
====================
page_content='a: name
b: age
c: gender
d: hobby' metadata={'source': '../data/stu.csv', 'row': 0}
====================
page_content='a: 张三01
b: 21
c: 男
d: 吃饭1,rap' metadata={'source': '../data/stu.csv', 'row': 1}
====================
page_content='a: 张三02
b: 22
c: 男
d: 吃饭2,rap' metadata={'source': '../data/stu.csv', 'row': 2}
====================
page_content='a: 张三03
b: 23
c: 女
d: 吃饭3,rap' metadata={'source': '../data/stu.csv', 'row': 3}
```

<br>

### 【3.23.5】总结 

1. langchain内置了许多种类的文档加载器：
   1. 文档加载器均继承于 BaseLoader类； 
   2. 返回Document类型的对象； 
   3. load方法一次性批量加载（返回list内含Document对象）， 如内容过多可能list太大，出现内存溢出问题； 
   4. lazy_load()方法会得到迭代器对象， 可用于for循环依次获取单个Document对象，适用于大文档避免内存存不下的情况；
2. CSVLoader用于加载csv文件， 加载成功后得到的即 Document对象； 

<br>

---

## 【3.24】langchain组件：JSONLoader

### 【3.24.1】JSONLoader介绍

1. JSONLoader：用于把json数据加载为Document类对象；<font color=red>使用jsonloader需要额外安装 pip install jq </font>;
2. jq 是一个跨平台的json解析工具， langchain底层对json解析就是基于jq工具实现的； 
   1. 将json数据的信息抽取出来， 封装为Document对象，抽取的时候依赖 jq_schema 语法； 

![json_loader_jq](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/json_loader_jq.png)

3. JSONLoader-代码示例 

![jsonloder_ex](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/jsonloder_ex.png)

<br>

### 【3.24.2】jsonloader代码实现

【test0323_json_loader.py】

```python
from langchain_community.document_loaders import JSONLoader

print("========== 【案例1】 使用jq schema抽取json文件 ")
loader = JSONLoader(
    file_path="../data/stu.json",
    # jq_schema=".name",
    # jq_schema=".other.addr"
    jq_schema=".",  # 抽取整个json文件
    text_content=False, # 告知JSONLoader，抽取的内容不是字符串
)
document = loader.load()
print(document)

print("========== 【案例2】使用jq schema抽取json 列表 文件 ")
loader = JSONLoader(
     file_path="../data/stu_list.json",
    jq_schema=".[].name", # 仅抽取数组的name属性
    text_content=False,  # 告知JSONLoader，抽取的内容不是字符串
)
document = loader.load()
print(document)

print("==========【案例3】 使用jq schema抽取json_lines 文件 ")
loader = JSONLoader(
    file_path="../data/json_line_stu_list.json",
    jq_schema=".name", # 仅抽取数组的name属性
    text_content=False,  # 告知JSONLoader，抽取的内容不是字符串
    json_lines=True   # 告知JSONLoader， 这是一个jsonlines文件（每一行都是一个标准的json对象）
)
document = loader.load()
print(document)
```

【运行结果】

```c++
	========== 【案例1】 使用jq schema抽取json文件 
[Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/stu.json', 'seq_num': 1}, page_content='{"name": "\\u5f20\\u4e09", "age": 11, "hobby": ["\\u5531\\u6b4c", "\\u8df3\\u821e", "rap"], "other": {"addr": "\\u6210\\u90fd", "tel": "123456"}}')]

========== 【案例2】使用jq schema抽取json 列表 文件 
[Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/stu_list.json', 'seq_num': 1}, page_content='张三01'), Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/stu_list.json', 'seq_num': 2}, page_content='张三02'), Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/stu_list.json', 'seq_num': 3}, page_content='张三03')]

==========【案例3】 使用jq schema抽取json_lines 文件 
[Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/json_line_stu_list.json', 'seq_num': 1}, page_content='张三01'), Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/json_line_stu_list.json', 'seq_num': 2}, page_content='张三02'), Document(metadata={'source': '/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/data/json_line_stu_list.json', 'seq_num': 3}, page_content='张三03')]

```

<br>

---

### 【3.24.3】总结 

1. JSONLoader依赖jq库， 通过 pip install jq 安装（或 poetry add jq ）：
   1. <font color=red>JSONLoader 使用jq的解析语法，场景如下 </font>：
      1. 点"." 表示根， [] 表示数组；  
      2. .name 表示根取name的值；  
      3. hobby[1] 表示取hobby数组的第2个元素； 
      4. .[] 表示将数组内的每个字典(json对象) 都获取到 
      5. .[].name 表示获取数组内每个字典（json对象）的name对应的值；  
2. JSONLoader 初始化有4个主要参数：
   1. file_path: 文件路径，必填；  
   2. jq_schema: jq解析语法， 必填；  
   3. text_context: 抽取到的是否是字符串， 默认为True，非必填；  
   4. json_lines： 是否为JsonLines文件， 默认为False， 非必填； 
      1. JSONLines文件： 每一行都是一个独立的字典（json对象）

<br>

---

## 【3.25】TextLoader与文档分割器

### 【3.25.1】TextLoader文档加载器（读取文本文件）

1. TextLoader: 读取文本文件（如.txt），将全部内容放入一个Document对象中； 
2. 问题：把所有内容都放入一个Document对象，<font color=red>若文档很大，则加载到一个Document对象中是否不太合适</font>？
   1. <font color=red>解决方法： 使用文档分割器-RecursiveCharacterTextSplitter </font>;

### 【3.25.2】文档分割器-RecursiveCharacterTextSplitter 

1. <font color=red>文档分割器-RecursiveCharacterTextSplitter ，递归字符文本分割器，主要用于按照自然段落分割大文档</font>； 是langchain官方推荐的默认字符分割器； 它在保持上下文完整性和控制片段大小之间实现了良好平衡，开箱即用效果佳；

2. 代码示例：

![textloader_recursiveCharaterTextSplitter](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/textloader_recursiveCharaterTextSplitter.png)

### 【3.25.3】代码实现 

【test_0325_text_loader.py】文本加载与分割

```python
from langchain_community.document_loaders import  TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 【案例1】使用TextLoader读取文本文件
print("====== 【案例1】 使用TextLoader读取文本文件：")
loader = TextLoader(
    file_path="../data/python_base_syntax.txt",
)

documents = loader.load()
# print(documents)
# print(len(documents)) # 1

# 【案例2】使用 RecursiveCharacterTextSplitter 分割字符
print("========== 【案例2】使用 RecursiveCharacterTextSplitter 分割字符" )
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 分段的最大字符数
    chunk_overlap=50,  # 分段之间允许的重叠字符数
    separators=["\n\n", "\n", "!", " ", "!"], # 文本自然段落分割的依据符号
    length_function=len # 统计字符的依据函数
)

# 分割文本
split_docs = splitter.split_documents(documents)
print(len(split_docs)) # 18 
loop_time = 0
for doc in split_docs:
    loop_time = loop_time + 1
    print("="*20, "第" + str(loop_time), "个段落")
    print(doc)
    print("=" * 20)
```

【运行结果】

```c++
===== 【案例1】 使用TextLoader读取文本文件：
========== 【案例2】使用 RecursiveCharacterTextSplitter 分割字符
18
==================== 第1 个段落
page_content='##!/usr/bin/env python3
......
```

<br>

### 【3.25.4】总结

1. TextLoadert： 是一个简单的加载器，可以加载文本文件内容， 返回仅有一个Document对象的list；
2. RecursiveCharacterTextSplitter： 递归字符文本分割器， 是langchain官方推荐的默认分割器； 
   1. 基于文本的自然段落分割大文档为 小文档； 
   2. 可以指定小文档的最大字符数， 重叠字符数；  
   3. 可以手动指定段落划分的依据（符号），以及字符数量统计函数； 

![pypdf_ex](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/pypdf_ex.png)

<br>

---

## 【3.26】PyPDFLoader-pdf文件加载器 

1. langchain内支持许多pdf加载器，我们选择其中的PyPDFLoader加载器； 
2. PyPDFLoader加载器， 依赖PyPDF库，所以需要事先安装PyPDF； poetry add pypdf 

### 【3.26.1】代码实现 

【test_0326_pypdfloader.py】

```python
# PyPDFLoader-pdf文件加载器
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="../data/python_syntax.pdf",
    # mode="page" # 默认是page模式，每个页面形成一个document文档对象
    mode = "single",  # 不管多少页，只返回一个document对象
    # password="123455"  # pdf文件打开密码
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("=" * 20, f"第{str(i)}个段落")
```

【运行结果】

```c++
page_content='#!/usr/bin/env python3
-- coding: utf-8 --  
"""
Python基础语法示例
展示了Python的核⼼语法特性
"""
import sys
import math
from typing import List, Dict, Optional
==================== 1. 
注释 
====================
 
这是单⾏注释  
"""
这是多⾏注释
可以⽤来写⽂档字符串
"""
==================== 2.' metadata={'producer': 'macOS 版本15.1（版号24B2082） Quartz PDFContext', 'creator': 'Typora', 'creationdate': "D:20260322031716Z00'00'", 'moddate': "D:20260322031716Z00'00'", 'source': '../data/python_syntax.pdf', 'total_pages': 17, 'page': 0, 'page_label': '1'}
==================== 第1个段落
page_content='==================== 2. 
变量和数据类型 
====================
 
print("=" * 50)
print("2. 变量和数据类型")
print("=" * 50)
数字类型  
integer_var = 42              # 整数
float_var = 3.14159           # 浮点数
complex_var = 3 + 4j          # 复数
boolean_var = True            # 布尔值
print(f"整数: {integer_var}, 类型: {type(integer_var)}")
print(f"浮点数: {float_var}, 类型: {type(float_var)}")
print(f"复数: {complex_var}, 类型: {type(complex_var)}")
print(f"布尔值: {boolean_var}, 类型: {type(boolean_var)}")
字符串  
string_var = "Hello, Python!"
multi_line_string = """这是
多⾏
字符串"""
print(f"字符串: {string_var}")
print(f"字符串⻓度: {len(string_var)}")
print(f"字符串切⽚: {string_var[0:5]}")' metadata={'producer': 'macOS 版本15.1（版号24B2082） Quartz PDFContext', 'creator': 'Typora', 'creationdate': "D:20260322031716Z00'00'", 'moddate': "D:20260322031716Z00'00'", 'source': '../data/python_syntax.pdf', 'total_pages': 17, 'page': 1, 'page_label': '2'}
==================== 第2个段落
  .......
```

<br>

---

## 【3.27】VectorStores向量存储

### 【3.27.1】Vector Stores 向量存储

1. 基于langchain的向量存储，存储嵌入数据，并执行相似性搜索；

![rag_flow](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_flow.png)

2. <font color=red>如上图，这是一个典型的向量存储应用，也是典型的rag流程（这张rag图非常重要）</font>；
3. 这部分开发主要涉及到：
   1. 如何文本转向量； 
   2. 创建向量存储，基于向量存储完成：
      1. 存入向量； 
      2. 删除向量； 
      3. 向量检索；
   3. langchain为向量存储提供了统一接口：
      1. add_documents ; 
      2. delete
      3. Similarity_search 
4. 向量存储的代码示例：

![vector_stores](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/vector_stores.png)

<br>

### 【3.27.2】向量存储代码实现

【案例1-内存向量存储-InMemoryVectorStore】test_0327_memory_vector_store.py

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# 创建内存向量存储对象（内存数据库）
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(),
)

loader = CSVLoader(
    file_path="../data/info.csv",
    encoding="utf-8",
    source_column="source", # 指定本条数据的来源
)

documents = loader.load()
print(documents[0])
print(documents[1])
# page_content='source: 百度
# info: python是世界上最好的编程语言' metadata={'source': '百度', 'row': 0}
# page_content='source: 必应
# info: python学起来很简单' metadata={'source': '必应', 'row': 1}

# 向量存储的 新增，删除，检索
vector_store.add_documents(
    documents=documents, # 被添加的文档，类型：list[Document]
    ids=["id" + str(i) for i in range(1, len(documents)+1)] # 给添加的文档提供id（字符串） list[str]
)

# 删除 传入[id, id...]
vector_store.delete(["id1", "id2"])

# 检索
print("\n\n\n========== 检索 ")
result = vector_store.similarity_search(
    "python是不是简单易学",
    3, # 检索出几条最相似的结果
)
print(result)
```

【运行结果】

```c++
[Document(id='id3', metadata={'source': '百度', 'row': 2}, page_content='source: 百度\ninfo: langchain极大方便了模型开发'), Document(id='id4', metadata={'source': '必应', 'row': 3}, page_content='source: 必应\ninfo: 如何快速减肥'), Document(id='id5', metadata={'source': '百度', 'row': 4}, page_content='source: 百度\ninfo: 明天晚上吃啥子')]
```

<br>

【案例2-向量数据库存储-Chroma】test_0327_consistent_vector_store.py

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

# Chroma 向量数据库（轻量级的）
# 确保 langchain-chroma chromadb 这两个库安装了的

# 创建内存向量存储对象（内存数据库）
vector_store = Chroma(
    collection_name="test", # 类似于数据库表名
    embedding_function=DashScopeEmbeddings(), # 提供嵌入模型
    persist_directory="./chroma_db", # 指定数据存放的文件夹
)

loader = CSVLoader(
    file_path="../data/info.csv",
    encoding="utf-8",
    source_column="source", # 指定本条数据的来源
)

documents = loader.load()
print(documents[0])
print(documents[1])
# page_content='source: 百度
# info: python是世界上最好的编程语言' metadata={'source': '百度', 'row': 0}
# page_content='source: 必应
# info: python学起来很简单' metadata={'source': '必应', 'row': 1}

# 向量存储的 新增，删除，检索
vector_store.add_documents(
    documents=documents, # 被添加的文档，类型：list[Document]
    ids=["id" + str(i) for i in range(1, len(documents)+1)] # 给添加的文档提供id（字符串） list[str]
)

# 删除 传入[id, id...]
vector_store.delete(["id1", "id2"])

# 检索
print("\n\n\n========== 检索 ")
result = vector_store.similarity_search(
    "python是不是简单易学",
    3 # 检索出几条最相似的结果
)
print(result)
```

【运行结果】

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

# Chroma 向量数据库（轻量级的）
# 确保 langchain-chroma chromadb 这两个库安装了的

# 创建内存向量存储对象（内存数据库）
vector_store = Chroma(
    collection_name="test", # 类似于数据库表名
    embedding_function=DashScopeEmbeddings(), # 提供嵌入模型
    persist_directory="./chroma_db", # 指定数据存放的文件夹
)

loader = CSVLoader(
    file_path="../data/info.csv",
    encoding="utf-8",
    source_column="source", # 指定本条数据的来源
)

documents = loader.load()
print(documents[0])
print(documents[1])
# page_content='source: 百度
# info: python是世界上最好的编程语言' metadata={'source': '百度', 'row': 0}
# page_content='source: 必应
# info: python学起来很简单' metadata={'source': '必应', 'row': 1}

# 向量存储的 新增，删除，检索
vector_store.add_documents(
    documents=documents, # 被添加的文档，类型：list[Document]
    ids=["id" + str(i) for i in range(1, len(documents)+1)] # 给添加的文档提供id（字符串） list[str]
)

# 删除 传入[id, id...]
vector_store.delete(["id1", "id2"])

# 检索
print("\n\n\n========== 检索 ")
result = vector_store.similarity_search(
    "python是不是简单易学",
    3 # 检索出几条最相似的结果
)
print(result)
```

<br>

【案例3-向量数据库存储+仅检索】test_0327_only_search_consistent_vector_store.py

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

# Chroma 向量数据库（轻量级的）
# 确保 langchain-chroma chromadb 这两个库安装了的

# 创建内存向量存储对象（内存数据库）
vector_store = Chroma(
    collection_name="test", # 类似于数据库表名
    embedding_function=DashScopeEmbeddings(), # 提供嵌入模型
    persist_directory="./chroma_db", # 指定数据存放的文件夹
)

# 删除保存文档到chroma向量数据库的代码，仅保留检索代码，如下。因为文档嵌入后已经被持久化到chroma向量数据库中。

# 检索
print("\n\n\n========== 检索 ")
result = vector_store.similarity_search(
    "python是不是简单易学",
    3, # 检索出几条最相似的结果
    filter={"source":"百度"} # 或有，设置过滤条件
)
print(result)
```

<br>

### 【3.27.3】总结

1. langchain内部提供了向量存储功能，可以基于：
   1. InMemoryVectorStore, 完成内存向量存储； 
   2. Chroma， 外部数据库向量存储；  
2. 向量存储类均提供了3个通用API接口：
   1. add_document: 添加文档到向量存储；  
   2. delete, 从向量存储中删除文档 
   3. Similarity_serach: 相似性搜索； 

<br>

---

## 【3.28】langchain检索向量并构建提示词 

### 【3.28.1】代码实现

【test_0328_retrieve_vector.py】检索向量

```python
"""
提示词：用户的提问 + 向量库中检索到的参考资料
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料：{context}"),
        ("user", "用户提问：{input}")
    ]
)

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 准备一下资料（向量库的数据）
# add_texts 传入一个list[str]
vector_store.add_texts(
    [
        "减肥就是要少吃多练",
        "在减脂期间吃东西很重要，清淡少油控制卡路里舍摄入并运动起来",
        "跑步是很好的运动哦"
    ]
)
input_text = "怎么减肥？"


# 检索向量库
result = vector_store.similarity_search(input_text, 2)
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"

# 打印参考资料
print("参考资料=", reference_text)

def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# 创建 chain对象
chain = prompt | print_prompt | model | StrOutputParser()
invoke_result = chain.invoke({"input": input_text, "context":reference_text})
print(invoke_result)
```

【运行结果】

```c++
参考资料= [减肥就是要少吃多练在减脂期间吃东西很重要，清淡少油控制卡路里舍摄入并运动起来]
System: 以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料：[减肥就是要少吃多练在减脂期间吃东西很重要，清淡少油控制卡路里舍摄入并运动起来]
Human: 用户提问：怎么减肥？
====================
减肥的关键在于“少吃多练”：  
1. **饮食方面**：选择清淡、少油的食物，严格控制每日热量摄入；  
2. **运动方面**：坚持规律运动，增加热量消耗。  

通过合理控制饮食与积极运动相结合，才能有效减脂。
```

<br>

### 【3.28.2】总结 

1. 向量存储的实例， 通过 add_texts(list[str]) 方法可以快速添加到向量存储中； 
2. 流程：
   1. 先通过向量存储检索匹配信息；
   2. 将用户提问和匹配信息一同封装到提示词模版中提问模型； 

<br>

---

## 【3.29】langchain框架的RunnablePassthrough的使用

### 【3.29.1】RunnablePassthrough的使用

【检索向量库代码回顾】

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料：{context}"),
        ("user", "用户提问：{input}")
    ]
)

# 准备一下资料（向量库的数据）
# add_texts 传入一个list[str]
vector_store.add_texts(
    [
        "减肥就是要少吃多练",
        "在减脂期间吃东西很重要，清淡少油控制卡路里舍摄入并运动起来",
        "跑步是很好的运动哦"
    ]
)
input_text = "怎么减肥？"

# 检索向量库，生成提示词 
result = vector_store.similarity_search(input_text, 2)
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"

# 打印参考资料
print("参考资料=", reference_text)

def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# 创建 chain对象
chain = prompt | print_prompt | model | StrOutputParser()
# 生成的提示词，用于调用llm
invoke_result = chain.invoke({"input": input_text, "context":reference_text})
print(invoke_result)
```

【问题】是否可以把向量检索加入到langchain链中； 即把vector_store.similarity_search(..)调用过程加入到链中； 

- 使用 RunnablePassthrough 类来实现； 

<br>

---

### 【3.29.2】RunnablePassthrough代码实现向量检索入链

【test_0329_runnable_passthrough_baseuse.py】RunnablePassthrough基本使用 

```python
# RunnablePassthrough代码实现向量检索入链
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料：{context}"),
        ("user", "用户提问：{input}")
    ]
)

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 准备一下资料（向量库的数据）
# add_texts 传入一个list[str]
vector_store.add_texts(
    [
        "减肥就是要少吃多练",
        "在减脂期间吃东西很重要，清淡少油控制卡路里舍摄入并运动起来",
        "跑步是很好的运动哦"
    ]
)
input_text = "怎么减肥？"

# langchain中向量存储对象，有一个方法：as_retriever，可以返回一个Runnable接口的子类实例对象
# retriever 是Runnable接口的子类对象，它就可以入链
retriever = vector_store.as_retriever(search_kwargs={"k":2})

def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# 自定义格式化函数
def format_func(docs):
    if not docs:
        return "无相关参考资料"
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    return formatted_str + "]"

# 创建chain
# chain = retriever | prompt | model | StrOutputParser()
chain = ( {"input": RunnablePassthrough(), "context": retriever | format_func}
         | prompt | print_prompt | model | StrOutputParser() )
"""
retriever: 
    - 输入： 用户提问        str
    - 输出： 向量库的检索结果 list[Document]
prompt: 
    - 输入： 用户提问 + 向量库的检索结果  dict
    - 输出： 完整的提示词               PromptValue 
"""
result = chain.invoke(input_text)
print("大模型回复结果=====\n", result)
```

【运行结果】

```c++
System: 以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料：[减肥就是要少吃多练在减脂期间吃东西很重要，清淡少油控制卡路里舍摄入并运动起来]
Human: 用户提问：怎么减肥？
====================
大模型回复结果=====
 减肥的关键在于“少吃多练”：  
1. **饮食控制**：选择清淡、少油的食物，严格控制每日卡路里摄入；  
2. **坚持运动**：通过规律锻炼增加热量消耗，促进脂肪燃烧。  
```

<br>

---

# 【4】rag实战项目

## 【4.1】rag项目案例介绍

### 【4.1.1】rag回顾

1. <font color=red>rag即检索，增强，生成，其主要分为2条线，包括离线处理+在线处理</font>：
   1. 离线处理：向私有知识库（向量存储）源源不断添加私有知识文档：
      1. 向知识库添加来自未来的知识文档（基于模型训练完成时间）；
      2. 向模型添加私有知识文档；
      3. 给出模型参考资料，规避模型幻觉（一本正经的胡说八道）
   2. 在线处理：用户提问会基于私有知识库做检索，获取参考资料，同步组装新提示词询问大模型获取结果；

![rag_review](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_review.png)

<br>

### 【4.1.2】项目需求和思路

1. 本次项目以某东商品衣服为例，以衣服属性构建本地知识。
2. 使用者可以<font color=red>自由更新</font>本地知识，用户问题的答案也是<font color=red>基于本地知识</font>生成的；

![rag_online_offline2](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_online_offline2.png)

3. 项目主要实现如下代码文件：

![rag_proj_files](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_proj_files.png)

<br>
---

## 【4.2】rag项目-文本上传web服务

### 【4.2.1】项目需求和思路

![rag_web_upload](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/rag_web_upload.png)

<br>

### 【4.2.2】基于Streamlit 完成web网页上传服务

【app_file_uploader.py】

```python
"""
基于Streamlit 完成web网页上传服务

pip install streamlit
poetry add streamlit
"""

import streamlit as st

# 添加网页标题
st.title("知识库更新服务")

# file_uploader 添加所需文件上传服务
upload_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False, # 仅接受单文件上传
)
if upload_file is not None:
    # 提取文件信息
    file_name = upload_file.name
    file_type = upload_file.type
    file_size = upload_file.size/1024 # KB

    st.subheader(f"文件名:{file_name}")
    st.write(f"格式：{file_type}, 大小：{file_size:.2f} KB")

    # 获取文件内容：get_value -> bytes -> decode('utf-8')
    text = upload_file.getvalue().decode("utf-8")
    st.write(text)

# 命令行运行：  streamlit run app_file_uploader.py 打开浏览器查看页面效果
```

运行效果：

![file_upload_page](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/file_upload_page.png)

<br>

---

## 【4.3】rag项目-md5工具函数开发

【代码实现】knowledge_base.py

```python
"""
知识库
"""

import os
import config_data as config
import hashlib

def check_md5(md5_str: str):
    """
        检查传入的md5字符串是否已经被处理过
        False-没有被处理过, True-已处理过
    """
    if not os.path.exists(config.md5_path):
        # 若文件不存在，那肯定没处理过这个md值
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding="utf-8").readlines():
            line = line.strip() # 处理字符串前后的空格和回车
            if line == md5_str:
                return True # 已处理过
        return False

def save_md5(md5_str: str):
    """将传入的md5字符串记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串=转换为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding = encoding)

    # 创建md5对象
    md5_obj = hashlib.md5() # 得到md5 对象
    md5_obj.update(str_bytes)  # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest() # 得到md5的16进制字符串

    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        self.chroma = None # 向量存储的实力 Chroma向量数据库
        self.spliter = None # 文本分割器对象

    def upload_by_str(self, data, filename):
        """将传入的字符串"""

# md5是加签算法，无论字符串多长，都能够得到固定长度（如32位）的16进制字符串
if __name__ == '__main__':
    r1 = get_string_md5("张三01")
    r2 = get_string_md5("张三01")
    r3 = get_string_md5("张三03")

    print(r1)
    print(r2)
    print(r3)

    print("保存并检查md5字符串")
    save_md5(r1)
    print(check_md5(r1))
```

【运行结果】

```c++
27a37907cd6ba6d44fd57c8a84c6f8f1
27a37907cd6ba6d44fd57c8a84c6f8f1
eb280b4a9b184bd96c9b96737e482a4d
保存并检查md5字符串
True
```

<br>

---

## 【4.4】rag项目-知识库更新（添加到向量库）

【更新知识库-向量库】update_knowledge_base.py

```python
"""
更新知识库
"""

import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str: str):
    """
        检查传入的md5字符串是否已经被处理过
        False-没有被处理过, True-已处理过
    """
    if not os.path.exists(config.md5_path):
        # 若文件不存在，那肯定没处理过这个md值
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding="utf-8").readlines():
            line = line.strip() # 处理字符串前后的空格和回车
            if line == md5_str:
                return True # 已处理过
        return False

def save_md5(md5_str: str):
    """将传入的md5字符串记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串=转换为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding = encoding)

    # 创建md5对象
    md5_obj = hashlib.md5() # 得到md5 对象
    md5_obj.update(str_bytes)  # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest() # 得到md5的16进制字符串

    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        # 若文件夹不存在则创建，否则跳过
        os.makedirs(config.persist_directory, exist_ok = True)

        self.chroma = Chroma(
            collection_name=config.collection_name, # 数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory, # 数据库本地存储文件夹
        ) # 向量存储的实力 Chroma向量数据库
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, # 分割后的文本段的最大长度
            chunk_overlap = config.chunk_overlap, # 连续文本段之间允许重复的字符数量
            separators=config.separators, # 自然段落划分的符号
            length_function=len,  # 使用python自带的len函数做长度统计的依据
        ) # 文本分割器对象

    def upload_by_str(self, data, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data, encoding = "utf-8")

        if check_md5(md5_hex):
            return "跳过，内容已存在知识库中"
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 添加到向量库
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"tom"
        }
        # 内存加载到向量库
        self.chroma.add_texts(
            # iterable -> list \ tuple
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks ],
        )

        # 记录已经处理过的数据，支持幂等
        save_md5(md5_hex)

        return "内容成功更新到rag向量库"

# md5是加签算法，无论字符串多长，都能够得到固定长度（如32位）的16进制字符串
if __name__ == '__main__':
    service = KnowledgeBaseService()
    result = service.upload_by_str("李四03", "testfile")
    print(result)
```

<br>

---

## 【4.5】rag项目-完成离线流程开发

【app_file_uploader_streamlit_session_state.py】基于streamlit session state实现文件上传

```python
"""
基于Streamlit 完成web网页上传服务

pip install streamlit
poetry add streamlit
"""
import time

import streamlit as st

from poetry_demo.heima_rag.ragproj.update_knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")

# file_uploader 添加所需文件上传服务
upload_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False, # 仅接受单文件上传
)

# session_state就是一个字典
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if upload_file is not None:
    # 提取文件信息
    file_name = upload_file.name
    file_type = upload_file.type
    file_size = upload_file.size/1024 # KB

    st.subheader(f"文件名:{file_name}")
    st.write(f"格式：{file_type}, 大小：{file_size:.2f} KB")

    # 获取文件内容：get_value -> bytes -> decode('utf-8')
    text = upload_file.getvalue().decode("utf-8")
    # st.write(text) # 替换为 KnowledgeBaseService

    # 在spinner内的代码执行过程中，会有一个转圈动画
    with st.spinner("载入知识库中"):
        time.sleep(1) 
        upload_result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(upload_result) # 在页面直接看到结果

    st.session_state["counter"] +=1

# 命令行运行：  streamlit run app_file_uploader_streamlit_session_state.py 打开浏览器查看页面效果

print(f'上传了{st.session_state["counter"]}个文件')
```

【上传效果】

![streamlit_session_state_upload](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/streamlit_session_state_upload.png)

<br>

---

## 【4.6】rag项目-在线流程向量存储服务

### 【4.6.1】在线流程向量存储思路

![online_vector_store](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/img/online_vector_store.png)

<br>

【向量存储服务】vector_stores.py

```python
# 向量存储服务
from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器， 方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={
            "k":config.similarity_threshold
        })

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
```

<br>

---

## 【4.7】rag项目-rag服务核心代码

【rag.py】检索增强生成-简单版 

```python
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self):
        # 向量服务：用于检索
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
            collection_name=config.collection_size_recommend,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}"),
                ("user", "请回答用户提问:{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.chain = self.__get_chain()

    def __get_chain(self):
        # 获取检索器对象
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str  =""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据: {doc.metadata}\n\n"

            return formatted_str

        chain = (
            {
                "input": RunnablePassthrough(),
                "context":retriever | format_document
            } | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )
        return chain

if __name__ == "__main__":
    result = RagService().chain.invoke("我体重180斤，尺码推荐")
    print(result)

```

<br>

【<font color=red>模型回复效果</font>】

```c++
====================
System: 以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：文档片段:身高：155-165cm, 体重75-95斤，建议尺码S.
身高：160-170cm, 体重90-115斤，建议尺码M.
身高：165-175cm, 体重115-135斤，建议尺码L.
身高：170-178cm, 体重130-150斤，建议尺码XL.
身高：175-182cm, 体重145-165斤，建议尺码2XL.
身高：178-185cm, 体重160-180斤，建议尺码3XL.
身高：180-190cm, 体重180-210斤，建议尺码4XL.
身高：190cm+，建议尺码5XL.
文档元数据: {'create_time': '2026-03-29 10:01:18', 'source': '尺码推荐.txt', 'operator': 'tom'}


Human: 请回答用户提问:我体重180斤，尺码推荐
====================
根据您提供的体重180斤，结合参考资料中的尺码推荐：

- 若您的身高在180–190cm之间，建议选择 **4XL**；
- 若您的身高在178–185cm之间，也可考虑 **3XL**，但接近上限。

为更准确推荐，请提供您的身高信息。若身高≥190cm，则建议选择 **5XL**。
```



---

## 【4.8】rag项目-历史会话记录功能的实现

### 【4.8.1】带有历史会话记录增强的chain实现

【with_memory_rag.py】

```python
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from poetry_demo.heima_rag.ragproj.module_rag_consistent_session_memory import get_history

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self):
        # 向量服务：用于检索
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
            collection_name=config.collection_size_recommend,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问:{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.chain = self.__get_chain()

    def __get_chain(self):
        # 获取检索器对象
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str  =""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据: {doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            print("---------", value)
            return value["input"]

        def format_for_prompt_template(value):
            # 拼接为字典 {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        # 创建历史会话记忆增强的链
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

if __name__ == "__main__":
    # session_id 配置（会话id）
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }

    result = RagService().chain.invoke({"input":"我体重180斤，尺码推荐"}, session_config)
    print(result)
```

<font color=red>第1次提问： 我体重180斤，尺码推荐</font>；

大模型回复：

```c++
--------- {'input': '我体重180斤，尺码推荐', 'history': []}
====================
System: 以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：文档片段:身高：155-165cm, 体重75-95斤，建议尺码S.
身高：160-170cm, 体重90-115斤，建议尺码M.
身高：165-175cm, 体重115-135斤，建议尺码L.
身高：170-178cm, 体重130-150斤，建议尺码XL.
身高：175-182cm, 体重145-165斤，建议尺码2XL.
身高：178-185cm, 体重160-180斤，建议尺码3XL.
身高：180-190cm, 体重180-210斤，建议尺码4XL.
身高：190cm+，建议尺码5XL.
文档元数据: {'source': '尺码推荐.txt', 'create_time': '2026-03-29 10:01:18', 'operator': 'tom'}


System: 并且我提供用户的对话历史记录，如下：
Human: 请回答用户提问:我体重180斤，尺码推荐
====================
Error in RootListenersTracer.on_chain_end callback: AttributeError("'tuple' object has no attribute 'type'")
根据您提供的体重180斤，结合参考资料中的尺码推荐：

- 体重180-210斤，对应建议尺码为 **4XL**。

因此，推荐您选择 **4XL** 尺码。
```

<br>

<font color=red>第2次提问： 春天穿什么颜色的衣服</font>；

```python
if __name__ == "__main__":
    # session_id 配置（会话id）
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }

    result = RagService().chain.invoke({"input":"春天穿什么颜色衣服"}, session_config)
    print(result)
```

<br>

---

## 【4.9】rag项目-聊天页面开发





























