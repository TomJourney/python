[TOC]



# 【README】

1. 本文总结自： 《从LLM到Agent Skill，马克的技术工作坊》
2. 本文会讲解：LLM， Token， Context， Prompt， Agent， Agent Skill， MCP， Tool等概念，及它们的相互联系；

<br>

---

# 【1】LLM-大模型

1. <font color=red>基本上所有的大模型都是通过 Transformer架构训练出来的</font>； 
   1. Transformer架构最早是google提出来的，对应的论文名字为：<font color=red>《Attentions Is All You Need》</font>；
2. 大模型是如何工作的？<font color=red>大模型本质上就是一个文字接龙游戏</font>； 

<br>

---

# 【2】Token

1. 马克的视频怎么样？ 
   1. 编码：
      1. 切分：切分为4个token；
      2. 映射：映射到4个数字，每个数字称为TokenId；

![llm_2_skill_01](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/llm_to_AgentSkills/img/llm_2_skill_01.png)

<br>

2. 解码：
   1. 解码；
   2. 映射；

![llm_2_skill_02](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/llm_to_AgentSkills/img/llm_2_skill_02.png)

## 【2.1】token总结

1. <font color=red>token是大模型处理文本的基本单位</font>； 

<br>

## 【2.2】tokenizer-词元化器实践 

1. 词元化器官网：[https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

![llm_2_skill_03](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/llm_to_AgentSkills/img/llm_2_skill_03.png)

【总结】

- 词与token并不是一对一的关系；
- <font color=red>你可以把token理解为模型自己学会的一套文本切分规则</font>； 切出来的每一块，就是它一次能够处理的最小单位； 
- <font color=red>平均来讲：1个token </font>
  -  等于0.75个单词；
  - 等于1.5~2个汉字；

补充：token到底是怎么生成的，可以看下这个视频《token生成机制全拆解，马克的技术工作坊》；它详细讲述了如何使用BPE算法来训练和使用Tokenizer； 

<br>

---

# 【3】Context-模型上下文

1. 模型是有记忆的；
   1. 业务场景：
      1. 用户：你好，我叫马克；
      2. 大模型： 你好，马克；
      3. 用户：我叫什么
      4. 大模型：你叫马克；
2. 好像大模型是有记忆的一样；<font color=red>大模型本质上是一个数学函数，那大模型是怎么记住之前的聊天内容的呢</font>？
   1. 用户每次：给大模型发送消息的时候，并不只是会发送我们的问题，背后的程序会自动把你之前的整段对话历史找出来一起发给大模型； 
   2. 这样：有了用户问题，有了对话历史，模型每次看到的都是完整的对话内容了；<font color=red>所以模型才知道之前发生了些什么</font>；
3. <font color=red>这就引出了Context的概念，即上下文</font>； 
   1. Context定义： 它代表大模型每次处理任务时所接收到的信息总和；
   2. 我们刚才聊到用户问题和对话历史，都是大模型所接受到的消息，所以它们都是Context的一部分；
   3. 此外，Context还有其他的内容；包括大模型正在输出的每一个token，也会被追加进来；
      1. 还会有工具列表（如天气工具，定位工具），System Prompt（系统提示词，如你是一个代码专家） 等信息； 
   4. <font color=red>总结：Context就是大模型每次处理任务时所接收到的信息总和</font>； 
      1. 从某种程度上说，可以把context看做是大模型的临时记忆体；
4. <font color=red>紧接着，引入下一个问题——这个Context能有多大，它能够塞多少token呢</font>？ 
   1. 这就引出了 Context Window这个概念； 

<br>

---

## 【3.1】Context Window-上下文窗口 

1. <font color=red>Context Window-上下文窗口：代表Context能够容纳的最大Token数量</font>； 
   1. 如：Context Window为1万，就代表模型最多能够处理1万个token； 不过1万的Context Window算是很小的；
   2. 目前主流的大模型：都有着非常大的Context Window；
      1. 比如GPT-5.4的Context Window是105万；
      2. Gemini 3.1 Pro的Context Window是100万； 
      3. Claude Opus 4.6的Context Window是100万； 

![llm_2_skill_04](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/llm_to_AgentSkills/img/llm_2_skill_04.png)

2. 我们之前说过：1个token大约是1.5个汉字；那么100万个token大概是150万个汉字；那整个哈利波特全集的内容都可以装下；

<br>

---

## 【3.2】实践-如何根据公司产品手册来回答用户问题？

1. <font color=red>问题：如何根据公司产品手册来回答用户问题</font>？
   1. 你要把这个手册的全部内容，跟着用户问题一起扔给大模型吗？这不是一个好的解决方案，因为这个产品手册太长了；即使模型的Context Window不被称爆， 你的成本也无法控制； 
   2. <font color=red>解决方案：这就需要一个叫做RAG的技术了</font>；

2. <font color=red>RAG技术：检索增强生成</font>； 
   1. 它可以从产品手册中抽取与用户问题最为匹配的几个片段，然后只把这几个片段发给大模型；让大模型只根据这几个片段来回答用户的 问题；
   2. 这样大模型接到的就不是一整本书，可能只是几段话； 这就也不会收到 Context Window的限制了；
   3. 进一步学习rag技术，参见《rag工作机制详解，马克的技术工作坊》

<br>

---

# 【4】Prompt-提示词 

1. <font color=red>Prompt定义：是大模型接收的具体问题或指令</font>； 
   1. 如：用户向大模型提需求，帮我写一首诗，这句话【帮我写一首诗】就是一个提示词；
2. 这里面有一个问题：如果你简单的说，给我写一首诗；那么大模型可能只会写一首古诗；但也可能给你写现代诗；
   1. <font color=red>为什么会这样呢？ 因为你的prompt太模糊了，大模型不知道你具体想要什么</font>；
   2. <font color=red>所以：prompt怎么写，直接决定了大模型的输出质量 </font> ；一个好的prompt应该是清晰的，具体的，明确的；
      1. 比如你可以这样写：请帮我写一首五言绝句，主题是秋天的落叶，风格要悲凉一点。
      2. 这样一来，大模型就清楚多了， 它生成的内容也就更加符合你的预期；
   3. <font color=red>这就是为什么有个专门领域，叫做Prompt Engineering，提示词工程</font>；

## 【4.1】Prompt Engineering-提示词工程

1. 提示词工程：说白了，也就是研究怎么把话说明白，让大模型更精准地理解你的意图；
2. <font color=red>这个领域曾经比较火，但现在还在提它的人寥寥无几</font>； 
   1. 一方面：因为门槛太低，本质上就是把话说清楚； 
   2. 另一方面：大模型的能力越来越强，即使提示词含糊不清，大模型也能够大致猜出你的意图来；这种情况下，也就不需要在提示词上花太多功夫；

<br>

---

## 【4.2】prompt分类（用户提示词+系统提示词）

1. 你有没有想过一个问题：有些时候，我们不仅要告诉大模型它要处理的具体任务，还要告诉它人设和做事规则；<font color=red>也就是告诉大模型它是谁，它应该按照什么规则做事；所以这就引出了两种不同的prompt</font>；
   1. User Prompt-用户提示词：说明具体任务；它是用户自己输入的；
   2. System Prompt-系统提示词：说明人设和做事规则；它是开发者在后台配置的；

<br>

### 【4.2.1】prompt分类实践 

1. 业务背景：假设你要做一个数学辅导机器人，你希望它不要直接告诉学生答案；而是要引导学生思考；这时候你就需要两种prompt；
   1. <font color=red>第1种：system prompt，系统提示词</font>；
      1. 例：你是一位耐心的数学老师。当学生问你数学问题时，不要直接给出答案，而是要一步步引导学生思考，帮助它们理解解题思路。
      2. <font color=red>注意：这段话是你作为开发者，在后台设置的，用户根本看不到，但它会一直影响大模型的行为</font>；
   2. <font color=red>第2种：user prompt，用户提示词</font>；
      1. 例：学生问到， 3+5 等于几？ 
2. <font color=red>大模型看到上述系统提示词与用户提示词后，它会这样想</font>：
   1. 我的角色是数学老师，我要引导学生思考，而不是直接说答案；好，那我就这样回答：
      1. 我们可以这样想， 你手里有3个苹果，然后又拿了5个苹果，现在一共有多少个苹果呢？你可以数一数看；
      2. 看到了吗？ 如果没有System Prompt，大模型可能就直接说出答案8了；
         1. 但因为有了System Prompt的约束，它知道自己要扮演一个引导式的老师，所以回答就完全不一样；
3. <font color=red>相信，你现在可以理解 用户提示词与系统提示词的区别了</font>；
   1. 有了它们的配合，大模型既能够守住规矩，又能够完成你的具体需求；

<br>

---

# 【5】Tool工具

## 【5.1】大模型的一个问题

1. 大模型的一个问题：它无法感知外部环境；
   1. 假设你问大模型： 今天上海的天气怎么样？
   2. 大模型会说：抱歉，我无法实时获取天气信息。我的知识库截止到2025年10月，无法提供当前的天气数据；
      1. <font color=red>为什么会这样呢</font>？  因为大模型只是个文字接龙游戏，它的能力是根据训练数据来预测下一个词，但它真的没有办法去查天气预报网站，拿到实时的天气数据；
      2. <font color=red>解决方法： 这就需要tool工具了</font>；

## 【5.2】Tool-工具或函数

1. <font color=red>tool定义：本质上就是一个函数</font>； 你给它输入，它就给你输出；
   1. 比如一个天气查询工具：它的输入包含两个参数，分别是城市和日期；
      1. 传入后，函数内部一通操作，比如它可能会调用气象局的接口；最后它都会给你一个输出，告诉你对应的天气信息；
      2. 有了它，大模型就可以回答天气相关的问题了； 

<br>

### 【5.2.1】从用户提问到大模型回答的完整流程

1. <font color=red> 整个流程涉及的角色包括： 用户， 大模型， 天气查询工具，平台（理解为传话筒）</font>；
   1. <font color=red>因为用户，大模型，天气查询工具，这3个角色没有办法直接对话</font>；所以我们就需要平台这个角色来做传递信息的工作；
   2. 平台本质上： 就是一段代码，用来负责上传下达的； 

2. 整体时序图如下:

![llm_2_skill_05](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/llm_to_AgentSkills/img/llm_2_skill_05.png)

<br>

【时序图解释】

1. <font color=red>每个角色（大模型，工具，平台）都有自己的职责</font>；
   1. <font color=red>大模型的职责有2个</font>：
      1. 选择工具：具体来说，是选择需要调用的工具，并生成对应的工具参数；
      2. 归纳总结： 也就是在拿到工具的执行结果后，模型需要对结果做一个归纳总结； 
   2. 工具的职责只有1个：查询天气；
   3. 平台的职责：串联流程；
      1. 比如告诉模型哪些工具可用，根据模型的指令去调用工具；

<br>

3. <font color=red>所以大家要分清楚每个角色在干什么</font>？
   1. 有人可能会以为调用工具的是模型；尤其初学者很容易就这么想；
   2. <font color=red>但是模型能够做的仅仅是输出一段文本</font>，告诉平台它想要调用什么工具，而调用工具这个事情最终还是需要平台来完成；

## 【5.3】总结-Tool的本质

1. <font color=red>tool的本质：就是给大模型提供一套它可以调用的外部能力；让大模型能够感知和影响外部环境</font>；

<br>

---

# 【6】MCP-模型上下文协议









































































