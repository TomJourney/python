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
   1. <font color=red>蒸馏模型：可以理解为是大模型的学生，它学习了大模型的核心功能；因为标准大模型的运行对硬件性能要求很高； </font>
   2. <font color=red>为了满足在个人pc机上运行大模型，蒸馏模型应运而生</font>；

---

## 【1.6】代码调用ollama的本地模型































