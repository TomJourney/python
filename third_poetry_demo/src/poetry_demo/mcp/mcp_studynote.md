[TOC]

# 【README】

本文总结自： MCP终极指南 - 从原理到实战，带你深入掌握MCP；

[https://www.bilibili.com/video/BV1uronYREWR/?spm_id_from=333.337.search-card.all.click&vd_source=a7f8b3035e870c8df27f1d01a17aac7f](https://www.bilibili.com/video/BV1uronYREWR/?spm_id_from=333.337.search-card.all.click&vd_source=a7f8b3035e870c8df27f1d01a17aac7f)

<br>

---

# 【1】MCP-基础

1. mcp： model context protocol， 模型上下文协议； <font color=red>是 Anthropic公司在2024年11月25日发布的一个协议</font>；
2. <font color=red>mcp用途：让大模型更好使用各类工具的一个协议</font>； 
   1. 如，借助mcp，我们可以让模型使用浏览器上网查询信息，可以让模型操作Unity编写游戏，也可以让模型查询实时路况； 
   2. 因为大模型本身其实只会问答，它并不会使用外部工具；
   3. <font color=red>而mcp的出现，就等于是让大模型拥有了使用各种外部工具的能力</font>；
3. 要想使用mcp，你还需要使用mcp host； 

<br>

## 【1.1】MCP Host

1. <font color=red>Mcp host定义: 指运行大模型并负责协调工具/上下文的宿主程序</font>；
   1. 常见的 mcp host包括： Claude  Desktop， Cursor，Cline， Cherry Studio 等；
2. 本文以 Cline为例，介绍mcp的使用方法； 

<br>

---

## 【1.2】安装mcp host（Cline）

1. Cline定义： cline是VSCode的一个插件；
2. Cline安装：搜索插件，点击安装；

3. 配置Cline用的API Key；

---

### 【1.2.1】MCP Server

1. <font color=red>MCP Server：指一个普通应用程序，并不是传统服务器的概念</font>；
   1. 它就是一个程序而已，只不过这个程序的执行符合MCP协议；
   2. 如：大部分MCP Server都是本地通过Node或通过python启动的；
      1. 只不过在使用过程中可能会联网，也可能不联网，纯本地使用也是可以的；
      2. 不管联网与否，它都可以叫做MCP Server；
2. ChatGPT的MCP Server定义：为大模型提供具体能力（如工具，数据）的程序；
   1. 提供工具：如读写本地文件，调用数据库，执行脚本，查询api；
   2. 提供数据源：如本地文档，git仓库， 公司知识库，云端数据等；  
3. 【例】假设我正在用 Claude + MCP：
   1. 场景：帮忙我找一下项目里的login代码：
   2. MCP流程：如下：
      1. MCP Host(Claude) 接受请求；
      2. MCP Client转发调用； 
      3. MCP Server 执行：
         1. 搜索本地代码；
         2. 返回结果； 
         3. 模型生成答案； 
4. MCP Server的类型：
   1. 本地型：如文件系统server， git server；
   2. 网络型：API调用Server， 数据库server；
   3. 企业型： 内部知识库， ERP系统； 



![mcp_server](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_server.png)

<br>

---











































