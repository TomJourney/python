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

## 【1.3】MCP Server（mcp服务器）

1. <font color=red>MCP Server：指一个普通应用程序，并不是传统服务器的概念</font>；
   1. 它就是一个程序而已，只不过这个程序的执行符合MCP协议；
   2. 如：大部分MCP Server都是本地通过Node或通过python启动的；
      1. 只不过在使用过程中可能会联网，也可能不联网，纯本地使用也是可以的；
      2. 不管联网与否，它都可以叫做MCP Server；
   3. 刚才Cline想要安装的名为OpenWeatherMap的MCP Server内置了一些模块：
      1. <font color=red>这些模块在mcp领域的专业名词叫做Tool，即工具或函数</font>；
2. ChatGPT的MCP Server定义：为大模型提供具体能力（如工具，数据）的程序；
   1. 提供工具：如读写本地文件，调用数据库，执行脚本，查询api；
   2. 提供数据源：如本地文档，git仓库， 公司知识库，云端数据等；  
   3. <font color=red>一个MCP Server可以包含多个工具</font>；
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

### 【1.3.1】MCPServer实践

1. 本地安装名为weather的mcp Server；
2. 使用cline这个mcp host运行weather这个mcp server，该mcpServer部署了多个天气相关的工具或函数；

<br>

### 【1.3.2】MCP Host与MCP Server回复用户问题的整体过程

1. MCP Host：Cline； MCP Server：本地编写的工具weather； 
2. Cline添加weather这个MCP Server，并加载MCP Server的工具列表；
3. 用户向Cline发送请求：纽约明天的天气怎么样；MCP Host + MCP Server + 大模型的整体协作时序如下：

![mcp_host_server_llm](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_host_server_llm.png)

<br>

### 【1.3.3】如何使用他人制作的MCP Server: uvx部分

1. mcp server市场：mcp.so , mcpmarket.com , smithery.ai 等；
2. mcp server： 大多是使用python或者node进行编写的； 
   1. 对应的启动程序一般是： uvx（python）， npx（node）； 
3. uvx：是 uv tool run 的缩写； （这个tool是uv领域的tool，非mcp中的工具tool）
   1. uv：uv就是python语言的一个包管理软件； 
   2. 而 uvx：可以用来直接启动python程序； 比如uvx ruff 可以用来安装并启动ruff这个程序；uvx会帮你把所需依赖+执行环境全都配置好，不需要你自己去处理；
4. 安装并运行uvx(https://github.com/astral-sh/uv)：

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh

tom@TomMacbook %1~ %# uvx pycowsay 'hellow world.'
Installed 1 package in 6ms

  -------------
< hellow world. >
  -------------
   \   ^__^
    \  (oo)\_______
       (__)\       )\/\
           ||----w |
           ||     ||

# 如果是下载到本地，则为其添加执行权限
chmod +x uv-installer.sh 
```

<br>

### 【1.3.4】MCP Server：fetch

1. fetch：用于抓取网页内容，通过uvx来启动的；
2. 打开 mcp.so ;  输入fetch查找，进入 https://mcp.so/server/fetch/test ，复制mcp config；

```json
{
  "mcpServers": {
    "fetch": {
      "args": [
        "mcp-server-fetch"
      ],
      "command": "uvx"
    }
  }
}
```

3. 把上述mcpServer的json配置拷贝到cline的mcpServer，执行安装名为fetch的mcp server；

![mcp_server_install](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_server_install.png)

<br>

---









































