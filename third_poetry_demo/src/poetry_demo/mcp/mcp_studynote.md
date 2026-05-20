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

1. <font color=red>Mcp host定义: 指请求大模型并负责协调工具/上下文的宿主程序</font>；
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
3. <font color=red>用户向Cline发送请求：纽约明天的天气怎么样；MCP Host + MCP Server + 大模型的整体协作时序如下</font>：

![mcp_host_server_llm](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_host_server_llm.png)

<br>

【<font color=red>个人总结的MCPServer与Agent/CLine及LLM的交互时序图</font>】

![Agent_MCPServer_LLM_sequence](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/Agent_MCPServer_LLM_sequence.jpg)

【补充】Agent(Cline) 也叫做 MCP Host; 

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

### 【1.3.4】调用MCP Server：fetch

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

4. 向cline提出问题：

```c++
请抓取下面这个网页的内容，并将其转换为markdown后放到目录/XXX/mcp里面的guides.md文件中：https://docs.astral.sh/uv/guides/install-python/ 
```

![mcp_fetch_effect](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_fetch_effect.png)

<br>

---

### 【1.3.5】如何使用他人制作的MCP Server：npx部分

1. 与uvx安装python程序类似， npx安装的是node程序；
2. 由于npx是node的一部分， 所以我们直接安装node.js 即可； 
3. nodejs安装完成后，打开https://mcpmarket.com/zh；
4. <font color=red>搜索一个 名为hotnews的mcp server：用于拉去热点新闻</font>； 
5. 同cline安装fetch类似， 复制 hotnews这个mcpserver的配置到cline，并安装； 

```json
"mcp-server-hotnews": {
      "command": "npx",
      "args": [
        "-y",
        "@wopal/mcp-server-hotnews"
      ]
    }
```

【安装效果】

![hot_news_mcp_server_npx](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/hot_news_mcp_server_npx.png)

<br>

#### 【1.3.5.1】新建一个对话：调用hotnews这个mcp server

1. 问题：获取今天最火的科技新闻； 

![hot_news_mcp_server_calling](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/hot_news_mcp_server_calling.png)

<br>

---

# 【2】MCP-进阶

1. 内容包括：
   1.  手动编写一个MCP Server；
   2. 分析MCP底层协议（拦截MCP Server的输入与输出）
      1. 如何在不借助MCP Host和任何编程语言的情况下，根据MCP协议的规范直接与MCP Server沟通； 
   3. 总结MCP含义与地位； 
2. 补充：MCP协议与语言无关，咱们可以通过python，或node， 或java，或c# 编写mcp Server； 

## 【2.1】自行创建MCP Server

### 【2.1.1.】环境搭建

1. 环境搭建：
   1. 安装python，且版本大于3.10
   2. 安装uv， python的包管理器； 
   3. vscode， 编写代码的IDE工具； 
   4. cline，vscode的插件，它是一个MCP Host；

### 【2.1.2】编写第一个MCP Server

例子参见： [https://modelcontextprotocol.io/docs/develop/build-server](https://modelcontextprotocol.io/docs/develop/build-server)

【初始化虚拟环境】

```shell
# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx

# Create our server file
touch weather.py
```

【编写MCP Server】

```python
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# initialize the MCP server
mcp = FastMCP("weather", log_level="ERROR")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

# 网络请求函数
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

# 格式化告警数据 
def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instruction", "No specific instructions provided")}
"""


# create tool named get_alerts # 天气预警
@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


# create tool named get_forecast # 天气预报
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period["name"]}:
Temperature: {period["temperature"]}°{period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

【vscode的cline插件安装weather这个MCP Server】

```json
{
  "mcpServers": {
    "fetch": {
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ]
    }, 
    "mcp-server-hotnews": {
      "command": "npx",
      "args": [
        "-y",
        "@wopal/mcp-server-hotnews"
      ]
    }, 
    "weather": {
      "disabled": false,
      "timeout": 60,
      "command": "uv",
      "args": [
        "--directory",
        "/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/advance/weather",
        "run",
        "weather.py"
      ], 
      "transportType": "stdio"
    }
  }
}
```

【测试】在cline中提问：纽约明天的天气怎么样

![diy_mcp_server_weather](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/diy_mcp_server_weather.png)

<br>

---

## 【2.2】分析MCP底层协议（拦截MCP Server的输入与输出）

1. 编写一个日志脚本 mcp_logger.py，截取输入与输出； 
2. 我们让cline与mcp_logger.py 沟通， 再让mcp_logger.py 与真正的mcp server沟通； 
   1. mcp_logger.py 在其中充当中间人的角色；作用是获取cline与mcp server的输入与输出，并把输入输出日志保存到日志文件中；
   2.  通过查看日志文件，我们就能够知道cline如何与mcp server沟通的了；

![mcp_logger_io](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_logger_io.png)

<br>

---

### 【2.2.1】实战解析：MCP底层协议的完整剖析过程

1. 代码参见： https://github.com/MarkTechStation/VideoCode 
2. 日志输入输出：
   1. 输入： cline -> MCP Server 
   2. 输出：MCP Server -> cline 

3. Mcp_logger 打印结果：

![mcp_logger_result](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_logger_result.png)

【日志解说】

1. 第1行输入：cline -> mcp server， protocolVersion 表示cline使用的mcp协议的版本；
2. 第2行输出：mcp server -> cline, mcp server的回复； 同时 capacities 表示mcp server不支持这些协议；<font color=red> 我的名字叫weather，版本好是1.1.0</font>； 
3. 第3行输入： cline发送的，大意是收到； 
4. 第4行输入：cline发送 tools/list ，请求mcp server返回工具列表； 
5. 第5行输出：weather mcp server返回了工具列表，及工具描述，工具调用参数； 
   1. 工具描述：其实就是我们函数的注释，在python领域中，这叫docstring，它是一种特殊的注释；
   2. inputSchema： 定义的是json结构，给出tool的入参规范的；<font color=red> （这个InputSchema也是 @mcp.tool()这个装饰器从我们的参数里面提取出来的）</font>
      1. 大家要知道：模型不仅要选择与用户问题最匹配的tool，还要用用户的问题把tool的参数提取出来； 而且这个参数必须要复合 InputSchema 的规定，这样才能成功调用tool背后的函数；  
6. 第6行输入到第9行输出： 
   1. cline在询问mcp server有没有资源和资源模板可以使用； 
   2. 可以看到resources和 resource_templates 的结果都是空列表； 也就是说 mcp server的回答是没有；
   3. resource中文是资源， resource_templates 可以理解为动态资源； 
   4. 资源就是一个文件，或一个报告之类的东西； 

【总结】<font color=red>cline到此时摸底就结束了；这一切都发生在我们注册工具的一瞬间； 后面就是要等待合适的时机再使用mcp server 了</font>；

<br>

---

【补充】

```python
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."
```

Mcp server的工具定义注解 @mcp.tool() ，它就会提取出工具（函数）注释，并且把它放到工具介绍的description中<font color=red>（第5行输出的工具描述结果，这样大模型就可以从descriptino中了解到tool的用途，方便到时候选择与用户问题最匹配的tool）</font>。  

```
 """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
```

<br>

【补充2】 json描述 

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "tools": [
            {
                "name": "get_alerts",
                "description": "Get weather alerts for a US state.\n\nArgs:\n    state: Two-letter US state code (e.g. CA, NY)\n",
                "inputSchema": {
                    "properties": {
                        "state": {
                            "title": "State",
                            "type": "string"
                        }
                    },
                    "required": [
                        "state"
                    ],
                    "title": "get_alertsArguments",
                    "type": "object"
                }
            },
            {
                "name": "get_forecast",
                "description": "Get weather forecast for a location.\n\nArgs:\n    latitude: Latitude of the location\n    longitude: Longitude of the location\n",
                "inputSchema": {
                    "properties": {
                        "latitude": {
                            "title": "Latitude",
                            "type": "number"
                        },
                        "longitude": {
                            "title": "Longitude",
                            "type": "number"
                        }
                    },
                    "required": [
                        "latitude",
                        "longitude"
                    ],
                    "title": "get_forecastArguments",
                    "type": "object"
                }
            }
        ]
    }
}
```

<font color=red>inputSchema： 定义的是json结构</font>； 

<br>

---

### 【2.2.2】实战解析：cline带着用户问题请求mcp server 的日志输出

1. 用户再次向cline提出之前的问题（纽约明天的天气怎么样）；
2. cline与mcp server的日志输出如下：

![cline_mcp_server_log2](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/cline_mcp_server_log2.png)

【日志解析】

1. 第10行：cline向mcp server发出调用工具get_forecast()请求及调用参数（经纬度）； 
   1. <font color=red>cline传入的参数结构，满足inputSchema的定义（模型提取参数的时候，就会遵守这个InputSchema的规范）</font>；
2. 第11行：拿到请求后，mcp server就会去请求对应函数，然后输出结果，text字段值；

<br>

---

## 【2.3】使用MCP底层协议直接与MCP server交互

1. 在了解了mcp协议底层通讯细节，在了解了细节后，<font color=red>我们都不需要一个mcp host（宿主程序），就可以直接与mcp server沟通； 你只需要保证你发给mcp server的数据复合这个格式就可以了（mcp_logger输出到logger.io日志中的日志，如输入: 输出:的字样）</font>；
2. 为了让大家更加透彻理解mcp协议，接下来，本文演示如何直接与mcp server沟通，不经过cline； 

<br>

### 【2.3.1】直接与mcp server交互

1. 在终端运行mcp server程序：

```shell
uv --directory /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/advance/weather/weather2 run weather2.py
```

2. 在终端发送打招呼内容；

![direct_mcp_server](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/direct_mcp_server.png)

<br>

## 【2.4】超越表象： mcp协议的真实含义与定位

1. 问题： 大模型是怎么使用mcp协议的；

<br>

### 【2.4.1】mcp协议的作用范围

![mcp_work_range](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/mcp_work_range.png)

【图片解说】<font color=red>mcp协议：作用于 mcp server 与 cline之间交互的部分；（红框所示）</font>；

1. <font color=red>mcp协议主要规定了两部分内容（函数的注册与使用）</font>：
   1. 每个mcp server有哪些函数可以使用；
   2. 如何调用这些函数，即每个函数的调用方式；

2. mcp协议：规定的是如何发现和调用函数的； 这套协议脱离大模型也是能够用的； 
   1. <font color=red>mcp协议本身并没有规定与模型的交互方式； 即没有规定 cline 与 大模型的交互要如何处理</font>； 

3. 实际上不同的mcp host与模型的交互确实是会存在很大差异：
   1. 比如说cline是用xml与模型沟通； 而 cherrystudio使用 Function Calling的格式与模型沟通；（Function Calling是openai提出的一套协议，用来规定模型是如何调用函数的；）

3. <font color=red>总结： mcp协议本没有规定如何与模型交互； 这一点非常重要；明白这一点，你就明白了mcp协议的本质</font>；

<br>

### 【MCP协议总结】

1.  MCP， 模型上下文协议：上下文就是环境；
   1. 什么是环境： 环境就是周围有哪些函数可以调用，从而获取到外界信息，比如获取天气信息，网络信息，文件信息等； 
   2. <font color=red>mcp就是让模型感知外部环境的一个协议，所以它叫做模型上下文协议</font>；
      1. mcp这个名字有一定误导性，因为它并没有规定与模型交互的规则，而实际情况顶多可以说mcp是给模型服务的；

<br>

---

# 【3】MCP Host如何与大模型沟通

1. MCP只规定了MCP Host 与 MCP Server之间的沟通协议，并没有对模型的输入和输出格式提出要求；因此不同MCP Host就可能用不同的格式与模型沟通；
   1. 如Cline用XML与模型沟通；
   2. 本文以Cline为例，演示MCP Host是如何与模型进行沟通的；

![MCPServer_MCPHost_LLM](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPServer_MCPHost_LLM.jpg)

<br>

---

## 【3.1】截获模型输入输出参数的原理

1. 要了解Cline与模型是如何交互的，我们最好能够抓取到Cline发给模型的请求；
2. <font color=red>实现方法：我们启一个本地服务器作为中间人，无论是cline发送请求给模型，还是模型返回答案给cline，都要先经过这个本地服务器才行</font>；
   1. 本地服务器在接收到Cline的请求和模型的返回后，会把具体内容写入到一个日志文件中；这样我们查看文件的内容就可以知道Cline与模型说了些什么； 

![MCPHost_LLM_LocalLog](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_LocalLog.jpg)

3. cline支持连接我们的本地服务器吗？
   1. <font color=red>Cline的Act Mode中API Provider，选择OpenAI Compatible </font>； 它的意思是对应模型提供商虽然不是OpenAI，但是它的API完全兼容OpenAI的格式；
      1. 我们选择它之后，把本地服务器地址填到 BaseURL里面；然后再填好 API Key，Model Id等信息；
      2. 剩下的事情就去编写本地服务器，并且确保本地服务器的输入和输出，符合OpenAI的格式规范； 

![MCPHost_LLM_01](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_01.png)

<br>

## 【3.2】中转服务器代码解释

1. 本地服务器代码：

【llm_logger.py】

```python
import httpx
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse


class AppLogger:
    def __init__(self, log_file="llm.log"):
        """Initialize the logger with a file that will be cleared on startup."""
        self.log_file = log_file
        # Clear the log file on startup
        with open(self.log_file, 'w') as f:
            f.write("")

    def log(self, message):
        """Log a message to both file and console."""

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

        # Log to console
        print(message)


app = FastAPI(title="LLM API Logger")
logger = AppLogger("llm.log")


@app.post("/chat/completions")
async def proxy_request(request: Request):

    body_bytes = await request.body()
    body_str = body_bytes.decode('utf-8')
    logger.log(f"模型请求：{body_str}")
    body = await request.json()

    logger.log("模型返回：\n")

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Authorization": request.headers.get("Authorization"),
                    },
            ) as response:
                async for line in response.aiter_lines():
                    logger.log(line)
                    yield f"{line}\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

【代码解说】

1. event_stream：事件流；它要处理模型的流式返回结果；
   1. 对应这里面的 "Accept": "text/event-stream" ；<font color=red>也就是要求模型使用流式返回</font>；
   2. <font color=red>这个流式返回的专业名称叫做 Server-Sent Events, 简称SSE</font> ；

<br>

### 【3.2.1】SSE-Server Sent Events-服务器发送事件

1. SSE介绍： 一般情况下，我们使用http访问一个网站的时候，我们的浏览器会发送给目标服务器一个请求，目标服务器会返回对应结果，一去一回一次交互就完成了；
   1. <font color=red>这个交互方式有个缺陷</font>： 它处理不了服务器连续发回多次想要的情况；因为大模型聊天页面返回的结果都是几个字几个字的返回，而只是一去一回的话显然无法做到这种效果；
2. <font color=red>所以目前主流的大模型聊天页面用的都是SSE</font>；
   1. 它的特点是： 浏览器只需要请求一次，服务器接收请求后会连续多次发送响应，每次响应的内容都是几个字；
      1. 而浏览器接收到几个字就显示几个字； 这样用户就可以及时接收到模型的返回；出来几个字就看几个字，体验就会好很多； 
      2. 等到所有的结果都显示完毕后，服务器会发送一个完成的标识；
      3. <font color=red>浏览器接收到标识后关闭SSE连接</font>； 页面显示模型回答完毕整个流程就结束了；

<br>

【llm_logger.py】部分代码解说

```python
    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            # 请求大模型，把cline的请求转发给llm
            async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Authorization": request.headers.get("Authorization"),
                    },
            ) as response:
                # 每一个line就代表服务器的一次返回；我们把这些返回记录下来；
                # 这里的yield就会把这个响应再发回给cline；
                async for line in response.aiter_lines():  
                    logger.log(line)
                    yield f"{line}\n"
```

【补充】

1. 比如模型返回的完整消息是： 纽约明天的温度是24度；<font color=red>那么yield就可能会执行6次</font>，如下： 
   1. yield "纽约\n"
   2. yield "明天的\n"
   3. yield "温度\n"
   4. yield "是\n"
   5. yield "24度\n"
   6. yield "(结束标识符)\n"

【MCPHost + 本地服务器 + LLM交互图】

![Cline_LocalServer+LLM_sequence](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/Cline_LocalServer+LLM_sequence.jpg)

【图片解说】

1. <font color=red>由于使用了SSE， LLM会连续发送多个消息，但我们对每个消息的处理过程都是一样的，步骤如下</font>；
   1. 步骤1：LLM会把消息发给本地服务器（中转服务器）； 【箭头4】
   2. 步骤2：本地服务器写入日志； 【箭头5】
   3. 步骤3：本地服务器再转发给Cline； 【箭头6】
2. 如此循环下去，这样当整个对话结束的时候，我们就可以到日志文件里面去找我们所需的内容了；
3. 日志文件包含2部分内容：
   1. 模型的请求；
   2. 模型的返回；

<br>

---

### 【3.2.2】配置中转服务器 

1. 进入到项目目录里：执行python -m venv .venv (新建一个虚拟环境)
   1. 新建虚拟环境，是为了防止我们后续安装的依赖影响到系统； 
2. 虚拟环境创建好之后，我们执行 source .venv/bin/activate 
3. 接着执行 pip install -r requirements.txt  ; 安装相关依赖；

【requirements.txt 】 

```shell
fastapi==0.109.2
uvicorn==0.27.1
httpx==0.26.0
```

【代码解说】 

- fastapi： 用于定义post接口；
- uvicorn： 用于运行服务器；
- httpx：用于向LLM（如OpenRouter）发起http请求；

【命令行执行】

```shell
tom@TomMacbook %1~ %# python -m venv .venv
tom@TomMacbook %1~ %# source .venv/bin/activate 
(.venv) tom@TomMacbook %1~ %# vim requirements.txt
(.venv) tom@TomMacbook %1~ %# 
(.venv) tom@TomMacbook %1~ %# pip install -r requirements.txt
```

4. 安装好依赖后，我们再执行 python llm_logger.py  ；即可启动我们的本地服务器了；
   1. 可以看出端口是 8000 

![MCPHost_LLM_03](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_03.png)

5. 然后来到cline页面，配置本地服务器；

![MCPHost_LLM_04](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_04.png)

【测试】在cline中发送请求给大模型

![MCPHost_LLM_05](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_05.png)

【查看本地服务器的日志】

![MCPHost_LLM_06](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_06.png)

<br>

---

# 【4】解读简单场景下Cline发往模型的请求

1. 日志说明

![MCPHost_LLM_07](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_07.png)

【日志结果】

```json
{
  "model": "deepseek/deepseek-chat-v3-0324",
  "messages": [
    {
      "role": "system",
      "content": "You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, ..."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<task>\n你好\n</task>"
        },
        {
          "type": "text",
          "text": "\n# task_progress RECOMMENDED\n\nWhen starting a new task, it is recommended to include a todo list using the task_progress parameter.\n\n\n1. Include a todo list using the task_progress parameter in your next tool call\n2. Create a comprehensive checklist of all steps needed\n3. Use markdown format: - [ ] for incomplete, - [x] for complete\n\n**Benefits of creating a todo/task_progress list now:**\n\t- Clear roadmap for implementation\n\t- Progress tracking throughout the task\n\t- Nothing gets forgotten or missed\n\t- Users can see, monitor, and edit the plan\n\n**Example structure:**```\n- [ ] Analyze requirements\n- [ ] Set up necessary files\n- [ ] Implement main functionality\n- [ ] Handle edge cases\n- [ ] Test the implementation\n- [ ] Verify results```\n\nKeeping the task_progress list updated helps track progress and ensures nothing is missed.\n"
        },
        {
          "type": "text",
          "text": "<environment_details>\n# Visual Studio Code Visible Files\n../Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json\n\n# Visual Studio Code Open Tabs\n../Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json\n\n# Current Time\n2026/5/17 下午8:54:20 (Asia/Shanghai, UTC+8:00)\n\n# Current Working Directory (/Users/rong/Desktop) Files\n(Desktop files not shown automatically. Use list_files to explore if needed.)\n\n# Workspace Configuration\n{\n  \"workspaces\": {\n    \"/Users/rong/Desktop\": {\n      \"hint\": \"Desktop\"\n    }\n  }\n}\n\n# Detected CLI Tools\nThese are some of the tools on the user's machine, and may be useful if needed to accomplish the task: git, npm, curl, jq, make, node, mysql, sqlite3, code, grep, sed, awk, brew, bundle. This list is not exhaustive, and other tools may be available.\n\n# Context Window Usage\n0 / 128K tokens used (0%)\n\n# Current Mode\nACT MODE\n</environment_details>"
        }
      ]
    }
  ],
  ...
}
```

【代码解说】

1. system： 用于设定系统提示词，它的功能是设定模型需要提前感知的一些消息；如模型需要扮演的角色，模型可以用的工具列表，模型返回结果的格式等；这些都是cline写的；别看这里面的context只有一行，实际是个长文本；
2. 系统提示词的content翻译如下：

```markdown
# mcp_localserver_logger.sql 中文翻译

你是 Cline，一名经验丰富的软件工程师，精通多种编程语言、框架、设计模式以及最佳实践。

## 工具使用（TOOL USE）

你可以访问一组需要用户批准后才能执行的工具。

当多个操作彼此独立时（例如同时读取多个文件、并行搜索），你可以在一次响应中使用多个工具。

对于存在依赖关系的操作（即前一步结果会影响下一步操作），请按顺序依次使用工具。

你会收到所有工具调用的执行结果。

请谨慎使用工具，并确保：

* 在真正需要时才调用工具
* 工具调用参数准确无误
* 在继续后续步骤前，正确分析工具返回结果

## 软件工程行为规范

你应该：

* 编写清晰、可维护、符合规范的代码
* 优先考虑代码可读性
* 遵循项目既有风格
* 在必要时添加注释
* 避免不必要的复杂设计
* 优先使用成熟稳定的实现方案
* 注意安全性、性能与可扩展性

## 问题分析

在开始编码前：

1. 先理解需求
2. 分析现有代码结构
3. 找出相关文件
4. 设计合理方案
5. 再进行修改

## 修改代码时

请尽量：

* 保持改动最小化
* 避免影响无关逻辑
* 不要破坏现有功能
* 保持向后兼容
* 遵循 DRY 原则（不要重复自己）

## 调试与排查

当遇到 Bug 时：

* 先分析根因
* 不要盲目修改
* 利用日志和错误信息定位问题
* 逐步验证假设
* 修复后验证结果

## 与用户沟通

你应该：

* 清晰解释你的思路
* 在必要时说明原因
* 对不确定内容保持透明
* 避免编造信息
* 使用简洁专业的表达

## 文件操作

修改文件前：

* 先读取文件内容
* 理解上下文
* 再进行修改

避免：

* 无意义的大规模重构
* 覆盖用户未要求修改的内容
* 删除重要逻辑

## 安全规范

不要：

* 泄露敏感信息
* 输出密钥、Token、密码
* 引入恶意代码
* 执行危险操作
* 删除关键数据

## 最佳实践

始终优先：

* 正确性
* 可维护性
* 稳定性
* 用户体验

而不是：

* 炫技
* 过度设计
* 不必要优化

```

3. 提示词中所说的工具与MCP的工具是不同的；<font color=red>提示词提到的工具一共包含两部分内容，包括Cline内置工具，MCP工具</font>：
   1. Cline内置工具：	
      1. 如写入文件，替换文件内容，读取文件，运行终端命令；
   2. MCP工具： 
      1. 如 天气预告，气象预警；
4. 所以： 模型是可以用到上述两种工具的；

<br>

## 【4.1】模型是怎么通过XML格式与Cline交互的

1. 例： 用户问题是：src/main.js这个文件写了什么？
   1. cline会把这个问题发送给模型；
   2. 模型接到问题后发现它需要先调用 reada_file 这个工具来读取main.js的文件内容； 
      1. 于是，模型按照system prompt里面给出的xml格式，向cline请求读取这个文件的内容；
   3. cline接到请求后读取了 main.js的内容，并且返回给了模型； 
   4. 模型再往后就可以自己总结出答案了；
   5. <font color=red>总结：只要是模型按照cline规定的这种xml格式返回，cline就可以帮助模型调用各种它想要调用的工具</font>；

![MCPHost_LLM_08](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_08.jpg)

1. <font color=red>那到底有哪些工具可以使用呢？模型是怎么知道读取文件用的工具名是 read_file呢</font>。 
   1.  Cline在工具部分会给模型详细解释有哪些工具可以调用； 
   2. 它也会告诉模型每个工具的名称，参数格式，用途等之类的信息；
      1. 如 execute_command： 用于执行终端命令； 它的参数一共包含两部分；
         1. 一个是命令内容； 
         2. 一个是是否需要用户同意； 

```json

      "type": "function",
      "function": {
        "name": "execute_command",
        "description": "Request to execute a CLI command on the system. Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task.",
        "strict": false,
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string",
              "description": "The CLI command to execute. This should be valid for the current operating system. Do not use the ~ character or $HOME to refer to the home directory. Always use absolute paths. The command will be executed from the current workspace, you do not need to cd to the workspace."
            },
            "requires_approval": {
              "type": "boolean",
              "description": "To indicate whether this command requires explicit user approval or interaction before it should be executed. For system/file altering operations like installing/uninstalling packages, removing/overwriting files, system configuration changes, network operations, or any commands that are considered potentially dangerous must be set to true. False for safe operations like running development servers, building projects, and other non-destructive operations."
            }
          },
          "required": [
            "command",
            "requires_approval"
          ],
          "additionalProperties": false
        }
      }
    },
```

2. 再后面是其他的工具说明；
   1. 如 read_file 用于读取文件内容； 
   2. write_to_file： 用于写入文件内容； 
   3. replace_in_file： 用于替换文件内容； 
   4. search_file： 用于搜索文件；
   5. list_file： 用户列举当前项目目录中的文件列表； 
   6. list_code_definition_name： 列举指定目录顶层原代码文件中使用到的定义名称，如类，方法等；
   7. brower_action ：
   8. <font color=red>工具8：重点看 use_mcp_tool</font>：这是用来使用MCP工具的； 它的参数一共分为3个：
      1. server_name: mcp服务器名称； 
      2. tool_name: mcp工具的名称（一个MCPServer可以有多个工具方法）；
      3. arguments： MCP工具的输入参数； 

【例】这个xml，代表模型想要调用weather这个MCPServer下的get_forecast工具；使用工具的参数为latitude，longitude； 

```xml
<use_mcp_tool>
<server_name>weather</server_name>
<tool_name>get_forecast</tool_name>
<arguments>
{
    "latitude": 40.7128,
    "longitude": -74.006
}
</arguments>
</use_mcp_tool>
```

<br>

9. 工具9： access_mcp_resouce: 用于获取mcp资源； 填写MCP服务器和资源的URL即可； 
10. 工具10： ask_followup_question：是模型向用户提问的时候，所使用的工具；
11. 工具11：attemp_completion:  这个工具用于返回最终结论； 
    1. 比如模型调用了一系列工具后，它可能会认为自己已经完成了用户给出的任务，或者已经知道了用户问题的答案；此时，模型就会把最终结论放倒 attemp_completion的result参数里面；
    2. <font color=red>cline接收到这个XML标签后，就会把result的结论显示出来；  对话到这里就结束了</font>；

### 【例】工具11实践

![MCPHost_LLM_09](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/mcp/img/MCPHost_LLM_09.jpg)

<br>















































