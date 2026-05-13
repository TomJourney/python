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

【总结】cline到此时摸底就结束了；这一切都发生在我们注册工具的一瞬间； 后面就是要等待合适的时机再使用mcp server 了；

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























































