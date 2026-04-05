import time

import streamlit as st

from poetry_demo.heima_rag.agent_proj.react_agent import ReactAgent

# 标题
st.title("智扫机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    # 显示用户提问
    st.chat_message("user").write(prompt)
    # 收集用户提问
    st.session_state["message"].append({"role":"user", "content":prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        # 调用大模型
        stream_result = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                # 整段流式返回
                # yield chunk

                # 单字符流式返回
                for word in chunk:
                    time.sleep(0.01)
                    yield word

        st.chat_message("assistant").write_stream(capture(stream_result, response_messages))
        # 收集大模型回复的最新一条信息
        st.session_state["message"].append({"role":"assistant", "content":response_messages[-1]})
# 运行网页程序命令： streamlit run web_app.py