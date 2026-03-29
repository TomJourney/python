import time
from with_memory_rag import RagService
import streamlit as st
import config_data as config

# 运行命令
# streamlit run app_qa_webpage.py

# 标题
st.title("智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [
        {
            "role":"assistant",
            "content":"你好，有什么可以帮助你的？"
        }
    ]
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user", "content":prompt})

    ai_result_list = []

    with st.spinner("AI思考中..."):
        # 同步写
        # result = st.session_state["rag"].chain.invoke({"input":prompt}, config.session_id)
        # st.chat_message("assistant").write(result)
        # st.session_state["message"].append({"role": "assistant", "content": result})

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # 流式写
        stream_result = st.session_state["rag"].chain.stream({"input":prompt}, config.session_id)
        st.chat_message("assistant").write_stream(capture(stream_result, ai_result_list))
        # ["a", "b", "c"]  "".join(list) -> abc
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_result_list)})


