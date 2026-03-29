import streamlit as st

# 运行命令
# streamlit run app_qa_webpage.py

# 标题
st.title("智能客服")
st.divider()

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户提问
    st.chat_message("user").write(prompt)

    with st.spinner("AI思考中..."):
        st.chat_message("assistant").write("你也好呀")