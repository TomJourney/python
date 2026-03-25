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