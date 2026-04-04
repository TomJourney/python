import os.path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from poetry_demo.heima_rag.agent_proj.model import model_factory
from poetry_demo.heima_rag.agent_proj.utils import file_handler
from poetry_demo.heima_rag.agent_proj.utils.config_handler import chroma_config
from langchain_text_splitters import RecursiveCharacterTextSplitter

from poetry_demo.heima_rag.agent_proj.utils.logger_handler import logger
from poetry_demo.heima_rag.agent_proj.utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=model_factory.embeddings_model,
            persist_directory=chroma_config["persist_directory"],
        )
        self.spliter =  RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_config["k"]})

    def load_document(self):
        """
        从数据文件夹读取数据文件， 转为向量存入向量库
        要计算文件的md5值，并做去重
        :return:None
        """
        def check_md5_hex(md5_for_check: str):
            # 若文件不存在，则创建文件
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                open(get_abs_path(chroma_config["md5_hex_store"]), "w", encoding="utf-8").close()
                return False # md5 没有被处理过

            # 若文件存在，则读取文件
            with open(get_abs_path(chroma_config["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True # 该md5保存过， 不做二次保存
                return False
        # 保存md5到文件
        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a") as f:
                f.write(md5_for_check + "\n")

        # 读取文件列表（把文件内容转为document对象）
        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return file_handler.txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return file_handler.pdf_loader(read_path)
            return []

        # 允许文件类型的文件路径
        allowd_files_path : list[str] = file_handler.listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )

        for path in allowd_files_path:
            # 获取文件的md5
            md5_hex = file_handler.get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"加载知识库： {path}内存已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"加载知识库：{path}内没有有效文本内容，跳过")
                    continue

                split_documents : list[Document] = self.spliter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"加载知识库：{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容保存到向量数据库
                self.vector_store.add_documents(split_documents)

                # 记录这个已经处理的md5，避免二次加载
                save_md5_hex(md5_hex)

                logger.info(f"加载知识库： {path}加载成功")
            except Exception as e:
                logger.error(f"加载知识库： {path}加载失败：{str(e)} ", exc_info=True)
                continue

# 测试用例
if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    print("---------- rag检索结果如下： ----------")
    retriever = vs.get_retriever()
    result = retriever.invoke("迷路")
    for entry in result:
        print(entry.page_content)
        print("."*20)

# 2026-04-04 18:30:35,734 - agent - INFO - vector_store.py:74 - 加载知识库： /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/选购指南.txt内存已经存在知识库内，跳过
# 2026-04-04 18:30:35,734 - agent - INFO - vector_store.py:74 - 加载知识库： /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/扫地机器人100问2.txt内存已经存在知识库内，跳过
# 2026-04-04 18:30:35,735 - agent - INFO - vector_store.py:74 - 加载知识库： /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/故障排除.txt内存已经存在知识库内，跳过
# 2026-04-04 18:30:35,735 - agent - INFO - vector_store.py:74 - 加载知识库： /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/扫拖一体机器人100问.txt内存已经存在知识库内，跳过
# 2026-04-04 18:30:35,735 - agent - INFO - vector_store.py:74 - 加载知识库： /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/维护保养.txt内存已经存在知识库内，跳过
# 2026-04-04 18:30:35,736 - agent - INFO - vector_store.py:74 - 加载知识库： /Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/heima_rag/agent_proj/data/扫地机器人100问.pdf内存已经存在知识库内，跳过
# ---------- rag检索结果如下： ----------
# 3. **什么是 dToF 导航技术？**
# - 直接飞行时间测距(direct Time-of-Flight)，比传统 LDS 测距更精准，探测距离可达 10 米。
# 4. **为什么有些机器人会"迷路"？**
# - 环境光线变化、反光表面干扰或传感器故障导致定位丢失。
# 5. **如何提高扫地机器人的建图精度？**
# - 选择配备激光雷达+AI 算法的机型，保持环境光线稳定，定期清洁传感器。
# ....................
# 21. 故障现象：机器人清扫路线混乱，无规律；检测：是否开启规划式清扫，地图是否完整，环境是否有新障碍物；修复：开启规划模式，重新建图，清理新障碍物。
#
# 22. 故障现象：分区清扫无法选择区域；检测：地图是否建图成功，APP是否卡顿，分区设置是否被删除；修复：重新建图，重启APP，重新设置分区。
# ....................
# 143. 故障现象：APP推送错误故障提示（如无故障却提示滤网堵塞）；检测：传感器是否误判，APP是否卡顿，是否需要更新；修复：擦拭传感器，重启APP，更新APP版本。
#
# 144. 故障现象：机器人在阳光直射下工作时，建图错乱；检测：避障传感器是否受强光干扰，是否有反光物；修复：避开阳光直射，遮挡反光物，重新建图。
# ....................