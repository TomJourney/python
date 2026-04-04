"""
文件处理工具
"""
import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from logger_handler import logger

# 获取文件的md5的16进制字符串
def get_file_md5_hex(file_path:str):
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]，文件{file_path}不存在")
        return
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是文件")
        return
    # 计算md5值
    md5_obj = hashlib.md5()
    chunk_size = 4096 # 4KB分片，避免文件过大撑爆内存
    try:
        with open(file_path, "rb") as f:  # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_object.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{file_path}md5失败，{str(e)}")

# 返回文件夹内的文件列表（允许的文件后缀）
def listdir_with_allowed_type(path:str, allowed_type:tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]-{path}不是文件夹")
        return allowed_type
    for f in os.listdir(path):
        if f.endswith(allowed_type):
            files.append(os.path.join(path, f))
    return tuple(files)

# 加载pdf文档
def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

# 加载text文档
def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath).load()

