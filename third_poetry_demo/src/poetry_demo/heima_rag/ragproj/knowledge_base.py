"""
知识库
"""

import os
import config_data as config
import hashlib

def check_md5(md5_str: str):
    """
        检查传入的md5字符串是否已经被处理过
        False-没有被处理过, True-已处理过
    """
    if not os.path.exists(config.md5_path):
        # 若文件不存在，那肯定没处理过这个md值
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding="utf-8").readline():
            line = line.strip() # 处理字符串前后的空格和回车
            if line == md5_str:
                return True # 已处理过
        return False

def save_md5(md5_str: str):
    """将传入的md5字符串记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串=转换为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding = encoding)

    # 创建md5对象
    md5_obj = hashlib.md5() # 得到md5 对象
    md5_obj.update(str_bytes)  # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest() # 得到md5的16进制字符串

    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        self.chroma = None # 向量存储的实力 Chroma向量数据库
        self.spliter = None # 文本分割器对象

    def upload_by_str(self, data, filename):
        """将传入的字符串"""