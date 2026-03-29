# 持久或长期会话记忆
import json
import os
from collections.abc import Sequence

from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
# message_to_dict: 单个消息对象 (BaseMessage类实例) -> 字典
# message_from_dict： [字典, 字典...] -> [消息, 消息...]
# AIMessage, HumanMessage, SystemMessage, 都是BaseMessage的子类

# 获取历史消息
def get_history(session_id):
    return DiyFileChatMessageHistory(session_id, "./chat_history")


class DiyFileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id # 会话id
        self.storage_path = storage_path  # 不同会话id的存储文件， 所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_message(self, message:Sequence[BaseMessage]) -> None:
        # Sequence序列 ： 类似list, tuple
        all_messages = list(self.messages) # 已有的消息列表
        all_messages.extend(message) # 新的和已有的融合为一个list

        # 将数据同步写入到本地文件
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将 BaseMessage 消息转为字典（借助 json模块以json字符串写入文件 ）
        new_messages= [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property  # @property 把message方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内： list[字典]
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                message_data = json.load(f) # 返回值就是： list[字典]
                return messages_from_dict(message_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump([], f)

