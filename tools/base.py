"""所有工具的基类。"""

from abc import ABC, abstractmethod


class Tool(ABC):
    """最小工具接口。继承此类以添加新功能。"""

    name: str
    description: str
    parameters: dict  # 函数参数的JSON架构

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """运行工具并返回文本结果。"""
        ...

    def schema(self) -> dict:
        """OpenAI 函数调用模式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }