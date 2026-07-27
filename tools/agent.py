"""子代理生成（受 Claude Code 的 AgentTool 启发，1397 行）。

思路：对于复杂的子任务，生成一个独立的代理，拥有自己的
对话历史和工具访问权限。这让主代理可以委派工作，
比如"去研究这个代码库并报告"，而不会污染自己的上下文窗口。

子代理运行完成后返回文本摘要。
"""

from .base import Tool


class AgentTool(Tool):
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    # 由 Agent.__init__ 在构造后设置
    _parent_agent = None

    def execute(self, task: str) -> str:
        if self._parent_agent is None:
            return "错误：代理工具未初始化（没有父代理）"

        # 在此处导入以避免循环依赖
        from core_test.agent import Agent

        parent = self._parent_agent
        sub = Agent(
            llm=parent.llm,
            tools=[t for t in parent.tools if t.name != "agent"],  # 禁止递归代理
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
        )

        try:
            result = sub.chat(task)
            # 截断过长的结果，避免撑爆父代理的上下文
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as e:
            return f"Sub-agent error: {e}"
