import concurrent.futures
import inspect
from core_test.llm import LLM
from core_test.tools import ALL_TOOLS
from core_test.tools.base import Tool
from core_test.tools.agent import AgentTool
from core_test.prompt import system_prompt
from core_test.context import ContextManager


class Agent:
    def __init__(self, llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)

        # 连接子代理能力
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """处理一条用户消息。可能涉及多轮 LLM/工具交互。"""
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
            )
            # 无工具调用 -> LLM 已完成，返回文本
            if not resp.tool_calls:
                self.messages.append(resp.message)
                return resp.content

            # 有工具调用 -> 执行（多个时并行执行，类似 Claude Code 的
            # StreamingToolExecutor，它并发运行独立的工具）
            self.messages.append(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # 并行执行多个工具调用
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # Ctrl+C 中断执行会导致助手 tool_calls 消息没有回复，
                # 从而污染下一个请求；需要回填
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # 如果工具输出较大则进行压缩
            self.context.maybe_compress(self.messages, self.llm)

        return "(已达到最大工具调用轮次)"

    def _exec_tool(self, tc) -> str:
        """执行单个工具调用，返回结果字符串。"""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # 先验证参数，这样工具内部抛出的 TypeError 不会被误标为调用者的参数错误
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        try:
            return tool.execute(**tc.arguments)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """使用线程并发运行多个工具调用。

        这受 Claude Code 的 StreamingToolExecutor 启发，它在模型仍在生成时
        就开始执行工具。我们简化为：当模型一次返回 N 个工具调用时，并行运行它们。
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    def _answer_pending_tool_calls(self, tool_calls):
        """为每个未获得回复的调用回填工具回复。

        OpenAI 兼容的 API 会拒绝助手消息中有 tool_calls 但没有匹配的 tool 回复的请求，
        因此当执行中途被中断时，这可以保持历史记录有效。
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """清除对话历史。"""
        self.messages.clear()

    def save_history(self, filename: str = ""):
        """自动编号保存对话历史到 D:\\CoreCoder\\history\\ 目录。"""
        import json
        import os
        import re

        save_dir = r"D:\CoreCoder\history"
        os.makedirs(save_dir, exist_ok=True)

        # 查找现有文件，确定下一个编号
        max_num = 0
        pattern = re.compile(r"^history_(\d+)\.json$")
        for fname in os.listdir(save_dir):
            m = pattern.match(fname)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num

        next_num = max_num + 1
        filepath = os.path.join(save_dir, f"history_{next_num}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

        return filepath

    def load_history(self, filename: str):
        """从 JSON 文件加载对话历史。"""
        import json
        with open(filename, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
