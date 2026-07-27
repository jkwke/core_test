from core_test.llm import LLM
from core_test.agent import Agent
from core_test.api_key import ensure_api_key
import colorama
from colorama import Fore, Style

print("欢迎使用 CoreCoder！")

def main():
    colorama.init()
    # ensure_api_key()
    
    llm = LLM(
        model="模型名称",
        api_key="API",
        base_url="",
        temperature=0.0,
        max_tokens=8192,
    )
    agent = Agent(llm=llm, max_context_tokens=256000)
    # 交互式 REPL
    _repl(agent)


def _repl(agent: Agent):
    """交互式读取-求值-打印循环。"""
    while True:

        user_input = input(f"{Fore.RED}[用户] > {Style.RESET_ALL}")

        if not user_input:
            continue

        # 内置命令
        if user_input.lower() in ("quit", "exit", "/quit", "/exit","tc","/tc"):
            print("\nBye!")
            break

        if user_input == "/help":
            _show_help()
            continue

        if user_input == "/save":
            _save_history(agent)
            continue

        if user_input == "/list":
            _list_histories()
            continue

        if user_input == "/reset":
            agent.reset()
            continue

        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: {p} 输入 + {c} 输出 = 合计：{p + c}"
            print(line)
            continue
        # 调用代理
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            print(f"{Fore.YELLOW}[助手]> {name}({_brief(kwargs)}){Style.RESET_ALL}")

        try:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                print()  # 流式令牌后的换行符
            else:
                # 响应未流式传输（在工具调用之后到来）
                print(response)
        except KeyboardInterrupt:
            print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            print(f"\n[red]Error: {e}[/red]")


def _show_help():
    print(
        '''
        /help         显示此帮助信息
        /reset        清空对话历史
        /tokens       显示令牌使用量
        /save         保存对话历史
        quit或tc      退出 CoreCoder
        '''
    )


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")


def _save_history(agent: Agent):
    filename = agent.save_history()
    print(f"对话历史已保存到 {filename}")


def _list_histories():
    import os
    directory = r"D:\CoreCoder\history"
    histories = [f for f in os.listdir(directory) if f.endswith(".json")]
    print("保存的对话历史:")
    for i, history in enumerate(histories):
        print(f"{i + 1}. {history}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")

