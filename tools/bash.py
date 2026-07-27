"""Shell 命令执行（Windows PowerShell）。

所有命令通过 PowerShell 执行，支持目录追踪和危险命令检测。
"""

import ctypes
import locale
import os
import re
import subprocess
import threading
from .base import Tool

_local = threading.local()

_DANGEROUS_PATTERNS = [
    (r"Remove-Item\s+.*-Recurse\b.*-Force\b", "Forced recursive removal"),
    (r"Remove-Item\s+.*-Force\b.*-Recurse\b", "Forced recursive removal"),
    (r"\bri\s+.*-Recurse\b.*-Force\b", "Forced recursive removal"),
    (r"\bFormat-Volume\b", "Format a volume"),
    (r"\bClear-Content\b.*[a-z]:\\", "Clear content on system drive"),
]


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute shell commands via PowerShell on Windows. "
        "Use this for running tests, installing packages, git operations, "
        "file operations, and other command-line tasks. "
        "Supports command chaining (&&, ;) and directory tracking across calls."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The PowerShell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, timeout: int = 120) -> str:
        warning = _check_dangerous(command)
        if warning:
            return (
                f"\u26a0 Blocked: {warning}\n"
                f"Command: {command}\n"
                f"If intentional, modify the command to be more specific."
            )

        cwd = getattr(_local, "cwd", None) or os.getcwd()
        encoding = _detect_encoding()

        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                encoding=encoding,
                errors="replace",
                timeout=timeout,
                cwd=cwd,
            )

            if proc.returncode == 0:
                _update_cwd(command, cwd)

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if stdout.startswith("\ufeff"):
                stdout = stdout[1:]
            if stderr.startswith("\ufeff"):
                stderr = stderr[1:]

            out = stdout
            if stderr:
                out += f"\n[stderr]\n{stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"

            if len(out) > 15_000:
                out = (
                    out[:6000]
                    + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
                    + out[-3000:]
                )
            return out.strip() or "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout}s"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


def _detect_encoding() -> str:
    """获取 Windows 控制台活动代码页编码。"""
    try:
        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        return f"cp{cp}"
    except Exception:
        return locale.getpreferredencoding()


def _check_dangerous(cmd: str) -> str | None:
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return reason
    return None


def _update_cwd(command: str, current_cwd: str):
    """跟踪 cd / Set-Location 的目录变更。"""
    running = current_cwd
    changed = False

    parts = [command]
    if "&&" in command:
        parts = command.split("&&")
    elif ";" in command:
        parts = command.split(";")

    for part in parts:
        part = part.strip()
        target = None

        if part.startswith("cd "):
            target = part[3:].strip().strip("'\"")
        elif part.startswith("Set-Location "):
            target = part[len("Set-Location "):].strip().strip("'\"")
            if target.startswith("-Path "):
                target = target[6:].strip().strip("'\"")

        if target:
            new_dir = os.path.normpath(
                os.path.join(running, os.path.expanduser(target))
            )
            if os.path.isdir(new_dir):
                running = new_dir
                changed = True

    if changed:
        _local.cwd = running
