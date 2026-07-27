"""搜索替换式文件编辑（Claude Code 的关键创新）。

核心思路：模型不是发送整个文件重写或行号补丁，
而是指定一个*精确*的子字符串来查找并替换。
该子字符串必须在文件中恰好出现一次，这消除了歧义，
使编辑安全且可审查。
"""

import difflib
from pathlib import Path

from .base import Tool

# 跟踪本次会话中修改过的文件，用于 /diff 命令
_changed_files: set[str] = set()


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match. "
        "old_string must appear exactly once in the file for safety. "
        "Include enough surrounding context to ensure uniqueness."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, file_path: str, old_string: str, new_string: str) -> str:
        try:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"

            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"Error: {file_path} is not a UTF-8 text file (edit_file only edits text files)"
            occurrences = content.count(old_string)

            if occurrences == 0:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                return (
                    f"Error: old_string not found in {file_path}.\n"
                    f"File starts with:\n{preview}"
                )
            if occurrences > 1:
                return (
                    f"Error: old_string appears {occurrences} times in {file_path}. "
                    f"Include more surrounding lines to make it unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content, encoding="utf-8")
            _changed_files.add(str(p))

            # 生成统一格式的 diff，让用户/LLM 能清楚看到修改了什么
            diff = _unified_diff(content, new_content, str(p))
            return f"Edited {file_path}\n{diff}"
        except Exception as e:
            return f"Error: {e}"


def _unified_diff(old: str, new: str, filename: str, context: int = 3) -> str:
    """生成新旧文件内容之间的紧凑统一格式 diff。"""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=context,
    )
    result = "".join(diff)
    # 截断过大的 diff
    if len(result) > 3000:
        result = result[:2500] + "\n... (diff truncated)\n"
    return result
