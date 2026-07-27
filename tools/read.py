"""带行号的文件读取工具。"""

from pathlib import Path
from .base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read file contents with line numbers. "
        "Shows file metadata (total lines, size) and warns on binary files. "
        "Always read a file before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Default 2000.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            raw = p.read_bytes()
            if b"\x00" in raw[:8192]:
                return (
                    f"File: {p} ({_format_size(len(raw))})\n"
                    f"[Binary file - cannot display as text]"
                )

            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            display_offset = start + 1

            numbered = [f"{display_offset + i:>6}\t{ln}" for i, ln in enumerate(chunk)]
            body = "\n".join(numbered)

            stat = p.stat()
            info = f"{p} ({_format_size(stat.st_size)}, {total} lines)"

            if not body:
                return f"{info}\n(empty file)"

            if start + limit < total:
                end_line = display_offset + len(chunk) - 1
                return (
                    f"{info}\n"
                    f"{body}\n"
                    f"... ({total} lines total, showing {display_offset}-{end_line})"
                )
            return f"{info}\n{body}"

        except Exception as e:
            return f"Error: {e}"


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024
    return f"{size:.1f}TB"
