from src.tools.calculator_tool import CALCULATOR_SCHEMA, calculator
from src.tools.file_tools import (
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
    list_files,
    read_file,
)
from src.tools.knowledge_tool import SEARCH_KNOWLEDGE_SCHEMA, search_knowledge


TOOL_SCHEMAS = [
    SEARCH_KNOWLEDGE_SCHEMA,
    CALCULATOR_SCHEMA,
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
]

__all__ = [
    "TOOL_SCHEMAS",
    "calculator",
    "list_files",
    "read_file",
    "search_knowledge",
]
