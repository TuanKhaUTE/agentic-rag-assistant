from collections.abc import Callable

from src.tools import TOOL_SCHEMAS, calculator, list_files, read_file, search_knowledge


TOOL_REGISTRY: dict[str, Callable[..., object]] = {
    "search_knowledge": search_knowledge,
    "calculator": calculator,
    "list_files": list_files,
    "read_file": read_file,
}

TOOL_SCHEMAS_BY_NAME = {
    str(schema["name"]): schema
    for schema in TOOL_SCHEMAS
}

OLLAMA_TOOL_SCHEMAS = [
    {"type": "function", "function": schema}
    for schema in TOOL_SCHEMAS
]


def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    return False


def _validate_arguments(tool_name: str, arguments: dict[str, object]) -> None:
    schema = TOOL_SCHEMAS_BY_NAME[tool_name]
    parameters = schema["parameters"]
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

    if parameters.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            raise ValueError(f"Unexpected argument(s): {', '.join(unexpected)}")

    for name, value in arguments.items():
        property_schema = properties[name]
        expected_type = property_schema.get("type")
        if expected_type and not _matches_json_type(value, expected_type):
            raise TypeError(f"Argument '{name}' must be of type {expected_type}")
        if expected_type in {"integer", "number"} and "minimum" in property_schema:
            if value < property_schema["minimum"]:
                raise ValueError(
                    f"Argument '{name}' must be at least {property_schema['minimum']}"
                )


def execute_tool(tool_name: str, arguments: object) -> dict[str, object]:
    """Validate and execute one explicitly allowlisted tool call."""
    if tool_name not in TOOL_REGISTRY:
        return {
            "tool": tool_name,
            "validation": "rejected",
            "success": False,
            "error": f"Unknown tool: {tool_name}",
        }

    if not isinstance(arguments, dict):
        return {
            "tool": tool_name,
            "validation": "rejected",
            "success": False,
            "error": "Tool arguments must be an object/dictionary",
        }

    try:
        _validate_arguments(tool_name, arguments)
    except (TypeError, ValueError) as error:
        return {
            "tool": tool_name,
            "validation": "rejected",
            "success": False,
            "error": str(error),
        }

    try:
        result = TOOL_REGISTRY[tool_name](**arguments)
    except Exception as error:
        return {
            "tool": tool_name,
            "validation": "allowed",
            "success": False,
            "error": str(error),
        }

    return {
        "tool": tool_name,
        "validation": "allowed",
        "success": True,
        "result": result,
    }
