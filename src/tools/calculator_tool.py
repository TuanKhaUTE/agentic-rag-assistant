import ast
import operator


CALCULATOR_SCHEMA = {
    "name": "calculator",
    "description": "Safely calculate a basic arithmetic expression.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Arithmetic using numbers, +, -, *, /, and parentheses. "
                    "Use each stated quantity once; N items used M times each is N * M."
                ),
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 50


def _calculate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only integer and decimal numbers are allowed")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        try:
            return _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError as error:
            raise ValueError("Division by zero is not allowed") from error

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_calculate_node(node.operand))

    raise ValueError(
        "Unsupported expression: use only numbers, +, -, *, /, and parentheses"
    )


def calculator(expression: str) -> int | float:
    """Evaluate basic arithmetic by walking a restricted Python AST."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Expression is not valid arithmetic") from error

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("Expression is too complex")

    return _calculate_node(tree.body)
