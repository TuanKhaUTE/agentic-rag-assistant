import io
import re
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import ollama

from config import GENERATION_MODEL
from src.agent.agent_loop import MAX_STEPS, MAX_STEPS_MESSAGE, run_agent
from src.agent.tool_registry import TOOL_REGISTRY, execute_tool


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SCENARIOS = [
    {
        "id": "A",
        "name": "Knowledge retrieval",
        "goal": "What is RAG and why is it useful? Use the study knowledge base.",
        "expected_tools": ["search_knowledge"],
    },
    {
        "id": "B",
        "name": "Arithmetic",
        "goal": "Calculate (250 + 50) / 6.",
        "expected_tools": ["calculator"],
    },
    {
        "id": "C",
        "name": "File listing",
        "goal": "List the files in the knowledge folder.",
        "expected_tools": ["list_files"],
    },
    {
        "id": "D",
        "name": "File reading",
        "goal": "Read knowledge/ai_agent.txt and summarize its main topic.",
        "expected_tools": ["read_file"],
    },
    {
        "id": "E",
        "name": "Multi-tool task",
        "goal": (
            "Use the study knowledge base to find what RAG is, then calculate "
            "how many chunks would be selected across 4 questions if Top-K is 3 "
            "for each question."
        ),
        "expected_tools": ["search_knowledge", "calculator"],
    },
    {
        "id": "F",
        "name": "Direct answer",
        "goal": "Say hello in one short sentence.",
        "expected_tools": [],
    },
]


def _knowledge_citations_are_observed(record: dict[str, object]) -> bool:
    observed_chunk_ids: set[str] = set()
    for event in record["tool_events"]:
        if event["tool"] != "search_knowledge" or not isinstance(event["result"], dict):
            continue
        for result in event["result"].get("results", []):
            observed_chunk_ids.add(str(result["chunk_id"]))

    citations = set(
        re.findall(r"\[([^\]]+\.txt#chunk-\d{3})\]", str(record["final_answer"]))
    )
    return bool(citations) and citations.issubset(observed_chunk_ids)


def _scenario_passed(record: dict[str, object]) -> bool:
    if record["actual_tools"] != record["expected_tools"]:
        return False
    if not record["final_answer_produced"]:
        return False
    if not all(bool(event["success"]) for event in record["tool_events"]):
        return False

    scenario_id = record["id"]
    answer = str(record["final_answer"])
    events = record["tool_events"]

    if scenario_id == "A":
        return _knowledge_citations_are_observed(record)
    if scenario_id == "B":
        return events[0]["result"] == 50.0 and "50" in answer
    if scenario_id == "C":
        return isinstance(events[0]["result"], list) and bool(events[0]["result"])
    if scenario_id == "D":
        return isinstance(events[0]["result"], str) and bool(events[0]["result"])
    if scenario_id == "E":
        calculator_events = [event for event in events if event["tool"] == "calculator"]
        return (
            calculator_events[0]["result"] == 12
            and "12" in answer
            and _knowledge_citations_are_observed(record)
        )
    if scenario_id == "F":
        return not events
    return False


def run_scenario(scenario: dict[str, object]) -> dict[str, object]:
    step_count = 0
    tool_events: list[dict[str, object]] = []

    def recording_chat(**kwargs: object) -> object:
        nonlocal step_count
        step_count += 1
        return ollama.chat(**kwargs)

    def recording_execute(tool_name: str, arguments: object) -> dict[str, object]:
        result = execute_tool(tool_name, arguments)
        tool_events.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "validation": result["validation"],
                "success": result["success"],
                "result": result.get("result"),
                "error": result.get("error"),
            }
        )
        return result

    transcript_buffer = io.StringIO()
    with (
        patch("src.agent.agent_loop.execute_tool", side_effect=recording_execute),
        redirect_stdout(transcript_buffer),
    ):
        final_answer = run_agent(
            str(scenario["goal"]),
            chat_callable=recording_chat,
        )

    record = {
        **scenario,
        "steps": step_count,
        "tool_events": tool_events,
        "actual_tools": [event["tool"] for event in tool_events],
        "final_answer": final_answer,
        "final_answer_produced": bool(final_answer) and final_answer != MAX_STEPS_MESSAGE,
        "transcript": transcript_buffer.getvalue(),
    }
    record["passed"] = _scenario_passed(record)
    return record


def print_scenario_record(record: dict[str, object]) -> None:
    print(f"\n{'=' * 88}")
    print(f"SCENARIO {record['id']}: {record['name']}")
    print(f"Expected tool behavior: {record['expected_tools'] or ['direct answer']}")
    print(record["transcript"].rstrip())
    print("\n=== SCENARIO RECORD ===")
    print(f"Agent steps: {record['steps']}")
    print(f"Actual tools: {record['actual_tools'] or ['none']}")
    for event in record["tool_events"]:
        print(
            f"Tool event: {event['tool']} | arguments={event['arguments']} | "
            f"validation={event['validation']} | success={event['success']}"
        )
    print(f"Final answer produced: {record['final_answer_produced']}")
    print(f"Pass/Fail: {'PASS' if record['passed'] else 'FAIL'}")


def run_safety_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    unknown_result = execute_tool("unknown_tool", {})
    checks.append(
        {
            "name": "Unknown tool rejection",
            "passed": (
                unknown_result["validation"] == "rejected"
                and unknown_result["success"] is False
            ),
            "detail": unknown_result,
        }
    )

    calculator_result = execute_tool("calculator", {"expression": "2 ** 8"})
    checks.append(
        {
            "name": "Invalid calculator input",
            "passed": calculator_result["success"] is False,
            "detail": calculator_result,
        }
    )

    traversal_result = execute_tool(
        "read_file",
        {"relative_path": "../../something.txt"},
    )
    checks.append(
        {
            "name": "File path traversal rejection",
            "passed": traversal_result["success"] is False,
            "detail": traversal_result,
        }
    )

    max_step_calls = 0

    def repeated_unknown_tool(**_: object) -> object:
        nonlocal max_step_calls
        max_step_calls += 1
        return SimpleNamespace(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "unknown_tool",
                            "arguments": {},
                        }
                    }
                ],
            }
        )

    max_steps_buffer = io.StringIO()
    with redirect_stdout(max_steps_buffer):
        max_steps_result = run_agent(
            "Safety check for maximum steps.",
            chat_callable=repeated_unknown_tool,
        )
    checks.append(
        {
            "name": "MAX_STEPS enforcement",
            "passed": (
                max_steps_result == MAX_STEPS_MESSAGE
                and max_step_calls == MAX_STEPS
            ),
            "detail": {
                "model_calls": max_step_calls,
                "final_message": max_steps_result,
            },
        }
    )

    return checks


def print_final_table(records: list[dict[str, object]]) -> None:
    print("\n=== FINAL AGENT EVALUATION TABLE ===")
    print("Scenario | Expected tool(s) | Actual tool(s) | Steps | Final answer? | Pass/Fail")
    print("-" * 112)
    for record in records:
        expected = ", ".join(record["expected_tools"]) or "none"
        actual = ", ".join(record["actual_tools"]) or "none"
        print(
            f"{record['id']} - {record['name']} | {expected} | {actual} | "
            f"{record['steps']} | "
            f"{'YES' if record['final_answer_produced'] else 'NO'} | "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )


def main() -> None:
    records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        record = run_scenario(scenario)
        records.append(record)
        print_scenario_record(record)

    safety_checks = run_safety_checks()
    print_final_table(records)

    print("\n=== SAFETY CHECKS ===")
    for check in safety_checks:
        print(f"{check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
        print(f"Detail: {check['detail']}")

    scenarios_passed = sum(bool(record["passed"]) for record in records)
    safety_passed = sum(bool(check["passed"]) for check in safety_checks)

    print("\n=== AGENT EVALUATION COMPLETE ===")
    print(f"Scenarios passed: {scenarios_passed}/{len(records)}")
    print(f"Safety checks passed: {safety_passed}/{len(safety_checks)}")
    print(f"Available tools: {', '.join(TOOL_REGISTRY)}")
    print(f"Agent model: {GENERATION_MODEL}")
    print(f"MAX_STEPS: {MAX_STEPS}")
    print("RAG available as tool: yes")

    if scenarios_passed != len(records) or safety_passed != len(safety_checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
