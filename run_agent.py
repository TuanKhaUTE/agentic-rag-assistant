import sys

from src.agent import run_agent


DEFAULT_GOAL = (
    "What is an AI agent and how does it use tools? Use the study knowledge base."
)


def main() -> None:
    user_goal = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_GOAL
    run_agent(user_goal)


if __name__ == "__main__":
    main()
