"""Configured evaluation frontend; hash-pinned evaluator code is not modified."""
from prediction_cli import main


if __name__ == "__main__":
    raise SystemExit(main("evaluate"))
