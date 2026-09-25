"""Configured inference frontend for images/folders, without evaluation metrics."""
from prediction_cli import main


if __name__ == "__main__":
    raise SystemExit(main("infer"))
