"""Entry point for openenv validate."""
import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import app  # noqa: E402


def main() -> None:
    """Start the SupportOpsEnv server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
