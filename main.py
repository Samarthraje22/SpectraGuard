"""Root entrypoint launching the SpectraGuard FastAPI server."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the SpectraGuard ASGI server."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
