import sys
from pathlib import Path

import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent))

from fake_server import FakeModelServer  # noqa: E402


@pytest_asyncio.fixture
async def fake():
    server = FakeModelServer()
    url = await server.start()
    server.url = url
    yield server
    await server.stop()
