"""Shared fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTODEPTH_DATA", str(tmp_path / "jobs"))
    from app.services.jobs import reset_store

    reset_store()
    from app.main import app

    return TestClient(app)
