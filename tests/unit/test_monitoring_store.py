"""Unit tests for the monitoring store's configuration gate.

These tests do not require a live database; they verify the store reports
readiness correctly and refuses to connect when DATABASE_URL is absent.
"""

import pytest

from src.api.monitoring_store import MonitoringStore


def test_not_ready_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = MonitoringStore()
    assert store.ready is False


def test_ready_with_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost:5432/db")
    store = MonitoringStore()
    assert store.ready is True


def test_connect_raises_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = MonitoringStore()
    with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
        store._connect()
