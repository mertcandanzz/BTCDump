"""Tests for settings persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from btcdump.web.settings_store import SettingsStore


@pytest.fixture
def store(tmp_path):
    return SettingsStore(tmp_path / "test_settings.json")


class TestSettingsStore:
    def test_default_values(self, store):
        data = store.get_all()
        assert "watchlist" in data
        assert "interval" in data

    def test_set_and_get(self, store):
        store.set("test_key", "test_value")
        assert store.get("test_key") == "test_value"

    def test_persistence(self, tmp_path):
        path = tmp_path / "persist_test.json"
        store1 = SettingsStore(path)
        store1.set("persisted", True)

        # New instance reads from disk
        store2 = SettingsStore(path)
        assert store2.get("persisted") is True

    def test_update_multiple(self, store):
        store.update({"key1": "val1", "key2": "val2"})
        assert store.get("key1") == "val1"
        assert store.get("key2") == "val2"

    def test_get_default(self, store):
        assert store.get("nonexistent", "default") == "default"

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "new_settings.json"
        store = SettingsStore(path)
        store.set("created", True)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["created"] is True
