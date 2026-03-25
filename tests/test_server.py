"""Integration tests for FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient

from btcdump.web.server import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert j["ml_features"] >= 100
        assert j["api_routes"] >= 100

    def test_dashboard(self, client):
        r = client.get("/api/dashboard")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "active_coin" in j

    def test_full_report(self, client):
        r = client.get("/api/full-report")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "platform" in j


class TestProviderEndpoints:
    def test_providers(self, client):
        r = client.get("/api/providers")
        assert r.status_code == 200

    def test_signal_cached(self, client):
        r = client.get("/api/signal/cached")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True


class TestPaperTradingEndpoints:
    def test_portfolio(self, client):
        r = client.get("/api/paper/portfolio")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "balance" in j

    def test_history(self, client):
        r = client.get("/api/paper/history")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True

    def test_journal(self, client):
        r = client.get("/api/paper/journal")
        assert r.status_code == 200


class TestSignalEndpoints:
    def test_signal_history(self, client):
        r = client.get("/api/signal-history")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "records" in j

    def test_signal_stats(self, client):
        r = client.get("/api/signal-history/stats")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True

    def test_signal_calibration(self, client):
        r = client.get("/api/signal-calibration")
        assert r.status_code == 200


class TestAnalysisEndpoints:
    def test_feature_importance_no_model(self, client):
        r = client.get("/api/feature-importance")
        assert r.status_code == 200
        # May fail gracefully if no model trained

    def test_market_breadth(self, client):
        r = client.get("/api/market-breadth")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True

    def test_trade_coach(self, client):
        r = client.get("/api/trade-coach")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "score" in j

    def test_risk_dashboard(self, client):
        r = client.get("/api/risk-dashboard")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True


class TestAlertEndpoints:
    def test_get_alerts(self, client):
        r = client.get("/api/alerts")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True

    def test_get_smart_alerts(self, client):
        r = client.get("/api/smart-alerts")
        assert r.status_code == 200

    def test_get_trading_rules(self, client):
        r = client.get("/api/trading-rules")
        assert r.status_code == 200


class TestSettingsEndpoints:
    def test_get_settings(self, client):
        r = client.get("/api/settings/all")
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "settings" in j

    def test_webhook_status(self, client):
        r = client.get("/api/webhook/status")
        assert r.status_code == 200

    def test_notification_status(self, client):
        r = client.get("/api/notifications/status")
        assert r.status_code == 200
