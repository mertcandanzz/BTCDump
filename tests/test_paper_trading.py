"""Tests for paper trading engine."""

import pytest

from btcdump.web.paper_trading import PaperTrader


@pytest.fixture
def trader():
    return PaperTrader(initial_balance=10000.0)


class TestPaperTrading:
    def test_initial_balance(self, trader):
        assert trader.balance == 10000.0

    def test_open_long(self, trader):
        result = trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "long"
        assert trader.balance == 9000.0  # 10% used

    def test_open_short(self, trader):
        trader.open_position("ETHUSDT", "short", 3000, size_pct=5)
        assert "ETHUSDT" in trader.positions

    def test_close_long_profit(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        result = trader.close_position("BTCUSDT", 55000)  # +10%
        assert result["pnl"] > 0
        assert result["pnl_pct"] > 0

    def test_close_short_profit(self, trader):
        trader.open_position("BTCUSDT", "short", 50000, size_pct=10)
        result = trader.close_position("BTCUSDT", 45000)  # -10% price = profit for short
        assert result["pnl"] > 0

    def test_duplicate_position_raises(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        with pytest.raises(ValueError, match="Already have position"):
            trader.open_position("BTCUSDT", "long", 50000, size_pct=10)

    def test_close_nonexistent_raises(self, trader):
        with pytest.raises(ValueError, match="No position"):
            trader.close_position("BTCUSDT", 50000)

    def test_portfolio_value(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        portfolio = trader.get_portfolio({"BTCUSDT": 55000})
        assert portfolio["total_value"] > 10000  # should be profitable

    def test_win_rate(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        trader.close_position("BTCUSDT", 55000)  # win
        trader.open_position("ETHUSDT", "long", 3000, size_pct=10)
        trader.close_position("ETHUSDT", 2500)  # loss
        portfolio = trader.get_portfolio({})
        assert portfolio["win_rate"] == 0.5

    def test_history_recorded(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        trader.close_position("BTCUSDT", 55000)
        history = trader.get_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "BTCUSDT"

    def test_reset(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        trader.reset()
        assert trader.balance == 10000.0
        assert len(trader.positions) == 0
        assert len(trader.history) == 0

    def test_journal(self, trader):
        trader.open_position("BTCUSDT", "long", 50000, size_pct=10)
        pos_id = list(trader.positions.values())[0].id
        trader.add_note(pos_id, "Good entry, confirmed by RSI")
        journal = trader.get_journal(pos_id)
        assert len(journal["notes"]) == 1
        assert "RSI" in journal["notes"][0]["note"]
