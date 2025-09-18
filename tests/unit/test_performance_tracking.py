"""Unit tests for performance tracking modules."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.performance_tracking.signal_tracker import SignalTracker
from src.performance_tracking.pnl_calculator import PnLCalculator
from src.performance_tracking.slippage_analyzer import SlippageAnalyzer
from src.performance_tracking.execution_monitor import ExecutionMonitor


class TestSignalTracker:
    """Test signal tracking functionality."""
    
    @pytest.fixture
    def signal_tracker(self):
        """Create signal tracker instance."""
        with patch('src.performance_tracking.signal_tracker.get_db'):
            return SignalTracker()
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        return {
            "id": "signal_001",
            "pair": "BTC/USDT",
            "entry_price": 50000.0,
            "target_price": 52000.0,
            "stop_loss": 49000.0,
            "timestamp": datetime.now(),
            "confidence": 85
        }
    
    @pytest.mark.asyncio
    async def test_track_signal(self, signal_tracker, sample_signal):
        """Test signal tracking."""
        signal_tracker.db = Mock()
        signal_tracker.db.add = Mock()
        signal_tracker.db.commit = AsyncMock()
        
        result = await signal_tracker.track_signal(sample_signal)
        
        assert result is not None
        assert result["id"] == sample_signal["id"]
        assert result["status"] == "TRACKED"
        signal_tracker.db.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_signal_status(self, signal_tracker):
        """Test signal status update."""
        signal_tracker.db = Mock()
        signal_tracker.db.query = Mock()
        signal_tracker.db.commit = AsyncMock()
        
        mock_signal = Mock()
        signal_tracker.db.query.return_value.filter_by.return_value.first.return_value = mock_signal
        
        result = await signal_tracker.update_status("signal_001", "EXECUTED", 50100.0)
        
        assert result is True
        assert mock_signal.status == "EXECUTED"
        assert mock_signal.actual_entry == 50100.0
    
    def test_calculate_signal_performance(self, signal_tracker, sample_signal):
        """Test signal performance calculation."""
        sample_signal["actual_entry"] = 50100.0
        sample_signal["actual_exit"] = 51900.0
        
        performance = signal_tracker.calculate_performance(sample_signal)
        
        assert "profit_pct" in performance
        assert "profit_usd" in performance
        assert performance["profit_pct"] > 0
        assert performance["profit_usd"] == 1800.0


class TestPnLCalculator:
    """Test P&L calculation functionality."""
    
    @pytest.fixture
    def pnl_calculator(self):
        """Create P&L calculator instance."""
        return PnLCalculator()
    
    @pytest.fixture
    def trades_data(self):
        """Create sample trades data."""
        return pd.DataFrame({
            'entry_price': [50000, 1800, 150],
            'exit_price': [52000, 1750, 160],
            'quantity': [0.1, 1.0, 10.0],
            'side': ['BUY', 'SELL', 'BUY'],
            'pair': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        })
    
    def test_calculate_trade_pnl(self, pnl_calculator):
        """Test individual trade P&L calculation."""
        pnl = pnl_calculator.calculate_trade_pnl(
            entry_price=50000,
            exit_price=52000,
            quantity=0.1,
            side='BUY'
        )
        
        assert pnl == 200.0  # (52000 - 50000) * 0.1
        
        # Test short trade
        pnl = pnl_calculator.calculate_trade_pnl(
            entry_price=1800,
            exit_price=1750,
            quantity=1.0,
            side='SELL'
        )
        
        assert pnl == 50.0  # (1800 - 1750) * 1.0
    
    def test_calculate_portfolio_pnl(self, pnl_calculator, trades_data):
        """Test portfolio P&L calculation."""
        total_pnl, breakdown = pnl_calculator.calculate_portfolio_pnl(trades_data)
        
        assert total_pnl > 0
        assert len(breakdown) == 3
        assert 'BTC/USDT' in breakdown
        assert 'ETH/USDT' in breakdown
        assert 'SOL/USDT' in breakdown
    
    def test_calculate_sharpe_ratio(self, pnl_calculator):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.005, 0.015])
        
        sharpe = pnl_calculator.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        assert sharpe != 0
    
    def test_calculate_max_drawdown(self, pnl_calculator):
        """Test maximum drawdown calculation."""
        equity_curve = pd.Series([10000, 11000, 10500, 12000, 11000, 13000, 12500])
        
        max_dd, dd_duration = pnl_calculator.calculate_max_drawdown(equity_curve)
        
        assert max_dd < 0
        assert max_dd > -1
        assert dd_duration >= 0


class TestSlippageAnalyzer:
    """Test slippage analysis functionality."""
    
    @pytest.fixture
    def slippage_analyzer(self):
        """Create slippage analyzer instance."""
        return SlippageAnalyzer()
    
    def test_calculate_slippage(self, slippage_analyzer):
        """Test slippage calculation."""
        slippage = slippage_analyzer.calculate_slippage(
            expected_price=50000.0,
            actual_price=50050.0,
            side='BUY'
        )
        
        assert slippage == 50.0
        assert slippage > 0  # Negative slippage for buy
        
        # Test sell slippage
        slippage = slippage_analyzer.calculate_slippage(
            expected_price=50000.0,
            actual_price=49950.0,
            side='SELL'
        )
        
        assert slippage == 50.0
        assert slippage > 0  # Negative slippage for sell
    
    def test_analyze_slippage_patterns(self, slippage_analyzer):
        """Test slippage pattern analysis."""
        trades = pd.DataFrame({
            'expected_price': [50000, 1800, 150, 50500, 1850],
            'actual_price': [50050, 1795, 151, 50520, 1848],
            'side': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
            'volume': [0.1, 1.0, 10.0, 0.2, 0.5]
        })
        
        patterns = slippage_analyzer.analyze_patterns(trades)
        
        assert 'average_slippage' in patterns
        assert 'max_slippage' in patterns
        assert 'slippage_by_hour' in patterns
        assert 'slippage_by_volume' in patterns
    
    def test_slippage_cost_calculation(self, slippage_analyzer):
        """Test slippage cost calculation."""
        cost = slippage_analyzer.calculate_slippage_cost(
            slippage_amount=50.0,
            quantity=0.1,
            price=50000.0
        )
        
        assert cost == 5.0  # 50 * 0.1
    
    def test_slippage_recommendations(self, slippage_analyzer):
        """Test slippage reduction recommendations."""
        high_slippage_data = pd.DataFrame({
            'slippage': [100, 150, 200, 180, 160],
            'volume': [1.0, 1.5, 2.0, 1.8, 1.6],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        
        recommendations = slippage_analyzer.get_recommendations(high_slippage_data)
        
        assert len(recommendations) > 0
        assert any('volume' in rec.lower() for rec in recommendations)


class TestExecutionMonitor:
    """Test execution monitoring functionality."""
    
    @pytest.fixture
    def execution_monitor(self):
        """Create execution monitor instance."""
        return ExecutionMonitor()
    
    @pytest.mark.asyncio
    async def test_monitor_execution(self, execution_monitor):
        """Test execution monitoring."""
        order = {
            "id": "order_001",
            "pair": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.1,
            "timestamp": datetime.now()
        }
        
        execution_monitor.exchange_client = Mock()
        execution_monitor.exchange_client.get_order_status = AsyncMock(
            return_value={"status": "FILLED", "filled_price": 50000.0}
        )
        
        result = await execution_monitor.monitor_execution(order)
        
        assert result["status"] == "FILLED"
        assert result["filled_price"] == 50000.0
    
    def test_calculate_execution_metrics(self, execution_monitor):
        """Test execution metrics calculation."""
        executions = pd.DataFrame({
            'order_time': pd.date_range('2024-01-01', periods=5, freq='S'),
            'execution_time': pd.date_range('2024-01-01 00:00:02', periods=5, freq='S'),
            'expected_price': [50000, 50100, 50200, 50300, 50400],
            'executed_price': [50010, 50110, 50190, 50310, 50395],
            'status': ['FILLED', 'FILLED', 'FILLED', 'PARTIAL', 'FILLED']
        })
        
        metrics = execution_monitor.calculate_metrics(executions)
        
        assert 'avg_execution_time' in metrics
        assert 'fill_rate' in metrics
        assert 'avg_price_improvement' in metrics
        assert metrics['fill_rate'] == 80.0  # 4 out of 5 filled
    
    def test_detect_execution_anomalies(self, execution_monitor):
        """Test execution anomaly detection."""
        executions = pd.DataFrame({
            'execution_time_ms': [100, 120, 110, 5000, 105, 115],  # One anomaly
            'slippage': [10, 15, 12, 200, 11, 14],  # One anomaly
            'status': ['FILLED', 'FILLED', 'FILLED', 'FAILED', 'FILLED', 'FILLED']
        })
        
        anomalies = execution_monitor.detect_anomalies(executions)
        
        assert len(anomalies) > 0
        assert any('execution_time' in str(a) for a in anomalies)
        assert any('slippage' in str(a) for a in anomalies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
