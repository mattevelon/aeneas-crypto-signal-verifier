"""
Feature Engineering Module

Extracts, transforms, and engineers features from trading signals for ML models.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import uuid
import logging
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import ta

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Signal, SignalPerformance, TelegramMessage
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.market_data import MarketDataClient
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class FeatureSet:
    """Container for engineered features"""
    signal_id: str
    timestamp: datetime
    price_features: Dict[str, float] = field(default_factory=dict)
    technical_features: Dict[str, float] = field(default_factory=dict)
    volume_features: Dict[str, float] = field(default_factory=dict)
    microstructure_features: Dict[str, float] = field(default_factory=dict)
    signal_features: Dict[str, float] = field(default_factory=dict)
    temporal_features: Dict[str, float] = field(default_factory=dict)
    risk_features: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array"""
        all_features = []
        for feature_dict in [
            self.price_features,
            self.technical_features,
            self.volume_features,
            self.microstructure_features,
            self.signal_features,
            self.temporal_features,
            self.risk_features
        ]:
            all_features.extend(feature_dict.values())
        return np.array(all_features)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'features': {
                'price': self.price_features,
                'technical': self.technical_features,
                'volume': self.volume_features,
                'microstructure': self.microstructure_features,
                'signal': self.signal_features,
                'temporal': self.temporal_features,
                'risk': self.risk_features
            }
        }


class FeatureEngineer:
    """Engineers features for ML models from trading signals"""
    
    def __init__(self, market_client: Optional[MarketDataClient] = None):
        self.redis = get_redis()
        self.market_client = market_client or MarketDataClient()
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.feature_importance = {}
        self.cache_ttl = 3600
    
    async def engineer_features(self, signal_id: str) -> Optional[FeatureSet]:
        """Engineer comprehensive feature set for a signal"""
        try:
            # Check cache
            cached = await self._get_cached_features(signal_id)
            if cached:
                return cached
            
            async with get_async_session() as session:
                signal = await self._get_signal(session, signal_id)
                if not signal:
                    return None
                
                features = FeatureSet(
                    signal_id=signal_id,
                    timestamp=datetime.utcnow()
                )
                
                # Extract features in parallel
                await asyncio.gather(
                    self._extract_price_features(signal, features),
                    self._extract_technical_features(signal, features),
                    self._extract_volume_features(signal, features),
                    self._extract_microstructure_features(signal, features),
                    self._extract_signal_features(signal, features, session),
                    self._extract_temporal_features(signal, features),
                    self._extract_risk_features(signal, features)
                )
                
                await self._cache_features(signal_id, features)
                return features
                
        except Exception as e:
            logger.error(f"Error engineering features for {signal_id}: {str(e)}")
            return None
    
    async def engineer_batch_features(self, signal_ids: List[str]) -> pd.DataFrame:
        """Engineer features for multiple signals"""
        try:
            feature_sets = []
            batch_size = 10
            
            for i in range(0, len(signal_ids), batch_size):
                batch = signal_ids[i:i + batch_size]
                batch_features = await asyncio.gather(
                    *[self.engineer_features(sid) for sid in batch]
                )
                feature_sets.extend([f for f in batch_features if f])
            
            if not feature_sets:
                return pd.DataFrame()
            
            df = self._feature_sets_to_dataframe(feature_sets)
            df = await self._add_derived_features(df)
            df = await self._select_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in batch feature engineering: {str(e)}")
            return pd.DataFrame()
    
    async def _extract_price_features(self, signal: Signal, features: FeatureSet):
        """Extract price-based features"""
        try:
            price_data = await self.market_client.get_price_history(
                signal.pair, interval='1h', limit=100
            )
            
            if not price_data:
                return
            
            prices = [p['close'] for p in price_data]
            current_price = prices[-1] if prices else 0
            
            features.price_features = {
                'current_price': current_price,
                'entry_price': float(signal.entry_price) if signal.entry_price else 0,
                'price_to_entry_ratio': current_price / float(signal.entry_price) if signal.entry_price else 0,
                'price_mean': np.mean(prices),
                'price_std': np.std(prices),
                'price_range': np.max(prices) - np.min(prices) if prices else 0
            }
            
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                features.price_features.update({
                    'return_mean': np.mean(returns),
                    'return_std': np.std(returns),
                    'return_skewness': float(pd.Series(returns).skew()),
                    'max_return': np.max(returns)
                })
                
        except Exception as e:
            logger.error(f"Error extracting price features: {str(e)}")
    
    async def _extract_technical_features(self, signal: Signal, features: FeatureSet):
        """Extract technical indicator features"""
        try:
            ohlcv_data = await self.market_client.get_ohlcv(
                signal.pair, interval='1h', limit=200
            )
            
            if not ohlcv_data or len(ohlcv_data) < 50:
                return
            
            df = pd.DataFrame(ohlcv_data)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_width'] = bb.bollinger_wband()
            
            # ATR
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            latest = df.iloc[-1]
            features.technical_features = {
                'rsi': latest['rsi'] if pd.notna(latest['rsi']) else 50,
                'macd': latest['macd'] if pd.notna(latest['macd']) else 0,
                'macd_signal': latest['macd_signal'] if pd.notna(latest['macd_signal']) else 0,
                'macd_diff': latest['macd_diff'] if pd.notna(latest['macd_diff']) else 0,
                'bb_width': latest['bb_width'] if pd.notna(latest['bb_width']) else 0,
                'atr': latest['atr'] if pd.notna(latest['atr']) else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {str(e)}")
    
    async def _extract_volume_features(self, signal: Signal, features: FeatureSet):
        """Extract volume-based features"""
        try:
            volume_data = await self.market_client.get_24h_volume(signal.pair)
            
            if not volume_data:
                return
            
            ohlcv_data = await self.market_client.get_ohlcv(
                signal.pair, interval='1h', limit=100
            )
            
            if ohlcv_data:
                volumes = [d['volume'] for d in ohlcv_data]
                
                features.volume_features = {
                    'volume_24h': volume_data.get('volume', 0),
                    'volume_mean': np.mean(volumes),
                    'volume_std': np.std(volumes),
                    'volume_ratio': volume_data.get('volume', 0) / np.mean(volumes) if np.mean(volumes) > 0 else 0
                }
                
                if len(ohlcv_data) > 10:
                    prices = [d['close'] for d in ohlcv_data]
                    features.volume_features['volume_price_corr'] = np.corrcoef(prices, volumes)[0, 1]
            
        except Exception as e:
            logger.error(f"Error extracting volume features: {str(e)}")
    
    async def _extract_microstructure_features(self, signal: Signal, features: FeatureSet):
        """Extract market microstructure features"""
        try:
            order_book = await self.market_client.get_order_book(signal.pair)
            
            if not order_book:
                return
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
                
                bid_depth = sum(float(b[1]) for b in bids[:10])
                ask_depth = sum(float(a[1]) for a in asks[:10])
                depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
                
                features.microstructure_features = {
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'mid_price': mid_price,
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'depth_imbalance': depth_imbalance
                }
            
        except Exception as e:
            logger.error(f"Error extracting microstructure features: {str(e)}")
    
    async def _extract_signal_features(self, signal: Signal, features: FeatureSet, 
                                      session: AsyncSession):
        """Extract signal-specific features"""
        try:
            features.signal_features = {
                'confidence_score': float(signal.confidence_score) if signal.confidence_score else 0,
                'risk_level_encoded': self._encode_risk_level(signal.risk_level),
                'has_stop_loss': 1 if signal.stop_loss else 0,
                'stop_loss_distance': abs(float(signal.entry_price) - float(signal.stop_loss)) if signal.stop_loss and signal.entry_price else 0,
                'take_profit_count': len(signal.take_profits) if signal.take_profits else 0,
                'risk_reward_ratio': self._calculate_risk_reward_ratio(signal),
                'signal_direction': 1 if signal.direction == 'long' else -1
            }
            
        except Exception as e:
            logger.error(f"Error extracting signal features: {str(e)}")
    
    async def _extract_temporal_features(self, signal: Signal, features: FeatureSet):
        """Extract time-based features"""
        try:
            now = datetime.utcnow()
            signal_time = signal.created_at
            
            features.temporal_features = {
                'hour_of_day': signal_time.hour,
                'day_of_week': signal_time.weekday(),
                'is_weekend': 1 if signal_time.weekday() >= 5 else 0,
                'signal_age_hours': (now - signal_time).total_seconds() / 3600,
                'hour_sin': np.sin(2 * np.pi * signal_time.hour / 24),
                'hour_cos': np.cos(2 * np.pi * signal_time.hour / 24)
            }
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
    
    async def _extract_risk_features(self, signal: Signal, features: FeatureSet):
        """Extract risk-related features"""
        try:
            volatility = await self.market_client.get_volatility(signal.pair)
            
            features.risk_features = {
                'volatility': volatility if volatility else 0,
                'position_size_suggested': self._calculate_kelly_position_size(signal),
                'max_loss_potential': self._calculate_max_loss(signal)
            }
            
        except Exception as e:
            logger.error(f"Error extracting risk features: {str(e)}")
    
    async def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/interaction features"""
        try:
            # Price momentum
            if 'price_features_current_price' in df.columns and 'price_features_price_mean' in df.columns:
                df['price_momentum'] = df['price_features_current_price'] / df['price_features_price_mean']
            
            # Risk-adjusted return
            if 'price_features_return_mean' in df.columns and 'price_features_return_std' in df.columns:
                df['sharpe_ratio'] = df['price_features_return_mean'] / df['price_features_return_std'].replace(0, 1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding derived features: {str(e)}")
            return df
    
    async def _select_features(self, df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
        """Select top k features based on importance"""
        try:
            if df.shape[1] <= k:
                return df
            
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
            target = np.random.randn(len(df))  # Placeholder - would use actual performance
            
            selected_features = selector.fit_transform(df.fillna(0), target)
            selected_columns = df.columns[selector.get_support()].tolist()
            
            self.feature_importance = dict(zip(df.columns, selector.scores_))
            
            return df[selected_columns]
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize features"""
        try:
            scaler = self.scalers.get(method, self.scalers['standard'])
            normalized_data = scaler.fit_transform(df.fillna(0))
            return pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return df
    
    def reduce_dimensions(self, df: pd.DataFrame, n_components: int = 20) -> pd.DataFrame:
        """Reduce dimensions using PCA"""
        try:
            if df.shape[1] <= n_components:
                return df
            
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(df.fillna(0))
            columns = [f'pc_{i+1}' for i in range(n_components)]
            
            return pd.DataFrame(reduced_data, columns=columns, index=df.index)
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            return df
    
    # Helper methods
    def _encode_risk_level(self, risk_level: str) -> float:
        """Encode risk level"""
        levels = {'low': 0.33, 'medium': 0.66, 'high': 1.0}
        return levels.get(risk_level, 0.5)
    
    def _calculate_risk_reward_ratio(self, signal: Signal) -> float:
        """Calculate risk/reward ratio"""
        if not signal.stop_loss or not signal.take_profits or not signal.entry_price:
            return 0
        
        risk = abs(float(signal.entry_price) - float(signal.stop_loss))
        reward = abs(float(signal.take_profits[0].get('price', signal.entry_price)) - float(signal.entry_price))
        
        return reward / risk if risk > 0 else 0
    
    def _calculate_kelly_position_size(self, signal: Signal) -> float:
        """Calculate Kelly Criterion position size"""
        # Simplified Kelly calculation
        win_rate = 0.6  # Placeholder
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(0.25, kelly))  # Cap at 25% position size
    
    def _calculate_max_loss(self, signal: Signal) -> float:
        """Calculate maximum potential loss"""
        if not signal.stop_loss or not signal.entry_price:
            return 0.1  # Default 10% max loss
        
        return abs(float(signal.entry_price) - float(signal.stop_loss)) / float(signal.entry_price)
    
    def _feature_sets_to_dataframe(self, feature_sets: List[FeatureSet]) -> pd.DataFrame:
        """Convert list of FeatureSets to DataFrame"""
        rows = []
        for fs in feature_sets:
            row = {'signal_id': fs.signal_id}
            for prefix, features in [
                ('price_features', fs.price_features),
                ('technical_features', fs.technical_features),
                ('volume_features', fs.volume_features),
                ('microstructure_features', fs.microstructure_features),
                ('signal_features', fs.signal_features),
                ('temporal_features', fs.temporal_features),
                ('risk_features', fs.risk_features)
            ]:
                for k, v in features.items():
                    row[f'{prefix}_{k}'] = v
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    async def _get_signal(self, session: AsyncSession, signal_id: str) -> Optional[Signal]:
        """Get signal from database"""
        result = await session.execute(
            select(Signal).where(Signal.id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_cached_features(self, signal_id: str) -> Optional[FeatureSet]:
        """Get cached features"""
        cached = await self.redis.get(f"features:{signal_id}")
        if cached:
            data = json.loads(cached)
            fs = FeatureSet(
                signal_id=data['signal_id'],
                timestamp=datetime.fromisoformat(data['timestamp'])
            )
            for key, value in data['features'].items():
                setattr(fs, f'{key}_features', value)
            return fs
        return None
    
    async def _cache_features(self, signal_id: str, features: FeatureSet):
        """Cache features"""
        await self.redis.setex(
            f"features:{signal_id}",
            self.cache_ttl,
            json.dumps(features.to_dict(), default=str)
        )
