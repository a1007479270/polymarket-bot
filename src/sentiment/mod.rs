//! Twitter Sentiment Analysis Module
//!
//! Provides real-time sentiment analysis for crypto markets using Twitter data.
//! Key features:
//! - KOL (Key Opinion Leader) tracking with weighted influence
//! - Sentiment scoring using VADER-style lexicon analysis
//! - Trend detection for sudden sentiment shifts
//! - Integration with ML prediction pipeline

pub mod twitter_client;
pub mod sentiment_analyzer;
pub mod kol_tracker;

pub use twitter_client::TwitterClient;
pub use sentiment_analyzer::{SentimentAnalyzer, SentimentResult, SentimentScore};
pub use kol_tracker::{KolTracker, KolProfile, InfluenceWeight};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Aggregated sentiment signal for a specific asset
#[derive(Debug, Clone)]
pub struct SentimentSignal {
    /// Asset symbol (e.g., "BTC", "ETH")
    pub symbol: String,
    /// Composite sentiment score (-1.0 to 1.0)
    pub score: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Number of tweets analyzed
    pub tweet_count: u32,
    /// Weighted KOL sentiment
    pub kol_sentiment: f64,
    /// Sentiment trend (positive = improving, negative = declining)
    pub trend: f64,
    /// Timestamp of analysis
    pub timestamp: u64,
}

impl SentimentSignal {
    /// Returns true if sentiment is strongly bullish
    pub fn is_bullish(&self) -> bool {
        self.score > 0.3 && self.confidence > 0.6
    }

    /// Returns true if sentiment is strongly bearish
    pub fn is_bearish(&self) -> bool {
        self.score < -0.3 && self.confidence > 0.6
    }

    /// Returns the trading bias based on sentiment (-1.0 to 1.0)
    pub fn trading_bias(&self) -> f64 {
        self.score * self.confidence * (1.0 + self.kol_sentiment.abs() * 0.5)
    }
}

/// Main sentiment engine that coordinates all sentiment analysis
pub struct SentimentEngine {
    twitter_client: Arc<TwitterClient>,
    analyzer: Arc<SentimentAnalyzer>,
    kol_tracker: Arc<RwLock<KolTracker>>,
    /// Cache of recent sentiment signals
    signal_cache: Arc<RwLock<HashMap<String, SentimentSignal>>>,
    /// Historical sentiment for trend detection
    history: Arc<RwLock<Vec<(u64, HashMap<String, f64>)>>>,
}

impl SentimentEngine {
    /// Create a new sentiment engine
    pub fn new(bearer_token: Option<String>) -> Self {
        Self {
            twitter_client: Arc::new(TwitterClient::new(bearer_token)),
            analyzer: Arc::new(SentimentAnalyzer::new()),
            kol_tracker: Arc::new(RwLock::new(KolTracker::new())),
            signal_cache: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Analyze sentiment for a specific crypto asset
    pub async fn analyze_asset(&self, symbol: &str) -> SentimentSignal {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Check cache first (valid for 60 seconds)
        {
            let cache = self.signal_cache.read().await;
            if let Some(cached) = cache.get(symbol) {
                if now - cached.timestamp < 60 {
                    return cached.clone();
                }
            }
        }

        // Fetch recent tweets
        let tweets = self.twitter_client.search_crypto_tweets(symbol, 100).await;

        // Analyze sentiment for each tweet
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        let mut kol_score = 0.0;
        let mut kol_weight = 0.0;

        let kol_tracker = self.kol_tracker.read().await;

        for tweet in &tweets {
            let result = self.analyzer.analyze(&tweet.text);
            let weight = 1.0 + (tweet.engagement_score() as f64).ln().max(0.0);

            total_score += result.compound * weight;
            total_weight += weight;

            // Check if author is a KOL
            if let Some(kol) = kol_tracker.get_kol(&tweet.author_id) {
                let kol_w = kol.influence_weight();
                kol_score += result.compound * kol_w;
                kol_weight += kol_w;
            }
        }

        let score = if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        };

        let kol_sentiment = if kol_weight > 0.0 {
            kol_score / kol_weight
        } else {
            0.0
        };

        // Calculate confidence based on tweet volume
        let confidence = (tweets.len() as f64 / 50.0).min(1.0);

        // Calculate trend from history
        let trend = self.calculate_trend(symbol, score).await;

        let signal = SentimentSignal {
            symbol: symbol.to_string(),
            score,
            confidence,
            tweet_count: tweets.len() as u32,
            kol_sentiment,
            trend,
            timestamp: now,
        };

        // Update cache
        {
            let mut cache = self.signal_cache.write().await;
            cache.insert(symbol.to_string(), signal.clone());
        }

        // Update history
        {
            let mut history = self.history.write().await;
            if history.is_empty() || now - history.last().unwrap().0 >= 300 {
                let mut scores = HashMap::new();
                scores.insert(symbol.to_string(), score);
                history.push((now, scores));
                // Keep only last 24 hours
                while history.len() > 288 {
                    history.remove(0);
                }
            } else if let Some((_, scores)) = history.last_mut() {
                scores.insert(symbol.to_string(), score);
            }
        }

        signal
    }

    /// Calculate sentiment trend over recent history
    async fn calculate_trend(&self, symbol: &str, _current: f64) -> f64 {
        let history = self.history.read().await;
        if history.len() < 3 {
            return 0.0;
        }

        let recent: Vec<f64> = history
            .iter()
            .rev()
            .take(6)
            .filter_map(|(_, scores)| scores.get(symbol).copied())
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..recent.len()).map(|i| (i * i) as f64).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Normalize slope to -1.0 to 1.0 range
        (slope * 10.0).clamp(-1.0, 1.0)
    }

    /// Add a KOL to track
    pub async fn add_kol(&self, profile: KolProfile) {
        let mut tracker = self.kol_tracker.write().await;
        tracker.add_kol(profile);
    }

    /// Get aggregated market sentiment across all tracked assets
    pub async fn market_sentiment(&self) -> f64 {
        let cache = self.signal_cache.read().await;
        if cache.is_empty() {
            return 0.0;
        }

        let total: f64 = cache.values().map(|s| s.score * s.confidence).sum();
        let weights: f64 = cache.values().map(|s| s.confidence).sum();

        if weights > 0.0 {
            total / weights
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_signal_bullish() {
        let signal = SentimentSignal {
            symbol: "BTC".to_string(),
            score: 0.5,
            confidence: 0.8,
            tweet_count: 100,
            kol_sentiment: 0.6,
            trend: 0.2,
            timestamp: 0,
        };
        assert!(signal.is_bullish());
        assert!(!signal.is_bearish());
    }

    #[test]
    fn test_sentiment_signal_bearish() {
        let signal = SentimentSignal {
            symbol: "BTC".to_string(),
            score: -0.5,
            confidence: 0.8,
            tweet_count: 100,
            kol_sentiment: -0.4,
            trend: -0.3,
            timestamp: 0,
        };
        assert!(!signal.is_bullish());
        assert!(signal.is_bearish());
    }

    #[test]
    fn test_sentiment_signal_neutral() {
        let signal = SentimentSignal {
            symbol: "BTC".to_string(),
            score: 0.1,
            confidence: 0.5,
            tweet_count: 20,
            kol_sentiment: 0.0,
            trend: 0.0,
            timestamp: 0,
        };
        assert!(!signal.is_bullish());
        assert!(!signal.is_bearish());
    }

    #[test]
    fn test_trading_bias() {
        let signal = SentimentSignal {
            symbol: "BTC".to_string(),
            score: 0.6,
            confidence: 0.9,
            tweet_count: 100,
            kol_sentiment: 0.8,
            trend: 0.2,
            timestamp: 0,
        };
        let bias = signal.trading_bias();
        assert!(bias > 0.5); // Strong bullish bias
    }

    #[tokio::test]
    async fn test_sentiment_engine_creation() {
        let engine = SentimentEngine::new(None);
        let market = engine.market_sentiment().await;
        assert_eq!(market, 0.0); // Empty cache
    }

    #[tokio::test]
    async fn test_calculate_trend_empty() {
        let engine = SentimentEngine::new(None);
        let trend = engine.calculate_trend("BTC", 0.5).await;
        assert_eq!(trend, 0.0); // No history
    }
}
