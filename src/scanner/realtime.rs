//! Real-time arbitrage scanner using WebSocket price updates
//!
//! Monitors price changes and immediately checks for arbitrage opportunities.

use super::{ArbitrageOpp, OpportunitySender, ScannerConfig};
use crate::client::{PriceStream, MarketUpdate};
use crate::error::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Market pair info for real-time tracking
#[derive(Debug, Clone)]
struct MarketPair {
    condition_id: String,
    question: String,
    slug: String,
    yes_token_id: String,
    no_token_id: String,
}

/// Real-time price cache
#[derive(Debug, Clone, Default)]
struct PriceCache {
    yes_ask: Option<Decimal>,
    yes_bid: Option<Decimal>,
    no_ask: Option<Decimal>,
    no_bid: Option<Decimal>,
}

/// Real-time arbitrage scanner using WebSocket
pub struct RealtimeArbitrageScanner {
    config: ScannerConfig,
    /// Token ID -> Market pair mapping
    token_to_market: Arc<RwLock<HashMap<String, MarketPair>>>,
    /// Condition ID -> Price cache
    price_cache: Arc<RwLock<HashMap<String, PriceCache>>>,
    /// Opportunity sender
    opp_tx: OpportunitySender,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl RealtimeArbitrageScanner {
    /// Create a new real-time scanner
    pub fn new(config: ScannerConfig, opp_tx: OpportunitySender) -> Self {
        Self {
            config,
            token_to_market: Arc::new(RwLock::new(HashMap::new())),
            price_cache: Arc::new(RwLock::new(HashMap::new())),
            opp_tx,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Register markets to monitor
    pub async fn register_markets(&self, markets: Vec<(String, String, String, String, String)>) {
        let mut token_map = self.token_to_market.write().await;
        let mut cache = self.price_cache.write().await;

        for (condition_id, question, slug, yes_token, no_token) in markets {
            let pair = MarketPair {
                condition_id: condition_id.clone(),
                question,
                slug,
                yes_token_id: yes_token.clone(),
                no_token_id: no_token.clone(),
            };

            token_map.insert(yes_token, pair.clone());
            token_map.insert(no_token, pair);
            cache.insert(condition_id, PriceCache::default());
        }

        info!("[RT Scanner] Registered {} markets", cache.len());
    }

    /// Get all token IDs for WebSocket subscription
    pub async fn get_token_ids(&self) -> Vec<String> {
        self.token_to_market.read().await.keys().cloned().collect()
    }

    /// Start processing price updates from WebSocket
    pub async fn start(&self, mut stream: PriceStream) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                return Err(crate::error::BotError::Internal(
                    "RT Scanner already running".into(),
                ));
            }
            *running = true;
        }

        info!("[RT Scanner] Starting real-time arbitrage scanning...");

        let scanner = self.clone_for_task();

        tokio::spawn(async move {
            while let Some(update) = stream.recv().await {
                if !*scanner.running.read().await {
                    break;
                }

                scanner.handle_price_update(update).await;
            }
        });

        Ok(())
    }

    /// Stop the scanner
    pub async fn stop(&self) {
        *self.running.write().await = false;
        info!("[RT Scanner] Stopped");
    }

    /// Clone for spawning tasks
    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_to_market: Arc::clone(&self.token_to_market),
            price_cache: Arc::clone(&self.price_cache),
            opp_tx: self.opp_tx.clone(),
            running: Arc::clone(&self.running),
        }
    }

    /// Handle a price update from WebSocket
    async fn handle_price_update(&self, update: MarketUpdate) {
        // Find the market for this token
        let market = {
            let map = self.token_to_market.read().await;
            map.get(&update.token_id).cloned()
        };

        let market = match market {
            Some(m) => m,
            None => return,
        };

        // Update price cache
        {
            let mut cache = self.price_cache.write().await;
            let entry = cache.entry(market.condition_id.clone()).or_default();

            if update.token_id == market.yes_token_id {
                if let Some(ask) = update.best_ask {
                    entry.yes_ask = Some(ask);
                }
                if let Some(bid) = update.best_bid {
                    entry.yes_bid = Some(bid);
                }
            } else if update.token_id == market.no_token_id {
                if let Some(ask) = update.best_ask {
                    entry.no_ask = Some(ask);
                }
                if let Some(bid) = update.best_bid {
                    entry.no_bid = Some(bid);
                }
            }
        }

        // Check for arbitrage
        self.check_arbitrage(&market).await;
    }

    /// Check if there's an arbitrage opportunity
    async fn check_arbitrage(&self, market: &MarketPair) {
        let prices = {
            let cache = self.price_cache.read().await;
            match cache.get(&market.condition_id) {
                Some(p) => p.clone(),
                None => return,
            }
        };

        // Need both Yes and No ask prices
        let (yes_ask, no_ask) = match (prices.yes_ask, prices.no_ask) {
            (Some(y), Some(n)) => (y, n),
            _ => return,
        };

        let total_cost = yes_ask + no_ask;
        let spread = dec!(1) - total_cost;

        // Check if opportunity exists
        if spread <= self.config.min_spread {
            return;
        }

        // Calculate profit (assume minimum liquidity for now)
        let min_size = Decimal::from(self.config.min_liquidity);
        let gross_profit = spread * min_size;
        let fees = gross_profit * self.config.taker_fee_rate;
        let net_profit = gross_profit - fees - self.config.gas_cost;

        if net_profit <= dec!(0) {
            return;
        }

        let profit_margin = spread / total_cost * dec!(100);

        let opp = ArbitrageOpp {
            condition_id: market.condition_id.clone(),
            question: market.question.clone(),
            slug: market.slug.clone(),
            yes_token_id: market.yes_token_id.clone(),
            no_token_id: market.no_token_id.clone(),
            yes_ask,
            no_ask,
            total_cost,
            spread,
            max_size: self.config.min_liquidity,
            profit_margin,
            net_profit,
            confidence: dec!(0.9), // Higher confidence for real-time
            detected_at: Utc::now(),
        };

        info!(
            "[RT Scanner] Arbitrage detected: {} spread={:.2}%, profit=${:.4}",
            market.slug,
            spread * dec!(100),
            net_profit
        );

        let _ = self.opp_tx.send(opp).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_register_markets() {
        let (tx, _rx) = mpsc::channel(10);
        let scanner = RealtimeArbitrageScanner::new(ScannerConfig::default(), tx);

        scanner
            .register_markets(vec![(
                "cond1".to_string(),
                "Question?".to_string(),
                "slug".to_string(),
                "yes_token".to_string(),
                "no_token".to_string(),
            )])
            .await;

        let ids = scanner.get_token_ids().await;
        assert_eq!(ids.len(), 2);
    }
}
