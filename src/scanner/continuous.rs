//! Continuous arbitrage scanner
//!
//! Background service that periodically scans all active markets for arbitrage.

use super::{ArbitrageOpp, OpportunitySender, ScannerConfig};
use crate::client::clob::ClobClient;
use crate::client::gamma::GammaClient;
use crate::error::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

/// Market info cache
#[derive(Debug, Clone)]
pub struct MarketInfo {
    pub condition_id: String,
    pub question: String,
    pub slug: String,
    pub yes_token_id: String,
    pub no_token_id: String,
    pub volume: Decimal,
    pub active: bool,
}

/// Continuous scanner that runs in the background
pub struct ContinuousScanner {
    config: ScannerConfig,
    gamma: GammaClient,
    clob: ClobClient,
    markets: Arc<RwLock<HashMap<String, MarketInfo>>>,
    opp_tx: OpportunitySender,
    running: Arc<RwLock<bool>>,
    
    // Metrics
    total_scans: Arc<RwLock<u64>>,
    opportunities_found: Arc<RwLock<u64>>,
}

impl ContinuousScanner {
    /// Create a new continuous scanner
    pub fn new(
        config: ScannerConfig,
        gamma: GammaClient,
        clob: ClobClient,
        opp_tx: OpportunitySender,
    ) -> Self {
        Self {
            config,
            gamma,
            clob,
            markets: Arc::new(RwLock::new(HashMap::new())),
            opp_tx,
            running: Arc::new(RwLock::new(false)),
            total_scans: Arc::new(RwLock::new(0)),
            opportunities_found: Arc::new(RwLock::new(0)),
        }
    }

    /// Start the scanner (returns immediately, runs in background)
    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                return Err(crate::error::BotError::Internal(
                    "Scanner already running".into(),
                ));
            }
            *running = true;
        }

        info!(
            "[Scanner] Starting continuous scanner, interval={}ms, min_spread={:.2}%",
            self.config.scan_interval.as_millis(),
            self.config.min_spread * dec!(100)
        );

        // Initial market load
        if let Err(e) = self.refresh_markets().await {
            warn!("[Scanner] Initial market load failed: {}", e);
        }

        // Spawn scan loop
        let scanner = self.clone_for_task();
        tokio::spawn(async move {
            scanner.scan_loop().await;
        });

        // Spawn refresh loop
        let scanner = self.clone_for_task();
        tokio::spawn(async move {
            scanner.refresh_loop().await;
        });

        Ok(())
    }

    /// Stop the scanner
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("[Scanner] Stopped");
    }

    /// Clone for spawning tasks (shares Arc references)
    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            gamma: self.gamma.clone(),
            clob: self.clob.clone(),
            markets: Arc::clone(&self.markets),
            opp_tx: self.opp_tx.clone(),
            running: Arc::clone(&self.running),
            total_scans: Arc::clone(&self.total_scans),
            opportunities_found: Arc::clone(&self.opportunities_found),
        }
    }

    /// Main scan loop
    async fn scan_loop(&self) {
        let mut ticker = interval(self.config.scan_interval);

        loop {
            ticker.tick().await;

            if !*self.running.read().await {
                break;
            }

            self.scan_all_markets().await;
        }
    }

    /// Market refresh loop
    async fn refresh_loop(&self) {
        let mut ticker = interval(self.config.refresh_interval);

        loop {
            ticker.tick().await;

            if !*self.running.read().await {
                break;
            }

            if let Err(e) = self.refresh_markets().await {
                warn!("[Scanner] Market refresh failed: {}", e);
            }
        }
    }

    /// Refresh market list from API
    pub async fn refresh_markets(&self) -> Result<()> {
        info!("[Scanner] Refreshing market list...");

        let markets = self.gamma.get_markets().await?;
        let mut market_map = HashMap::new();
        let mut new_count = 0;

        let current = self.markets.read().await;

        for market in markets {
            if market.outcomes.len() < 2 {
                continue; // Need Yes and No tokens
            }

            // Parse volume
            let volume = market.volume.unwrap_or(dec!(0));
            if volume < self.config.min_volume {
                continue;
            }

            let info = MarketInfo {
                condition_id: market.id.clone(),
                question: market.question.clone(),
                slug: market.slug.clone().unwrap_or_default(),
                yes_token_id: market.outcomes[0].token_id.clone(),
                no_token_id: market.outcomes[1].token_id.clone(),
                volume,
                active: true,
            };

            if !current.contains_key(&market.id) {
                new_count += 1;
            }
            market_map.insert(market.id, info);
        }

        drop(current);

        let total = market_map.len();
        *self.markets.write().await = market_map;

        info!("[Scanner] Market refresh complete: total={}, new={}", total, new_count);
        Ok(())
    }

    /// Scan all markets for arbitrage
    async fn scan_all_markets(&self) {
        let markets: Vec<MarketInfo> = {
            let m = self.markets.read().await;
            m.values().filter(|m| m.active).cloned().collect()
        };

        if markets.is_empty() {
            return;
        }

        {
            let mut scans = self.total_scans.write().await;
            *scans += 1;
        }

        // Concurrent scanning with semaphore
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent));
        let mut handles = Vec::new();

        for market in markets {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let scanner = self.clone_for_task();

            handles.push(tokio::spawn(async move {
                let result = scanner.check_market(&market).await;
                drop(permit);
                result
            }));
        }

        // Collect results
        for handle in handles {
            if let Ok(Some(opp)) = handle.await.unwrap_or(Ok(None)) {
                {
                    let mut found = self.opportunities_found.write().await;
                    *found += 1;
                }

                info!(
                    "[Scanner] Found arbitrage: {} spread={:.2}%, profit=${:.4}",
                    opp.slug,
                    opp.spread * dec!(100),
                    opp.net_profit
                );

                // Send to channel
                let _ = self.opp_tx.send(opp).await;
            }
        }
    }

    /// Check a single market for arbitrage
    async fn check_market(&self, market: &MarketInfo) -> Result<Option<ArbitrageOpp>> {
        // Fetch order books
        let yes_book = match self.clob.get_orderbook(&market.yes_token_id).await {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };

        let no_book = match self.clob.get_orderbook(&market.no_token_id).await {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };

        // Check liquidity
        if yes_book.asks.is_empty() || no_book.asks.is_empty() {
            return Ok(None);
        }

        let yes_ask = yes_book.asks[0].price;
        let no_ask = no_book.asks[0].price;
        let yes_size = yes_book.asks[0].size;
        let no_size = no_book.asks[0].size;

        // Calculate total cost
        let total_cost = yes_ask + no_ask;

        // Check for arbitrage opportunity (Yes + No < 1.0)
        if total_cost >= dec!(1) - self.config.min_spread {
            return Ok(None);
        }

        // Calculate liquidity
        let max_size = yes_size.min(no_size);
        if max_size < Decimal::from(self.config.min_liquidity) {
            return Ok(None);
        }

        // Calculate profit
        let spread = dec!(1) - total_cost;
        let profit_margin = spread / total_cost * dec!(100);
        let gross_profit = spread * max_size;
        let fees = gross_profit * self.config.taker_fee_rate;
        let net_profit = gross_profit - fees - self.config.gas_cost;

        if net_profit <= dec!(0) {
            return Ok(None);
        }

        Ok(Some(ArbitrageOpp {
            condition_id: market.condition_id.clone(),
            question: market.question.clone(),
            slug: market.slug.clone(),
            yes_token_id: market.yes_token_id.clone(),
            no_token_id: market.no_token_id.clone(),
            yes_ask,
            no_ask,
            total_cost,
            spread,
            max_size: max_size.try_into().unwrap_or(0),
            profit_margin,
            net_profit,
            confidence: dec!(0.8),
            detected_at: Utc::now(),
        }))
    }

    /// Get scan metrics
    pub async fn metrics(&self) -> (u64, u64) {
        let scans = *self.total_scans.read().await;
        let found = *self.opportunities_found.read().await;
        (scans, found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_arbitrage_calculation() {
        // Yes = 0.48, No = 0.48 → Total = 0.96 → Spread = 0.04 (4%)
        let yes_ask = dec!(0.48);
        let no_ask = dec!(0.48);
        let total_cost = yes_ask + no_ask;
        let spread = dec!(1) - total_cost;
        
        assert_eq!(total_cost, dec!(0.96));
        assert_eq!(spread, dec!(0.04));
        
        // Profit calculation
        let size = dec!(100);
        let gross = spread * size;
        assert_eq!(gross, dec!(4)); // $4 profit on 100 shares
    }

    #[test]
    fn test_no_arbitrage() {
        // Yes = 0.52, No = 0.52 → Total = 1.04 → No opportunity
        let yes_ask = dec!(0.52);
        let no_ask = dec!(0.52);
        let total_cost = yes_ask + no_ask;
        
        assert!(total_cost > dec!(1));
    }
}
