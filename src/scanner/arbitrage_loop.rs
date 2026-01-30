//! Main arbitrage loop - Discovery → Scan → Execute → Notify
//!
//! Complete end-to-end arbitrage system that:
//! 1. Discovers current crypto 15m/30m/hourly markets
//! 2. Monitors for arbitrage opportunities (Yes + No < 1.0)
//! 3. Executes trades via ArbitrageExecutor
//! 4. Sends notifications via Telegram

use super::crypto_market::{CryptoMarketDiscovery, MarketInterval};
use super::{ArbitrageOpp, ScannerConfig};
use crate::client::{ClobClient, GammaClient};
use crate::executor::{ArbitrageExecutor, ArbitrageExecutorConfig};
use crate::notify::Notifier;
use crate::error::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};
use tracing::{info, warn, error, debug};

/// Arbitrage loop configuration
#[derive(Debug, Clone)]
pub struct ArbitrageLoopConfig {
    /// Minimum spread to consider (e.g., 0.005 = 0.5%)
    pub min_spread: Decimal,
    /// Scan interval
    pub scan_interval: Duration,
    /// Market refresh interval
    pub refresh_interval: Duration,
    /// Maximum position per trade
    pub max_position: Decimal,
    /// Enable actual trading (false = dry run)
    pub live_trading: bool,
    /// Send notifications
    pub notify_enabled: bool,
}

impl Default for ArbitrageLoopConfig {
    fn default() -> Self {
        Self {
            min_spread: dec!(0.005),
            scan_interval: Duration::from_millis(500),
            refresh_interval: Duration::from_secs(300),
            max_position: dec!(50),
            live_trading: false,
            notify_enabled: true,
        }
    }
}

/// Statistics for the arbitrage loop
#[derive(Debug, Default)]
pub struct LoopStats {
    pub scans: u64,
    pub opportunities_found: u64,
    pub trades_executed: u64,
    pub trades_successful: u64,
    pub total_profit: Decimal,
    pub start_time: Option<chrono::DateTime<Utc>>,
}

/// Main arbitrage loop runner
pub struct ArbitrageLoop {
    config: ArbitrageLoopConfig,
    discovery: CryptoMarketDiscovery,
    clob: ClobClient,
    executor: ArbitrageExecutor,
    notifier: Notifier,
    stats: Arc<RwLock<LoopStats>>,
    running: Arc<RwLock<bool>>,
}

impl ArbitrageLoop {
    /// Create a new arbitrage loop
    pub fn new(
        config: ArbitrageLoopConfig,
        gamma_url: &str,
        clob: ClobClient,
        notifier: Notifier,
    ) -> Self {
        let executor_config = ArbitrageExecutorConfig {
            max_position_size: config.max_position,
            dry_run: !config.live_trading,
            ..Default::default()
        };

        Self {
            config,
            discovery: CryptoMarketDiscovery::new(gamma_url),
            clob: clob.clone(),
            executor: ArbitrageExecutor::new(clob, executor_config),
            notifier,
            stats: Arc::new(RwLock::new(LoopStats::default())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the arbitrage loop
    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                return Err(crate::error::BotError::Internal("Already running".into()));
            }
            *running = true;
        }

        {
            let mut stats = self.stats.write().await;
            stats.start_time = Some(Utc::now());
        }

        info!(
            "[ArbLoop] Starting - min_spread={:.2}%, live={}, notify={}",
            self.config.min_spread * dec!(100),
            self.config.live_trading,
            self.config.notify_enabled
        );

        if self.config.notify_enabled {
            let mode = if self.config.live_trading { "LIVE" } else { "DRY RUN" };
            let _ = self.notifier.send(&format!(
                "🚀 <b>Arbitrage Bot Started</b>\n\nMode: {}\nMin Spread: {:.2}%",
                mode,
                self.config.min_spread * dec!(100)
            )).await;
        }

        // Main scan loop
        let mut scan_ticker = interval(self.config.scan_interval);
        let mut refresh_ticker = interval(self.config.refresh_interval);

        loop {
            tokio::select! {
                _ = scan_ticker.tick() => {
                    if !*self.running.read().await {
                        break;
                    }
                    self.scan_and_execute().await;
                }
                _ = refresh_ticker.tick() => {
                    if !*self.running.read().await {
                        break;
                    }
                    debug!("[ArbLoop] Refreshing markets...");
                }
            }
        }

        info!("[ArbLoop] Stopped");
        Ok(())
    }

    /// Stop the loop
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        
        if self.config.notify_enabled {
            let stats = self.stats.read().await;
            let _ = self.notifier.send(&format!(
                "🛑 <b>Arbitrage Bot Stopped</b>\n\n\
                Scans: {}\n\
                Opportunities: {}\n\
                Trades: {} ({} successful)\n\
                Profit: ${:.4}",
                stats.scans,
                stats.opportunities_found,
                stats.trades_executed,
                stats.trades_successful,
                stats.total_profit
            )).await;
        }
    }

    /// Scan for opportunities and execute if found
    async fn scan_and_execute(&self) {
        // Scan 15m markets
        let markets = self.discovery.get_all_current_markets(MarketInterval::Min15).await;
        
        {
            let mut stats = self.stats.write().await;
            stats.scans += 1;
        }

        for market in markets {
            // Check for arbitrage
            if market.spread > self.config.min_spread {
                debug!(
                    "[ArbLoop] Opportunity: {} spread={:.2}%",
                    market.symbol,
                    market.spread * dec!(100)
                );

                {
                    let mut stats = self.stats.write().await;
                    stats.opportunities_found += 1;
                }

                // Create opportunity struct
                let opp = ArbitrageOpp {
                    condition_id: market.condition_id.clone(),
                    question: market.title.clone(),
                    slug: market.slug.clone(),
                    yes_token_id: market.up_token_id.clone(),
                    no_token_id: market.down_token_id.clone(),
                    yes_ask: market.up_price,
                    no_ask: market.down_price,
                    total_cost: market.sum,
                    spread: market.spread,
                    max_size: 100, // Default size
                    profit_margin: market.spread / market.sum * dec!(100),
                    net_profit: market.spread * self.config.max_position,
                    confidence: dec!(0.8),
                    detected_at: Utc::now(),
                };

                // Notify about opportunity
                if self.config.notify_enabled {
                    let _ = self.notifier.arbitrage_found(
                        &market.title,
                        market.up_price,
                        market.down_price,
                        market.spread,
                        opp.net_profit,
                    ).await;
                }

                // Execute
                match self.executor.execute(&opp).await {
                    Ok(result) => {
                        let mut stats = self.stats.write().await;
                        stats.trades_executed += 1;
                        if result.success {
                            stats.trades_successful += 1;
                            if let Some(profit) = result.actual_profit {
                                stats.total_profit += profit;
                            }
                        }

                        if self.config.notify_enabled {
                            let _ = self.notifier.arbitrage_executed(
                                &market.title,
                                result.success,
                                result.actual_profit,
                                result.latency_ms,
                                result.error.as_deref(),
                            ).await;
                        }

                        info!(
                            "[ArbLoop] Executed: {} success={} profit={:?}",
                            market.symbol, result.success, result.actual_profit
                        );
                    }
                    Err(e) => {
                        error!("[ArbLoop] Execution failed: {}", e);
                    }
                }
            }
        }
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> LoopStats {
        let stats = self.stats.read().await;
        LoopStats {
            scans: stats.scans,
            opportunities_found: stats.opportunities_found,
            trades_executed: stats.trades_executed,
            trades_successful: stats.trades_successful,
            total_profit: stats.total_profit,
            start_time: stats.start_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ArbitrageLoopConfig::default();
        assert_eq!(config.min_spread, dec!(0.005));
        assert!(!config.live_trading);
    }

    #[test]
    fn test_stats_default() {
        let stats = LoopStats::default();
        assert_eq!(stats.scans, 0);
        assert_eq!(stats.total_profit, dec!(0));
    }
}
