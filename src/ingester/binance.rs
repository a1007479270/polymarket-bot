//! Binance WebSocket integration for crypto market signals
//!
//! Monitors BTC/ETH price movements and generates signals for Polymarket.

use crate::error::Result;
use futures_util::StreamExt;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";

/// Price alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Minimum price change percentage to trigger alert
    pub min_change_pct: Decimal,
    /// Time window for price change calculation (seconds)
    pub window_secs: u64,
    /// Cooldown between alerts for same symbol (seconds)
    pub cooldown_secs: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            min_change_pct: Decimal::new(3, 0), // 3%
            window_secs: 300,                    // 5 minutes
            cooldown_secs: 600,                  // 10 minutes
        }
    }
}

/// Crypto price alert
#[derive(Debug, Clone)]
pub struct CryptoAlert {
    pub symbol: String,
    pub direction: AlertDirection,
    pub change_pct: Decimal,
    pub current_price: Decimal,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertDirection {
    Up,
    Down,
}

/// Binance ticker data
#[derive(Debug, Deserialize)]
struct TickerData {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "c")]
    close_price: String,
    #[serde(rename = "P")]
    price_change_pct: String,
}

/// Binance WebSocket monitor
pub struct BinanceMonitor {
    config: AlertConfig,
    symbols: Vec<String>,
    alert_tx: mpsc::Sender<CryptoAlert>,
}

impl BinanceMonitor {
    pub fn new(symbols: Vec<String>, alert_tx: mpsc::Sender<CryptoAlert>) -> Self {
        Self {
            config: AlertConfig::default(),
            symbols,
            alert_tx,
        }
    }

    pub fn with_config(mut self, config: AlertConfig) -> Self {
        self.config = config;
        self
    }

    /// Start monitoring Binance WebSocket
    pub async fn start(self) -> Result<()> {
        info!("Starting Binance WebSocket monitor for {:?}", self.symbols);

        // Build stream URL for multiple symbols
        let streams: Vec<String> = self
            .symbols
            .iter()
            .map(|s| format!("{}@ticker", s.to_lowercase()))
            .collect();
        let url = format!("{}/{}", BINANCE_WS_URL, streams.join("/"));

        loop {
            match self.connect_and_monitor(&url).await {
                Ok(_) => {
                    warn!("Binance WebSocket disconnected, reconnecting...");
                }
                Err(e) => {
                    error!("Binance WebSocket error: {}, reconnecting in 5s...", e);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }

    async fn connect_and_monitor(&self, url: &str) -> Result<()> {
        let (ws_stream, _) = connect_async(url)
            .await
            .map_err(|e| crate::error::BotError::WebSocket(e.to_string()))?;
        let (mut _write, mut read) = ws_stream.split();

        info!("Connected to Binance WebSocket");

        let mut last_alerts: std::collections::HashMap<String, Instant> =
            std::collections::HashMap::new();

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(ticker) = serde_json::from_str::<TickerData>(&text) {
                        self.process_ticker(&ticker, &mut last_alerts).await;
                    }
                }
                Ok(Message::Ping(_)) => {
                    debug!("Received ping, sending pong");
                    // Pong is handled automatically by tungstenite
                }
                Ok(Message::Close(_)) => {
                    warn!("WebSocket closed by server");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn process_ticker(
        &self,
        ticker: &TickerData,
        last_alerts: &mut std::collections::HashMap<String, Instant>,
    ) {
        let change_pct: Decimal = match ticker.price_change_pct.parse() {
            Ok(v) => v,
            Err(_) => return,
        };

        let current_price: Decimal = match ticker.close_price.parse() {
            Ok(v) => v,
            Err(_) => return,
        };

        let abs_change = if change_pct < Decimal::ZERO {
            -change_pct
        } else {
            change_pct
        };

        // Check if change exceeds threshold
        if abs_change >= self.config.min_change_pct {
            // Check cooldown
            if let Some(last_time) = last_alerts.get(&ticker.symbol) {
                if last_time.elapsed().as_secs() < self.config.cooldown_secs {
                    return;
                }
            }

            let direction = if change_pct > Decimal::ZERO {
                AlertDirection::Up
            } else {
                AlertDirection::Down
            };

            let alert = CryptoAlert {
                symbol: ticker.symbol.clone(),
                direction,
                change_pct,
                current_price,
                timestamp: Instant::now(),
            };

            info!(
                "🚨 Crypto Alert: {} {:?} {:.2}% @ ${:.2}",
                alert.symbol, alert.direction, alert.change_pct, alert.current_price
            );

            last_alerts.insert(ticker.symbol.clone(), Instant::now());

            if let Err(e) = self.alert_tx.send(alert).await {
                error!("Failed to send crypto alert: {}", e);
            }
        }
    }
}

/// Search Polymarket for crypto-related markets
pub fn get_crypto_keywords() -> Vec<&'static str> {
    vec![
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "solana", "sol", "dogecoin", "doge", "xrp", "ripple",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert_eq!(config.min_change_pct, Decimal::new(3, 0));
    }
}
