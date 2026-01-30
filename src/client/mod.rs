//! Polymarket API client
//!
//! This module provides interfaces to interact with Polymarket's APIs:
//! - CLOB API: Order placement, cancellation, and management
//! - Gamma API: Market data and information
//! - WebSocket: Real-time price updates via polymarket_ws
//! - Mock: Test clients for offline testing

pub mod clob;
pub mod gamma;
mod auth;
pub mod polymarket_ws;
pub mod mock;
pub mod orderbook_stream;
#[cfg(test)]
mod tests;

pub use clob::{ClobClient, OrderBook, OrderBookLevel};
pub use gamma::{GammaClient, CRYPTO_SERIES, CRYPTO_SEARCH_QUERIES};
pub use auth::PolySigner;
pub use orderbook_stream::{OrderBookManager, OrderBookUpdate, LocalOrderBook};

// WebSocket implementation based on official Polymarket docs
pub use polymarket_ws::{
    MarketWsClient, WsConfig, MarketEvent, PriceStream,
    BookMessage, PriceChangeMessage, LastTradePriceMessage,
    OrderLevel, PriceChange, MarketUpdate,
    ConnectionState, WS_BASE_URL, MARKET_CHANNEL,
};

use crate::config::PolymarketConfig;
use crate::error::Result;

/// Unified Polymarket client
pub struct PolymarketClient {
    pub clob: ClobClient,
    pub gamma: GammaClient,
    config: PolymarketConfig,
}

impl PolymarketClient {
    /// Create a new Polymarket client
    pub async fn new(config: PolymarketConfig) -> Result<Self> {
        let signer = PolySigner::from_private_key(&config.private_key, config.chain_id)?;
        let clob = ClobClient::new(&config.clob_url, signer, config.funder_address.clone())?;
        let gamma = GammaClient::new(&config.gamma_url)?;

        Ok(Self { clob, gamma, config })
    }

    /// Create a WebSocket stream for real-time market data
    ///
    /// Uses the official Polymarket WebSocket protocol with:
    /// - Proper message format (assets_ids, type: market)
    /// - Keep-alive PING every 10 seconds
    /// - Auto-reconnect with exponential backoff
    pub async fn market_stream(&self, token_ids: Vec<String>) -> Result<PriceStream> {
        PriceStream::connect(WsConfig::default(), token_ids).await
    }
}
