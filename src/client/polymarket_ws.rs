//! Polymarket WebSocket Client
//!
//! Clean implementation based on official documentation:
//! - https://docs.polymarket.com/quickstart/websocket/WSS-Quickstart
//! - https://docs.polymarket.com/developers/CLOB/websocket/market-channel
//!
//! Protocol:
//! - URL: wss://ws-subscriptions-clob.polymarket.com/ws/market
//! - Subscribe: {"assets_ids": [...], "type": "market"}
//! - Keep-alive: Send "PING" every 10 seconds

use crate::error::{BotError, Result};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

// =============================================================================
// Constants
// =============================================================================

/// Polymarket WebSocket base URL
pub const WS_BASE_URL: &str = "wss://ws-subscriptions-clob.polymarket.com";

/// Market channel endpoint
pub const MARKET_CHANNEL: &str = "market";

/// User channel endpoint (requires auth)
pub const USER_CHANNEL: &str = "user";

/// Ping interval (Polymarket requires every 10 seconds)
const PING_INTERVAL_SECS: u64 = 10;

/// Connection timeout
const CONNECT_TIMEOUT_SECS: u64 = 10;

/// Read timeout (detect stale connections)
const READ_TIMEOUT_SECS: u64 = 60;

/// Reconnect delays
const INITIAL_RECONNECT_DELAY_MS: u64 = 1000;
const MAX_RECONNECT_DELAY_MS: u64 = 60000;

// =============================================================================
// Message Types (from official docs)
// =============================================================================

/// Order level in the book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    pub price: String,
    pub size: String,
}

impl OrderLevel {
    pub fn price_decimal(&self) -> Option<Decimal> {
        Decimal::from_str(&self.price).ok()
    }

    pub fn size_decimal(&self) -> Option<Decimal> {
        Decimal::from_str(&self.size).ok()
    }
}

/// Book message - full orderbook snapshot
#[derive(Debug, Clone, Deserialize)]
pub struct BookMessage {
    pub event_type: String,
    pub asset_id: String,
    pub market: String,
    #[serde(default)]
    pub bids: Vec<OrderLevel>,
    #[serde(default)]
    pub asks: Vec<OrderLevel>,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub hash: String,
}

/// Price change entry
#[derive(Debug, Clone, Deserialize)]
pub struct PriceChange {
    pub asset_id: String,
    pub price: String,
    pub size: String,
    pub side: String, // "BUY" or "SELL"
    #[serde(default)]
    pub hash: String,
    #[serde(default)]
    pub best_bid: String,
    #[serde(default)]
    pub best_ask: String,
}

/// Price change message - order book updates
#[derive(Debug, Clone, Deserialize)]
pub struct PriceChangeMessage {
    pub event_type: String,
    pub market: String,
    pub price_changes: Vec<PriceChange>,
    #[serde(default)]
    pub timestamp: String,
}

/// Last trade price message
#[derive(Debug, Clone, Deserialize)]
pub struct LastTradePriceMessage {
    pub event_type: String,
    pub asset_id: String,
    pub market: String,
    pub price: String,
    pub size: String,
    pub side: String,
    #[serde(default)]
    pub fee_rate_bps: String,
    #[serde(default)]
    pub timestamp: String,
}

/// Tick size change message
#[derive(Debug, Clone, Deserialize)]
pub struct TickSizeChangeMessage {
    pub event_type: String,
    pub asset_id: String,
    pub market: String,
    pub old_tick_size: String,
    pub new_tick_size: String,
    #[serde(default)]
    pub timestamp: String,
}

/// All possible market channel events
#[derive(Debug, Clone)]
pub enum MarketEvent {
    /// Full orderbook snapshot
    Book(BookMessage),
    /// Price level changes
    PriceChange(PriceChangeMessage),
    /// Trade executed
    LastTradePrice(LastTradePriceMessage),
    /// Tick size changed
    TickSizeChange(TickSizeChangeMessage),
    /// Unknown event type
    Unknown(String),
}

// =============================================================================
// Simplified Market Update (for consumers who just want price data)
// =============================================================================

/// Simplified market update for price tracking
#[derive(Debug, Clone)]
pub struct MarketUpdate {
    /// Token ID (asset ID) - primary identifier
    pub token_id: String,
    /// Condition ID (market)
    pub market_id: String,
    /// Best bid price
    pub best_bid: Option<Decimal>,
    /// Best ask price
    pub best_ask: Option<Decimal>,
    /// Last trade price
    pub last_price: Option<Decimal>,
    /// Last trade size
    pub last_size: Option<Decimal>,
    /// Timestamp (unix ms)
    pub timestamp: u64,
    /// Event type that triggered this update
    pub event_type: String,
}

impl MarketUpdate {
    /// Alias for token_id (Polymarket calls it asset_id)
    #[inline]
    pub fn asset_id(&self) -> &str {
        &self.token_id
    }
}

// =============================================================================
// Client Configuration
// =============================================================================

/// WebSocket client configuration
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// Base URL (default: wss://ws-subscriptions-clob.polymarket.com)
    pub base_url: String,
    /// Ping interval in seconds (default: 10)
    pub ping_interval_secs: u64,
    /// Connection timeout in seconds (default: 10)
    pub connect_timeout_secs: u64,
    /// Read timeout in seconds (default: 60)
    pub read_timeout_secs: u64,
    /// Max reconnect attempts (0 = unlimited)
    pub max_reconnect_attempts: u32,
    /// Channel buffer size
    pub channel_buffer_size: usize,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            base_url: WS_BASE_URL.to_string(),
            ping_interval_secs: PING_INTERVAL_SECS,
            connect_timeout_secs: CONNECT_TIMEOUT_SECS,
            read_timeout_secs: READ_TIMEOUT_SECS,
            max_reconnect_attempts: 0,
            channel_buffer_size: 10000,
        }
    }
}

// =============================================================================
// Connection State
// =============================================================================

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
}

// =============================================================================
// Market WebSocket Client
// =============================================================================

/// Polymarket Market Channel WebSocket Client
///
/// # Example
/// ```ignore
/// let mut client = MarketWsClient::new(WsConfig::default());
/// let mut rx = client.connect(vec!["token_id_1".to_string()]).await?;
///
/// while let Some(event) = rx.recv().await {
///     match event {
///         MarketEvent::Book(book) => println!("Book: {:?}", book),
///         MarketEvent::PriceChange(pc) => println!("Price change: {:?}", pc),
///         _ => {}
///     }
/// }
/// ```
pub struct MarketWsClient {
    config: WsConfig,
    connected: Arc<AtomicBool>,
    reconnect_count: Arc<AtomicU64>,
    shutdown_tx: Option<broadcast::Sender<()>>,
    subscribe_tx: Option<mpsc::Sender<SubscribeCommand>>,
}

#[derive(Debug)]
enum SubscribeCommand {
    Subscribe(Vec<String>),
    Unsubscribe(Vec<String>),
}

impl MarketWsClient {
    /// Create a new market WebSocket client
    pub fn new(config: WsConfig) -> Self {
        Self {
            config,
            connected: Arc::new(AtomicBool::new(false)),
            reconnect_count: Arc::new(AtomicU64::new(0)),
            shutdown_tx: None,
            subscribe_tx: None,
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Get reconnection count
    pub fn reconnect_count(&self) -> u64 {
        self.reconnect_count.load(Ordering::Relaxed)
    }

    /// Connect to the market channel
    ///
    /// Returns a receiver for market events
    pub async fn connect(&mut self, asset_ids: Vec<String>) -> Result<mpsc::Receiver<MarketEvent>> {
        let (event_tx, event_rx) = mpsc::channel(self.config.channel_buffer_size);
        let (shutdown_tx, _) = broadcast::channel(1);
        let (subscribe_tx, subscribe_rx) = mpsc::channel(100);

        self.shutdown_tx = Some(shutdown_tx.clone());
        self.subscribe_tx = Some(subscribe_tx);

        let config = self.config.clone();
        let connected = Arc::clone(&self.connected);
        let reconnect_count = Arc::clone(&self.reconnect_count);

        tokio::spawn(async move {
            connection_loop(
                config,
                asset_ids,
                event_tx,
                subscribe_rx,
                shutdown_tx,
                connected,
                reconnect_count,
            )
            .await;
        });

        // Wait for initial connection
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(self.config.connect_timeout_secs) {
            if self.is_connected() {
                return Ok(event_rx);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Err(BotError::WebSocket("Connection timeout".to_string()))
    }

    /// Subscribe to additional assets
    pub async fn subscribe(&self, asset_ids: Vec<String>) -> Result<()> {
        if let Some(tx) = &self.subscribe_tx {
            tx.send(SubscribeCommand::Subscribe(asset_ids))
                .await
                .map_err(|_| BotError::WebSocket("Subscribe channel closed".to_string()))?;
        }
        Ok(())
    }

    /// Unsubscribe from assets
    pub async fn unsubscribe(&self, asset_ids: Vec<String>) -> Result<()> {
        if let Some(tx) = &self.subscribe_tx {
            tx.send(SubscribeCommand::Unsubscribe(asset_ids))
                .await
                .map_err(|_| BotError::WebSocket("Subscribe channel closed".to_string()))?;
        }
        Ok(())
    }

    /// Shutdown the connection
    pub fn shutdown(&self) {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }
    }
}

impl Drop for MarketWsClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// =============================================================================
// Connection Loop
// =============================================================================

async fn connection_loop(
    config: WsConfig,
    initial_assets: Vec<String>,
    event_tx: mpsc::Sender<MarketEvent>,
    mut subscribe_rx: mpsc::Receiver<SubscribeCommand>,
    shutdown_tx: broadcast::Sender<()>,
    connected: Arc<AtomicBool>,
    reconnect_count: Arc<AtomicU64>,
) {
    let mut shutdown_rx = shutdown_tx.subscribe();
    let mut attempt = 0u32;
    let mut delay_ms = INITIAL_RECONNECT_DELAY_MS;
    let mut current_assets = initial_assets;

    loop {
        // Check for shutdown
        if shutdown_rx.try_recv().is_ok() {
            info!("WebSocket shutdown requested");
            break;
        }

        info!(
            "Connecting to Polymarket WebSocket (attempt {})",
            attempt + 1
        );

        match connect_once(
            &config,
            &current_assets,
            &event_tx,
            &mut subscribe_rx,
            &mut shutdown_rx,
            &connected,
        )
        .await
        {
            Ok(updated_assets) => {
                // Normal disconnect, update assets and reset backoff
                current_assets = updated_assets;
                attempt = 0;
                delay_ms = INITIAL_RECONNECT_DELAY_MS;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                attempt += 1;

                if config.max_reconnect_attempts > 0 && attempt >= config.max_reconnect_attempts {
                    error!("Max reconnect attempts reached");
                    break;
                }
            }
        }

        connected.store(false, Ordering::Relaxed);
        reconnect_count.fetch_add(1, Ordering::Relaxed);

        info!("Reconnecting in {}ms...", delay_ms);
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
            _ = shutdown_rx.recv() => {
                info!("Shutdown during backoff");
                break;
            }
        }

        // Exponential backoff
        delay_ms = (delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
    }

    connected.store(false, Ordering::Relaxed);
}

async fn connect_once(
    config: &WsConfig,
    asset_ids: &[String],
    event_tx: &mpsc::Sender<MarketEvent>,
    subscribe_rx: &mut mpsc::Receiver<SubscribeCommand>,
    shutdown_rx: &mut broadcast::Receiver<()>,
    connected: &Arc<AtomicBool>,
) -> Result<Vec<String>> {
    // Build WebSocket URL
    let ws_url = format!("{}/ws/{}", config.base_url, MARKET_CHANNEL);

    // Connect with timeout
    let (ws_stream, _) = timeout(
        Duration::from_secs(config.connect_timeout_secs),
        connect_async(&ws_url),
    )
    .await
    .map_err(|_| BotError::WebSocket("Connection timeout".to_string()))?
    .map_err(|e| BotError::WebSocket(format!("Connection failed: {}", e)))?;

    let (mut write, mut read) = ws_stream.split();

    // Send initial subscription (official format)
    let subscribe_msg = serde_json::json!({
        "assets_ids": asset_ids,
        "type": MARKET_CHANNEL
    });

    write
        .send(Message::Text(subscribe_msg.to_string().into()))
        .await
        .map_err(|e| BotError::WebSocket(format!("Subscribe failed: {}", e)))?;

    info!(
        "Connected to Polymarket WebSocket, subscribed to {} assets",
        asset_ids.len()
    );
    connected.store(true, Ordering::Relaxed);

    // Track current subscriptions
    let mut current_assets: Vec<String> = asset_ids.to_vec();

    // Ping interval (Polymarket requires "PING" text every 10 seconds)
    let mut ping_interval = interval(Duration::from_secs(config.ping_interval_secs));
    ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let read_timeout = Duration::from_secs(config.read_timeout_secs);

    loop {
        tokio::select! {
            // Shutdown signal
            _ = shutdown_rx.recv() => {
                info!("Shutdown signal received");
                return Ok(current_assets);
            }

            // Ping timer (send "PING" text message per Polymarket protocol)
            _ = ping_interval.tick() => {
                debug!("Sending PING");
                if let Err(e) = write.send(Message::Text("PING".into())).await {
                    warn!("Failed to send PING: {}", e);
                    return Err(BotError::WebSocket("Ping failed".to_string()));
                }
            }

            // Subscribe/unsubscribe commands
            cmd = subscribe_rx.recv() => {
                match cmd {
                    Some(SubscribeCommand::Subscribe(ids)) => {
                        let msg = serde_json::json!({
                            "assets_ids": ids,
                            "operation": "subscribe"
                        });
                        if let Err(e) = write.send(Message::Text(msg.to_string().into())).await {
                            warn!("Failed to subscribe: {}", e);
                        } else {
                            current_assets.extend(ids);
                            debug!("Subscribed, total assets: {}", current_assets.len());
                        }
                    }
                    Some(SubscribeCommand::Unsubscribe(ids)) => {
                        let msg = serde_json::json!({
                            "assets_ids": ids,
                            "operation": "unsubscribe"
                        });
                        if let Err(e) = write.send(Message::Text(msg.to_string().into())).await {
                            warn!("Failed to unsubscribe: {}", e);
                        } else {
                            current_assets.retain(|a| !ids.contains(a));
                            debug!("Unsubscribed, total assets: {}", current_assets.len());
                        }
                    }
                    None => {
                        // Subscribe channel closed
                        return Ok(current_assets);
                    }
                }
            }

            // Read messages with timeout
            msg = timeout(read_timeout, read.next()) => {
                match msg {
                    Ok(Some(Ok(Message::Text(text)))) => {
                        if let Some(event) = parse_message(&text) {
                            if event_tx.send(event).await.is_err() {
                                info!("Event channel closed");
                                return Ok(current_assets);
                            }
                        }
                    }
                    Ok(Some(Ok(Message::Ping(data)))) => {
                        // Respond to server ping
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Ok(Some(Ok(Message::Close(_)))) => {
                        info!("Server closed connection");
                        return Ok(current_assets);
                    }
                    Ok(Some(Err(e))) => {
                        return Err(BotError::WebSocket(format!("Read error: {}", e)));
                    }
                    Ok(None) => {
                        info!("Stream ended");
                        return Ok(current_assets);
                    }
                    Err(_) => {
                        warn!("Read timeout - connection stale");
                        return Err(BotError::WebSocket("Read timeout".to_string()));
                    }
                    _ => {}
                }
            }
        }
    }
}

// =============================================================================
// Message Parsing
// =============================================================================

fn parse_message(text: &str) -> Option<MarketEvent> {
    // First, detect event type
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let event_type = value.get("event_type")?.as_str()?;

    match event_type {
        "book" => serde_json::from_str::<BookMessage>(text)
            .ok()
            .map(MarketEvent::Book),

        "price_change" => serde_json::from_str::<PriceChangeMessage>(text)
            .ok()
            .map(MarketEvent::PriceChange),

        "last_trade_price" => serde_json::from_str::<LastTradePriceMessage>(text)
            .ok()
            .map(MarketEvent::LastTradePrice),

        "tick_size_change" => serde_json::from_str::<TickSizeChangeMessage>(text)
            .ok()
            .map(MarketEvent::TickSizeChange),

        _ => Some(MarketEvent::Unknown(text.to_string())),
    }
}

// =============================================================================
// Convenience: Price Stream
// =============================================================================

/// Simplified price update stream
///
/// Converts raw market events into simple price updates
pub struct PriceStream {
    event_rx: mpsc::Receiver<MarketEvent>,
    /// Last known prices per asset
    prices: HashMap<String, MarketUpdate>,
}

impl PriceStream {
    /// Create from a market event receiver
    pub fn new(event_rx: mpsc::Receiver<MarketEvent>) -> Self {
        Self {
            event_rx,
            prices: HashMap::new(),
        }
    }

    /// Connect and create a price stream
    pub async fn connect(config: WsConfig, asset_ids: Vec<String>) -> Result<Self> {
        let mut client = MarketWsClient::new(config);
        let event_rx = client.connect(asset_ids).await?;

        // Keep client alive by leaking it (it will run in background)
        std::mem::forget(client);

        Ok(Self::new(event_rx))
    }

    /// Receive next price update
    pub async fn recv(&mut self) -> Option<MarketUpdate> {
        loop {
            let event = self.event_rx.recv().await?;

            match event {
                MarketEvent::Book(book) => {
                    // Extract best bid/ask from book
                    let best_bid = book
                        .bids
                        .first()
                        .and_then(|l| Decimal::from_str(&l.price).ok());
                    let best_ask = book
                        .asks
                        .first()
                        .and_then(|l| Decimal::from_str(&l.price).ok());

                    let update = MarketUpdate {
                        token_id: book.asset_id.clone(),
                        market_id: book.market.clone(),
                        best_bid,
                        best_ask,
                        last_price: None,
                        last_size: None,
                        timestamp: book.timestamp.parse().unwrap_or(0),
                        event_type: "book".to_string(),
                    };

                    self.prices.insert(book.asset_id, update.clone());
                    return Some(update);
                }

                MarketEvent::PriceChange(pc) => {
                    // Process each price change
                    for change in &pc.price_changes {
                        let best_bid = Decimal::from_str(&change.best_bid).ok();
                        let best_ask = Decimal::from_str(&change.best_ask).ok();

                        let update = MarketUpdate {
                            token_id: change.asset_id.clone(),
                            market_id: pc.market.clone(),
                            best_bid,
                            best_ask,
                            last_price: None,
                            last_size: None,
                            timestamp: pc.timestamp.parse().unwrap_or(0),
                            event_type: "price_change".to_string(),
                        };

                        self.prices.insert(change.asset_id.clone(), update.clone());

                        // Return first change (caller can call recv again for more)
                        return Some(update);
                    }
                }

                MarketEvent::LastTradePrice(trade) => {
                    let last_price = Decimal::from_str(&trade.price).ok();
                    let last_size = Decimal::from_str(&trade.size).ok();

                    // Merge with existing price data
                    let mut update = self
                        .prices
                        .get(&trade.asset_id)
                        .cloned()
                        .unwrap_or_else(|| MarketUpdate {
                            token_id: trade.asset_id.clone(),
                            market_id: trade.market.clone(),
                            best_bid: None,
                            best_ask: None,
                            last_price: None,
                            last_size: None,
                            timestamp: 0,
                            event_type: String::new(),
                        });

                    update.last_price = last_price;
                    update.last_size = last_size;
                    update.timestamp = trade.timestamp.parse().unwrap_or(0);
                    update.event_type = "last_trade_price".to_string();

                    self.prices.insert(trade.asset_id.clone(), update.clone());
                    return Some(update);
                }

                _ => {
                    // Skip tick_size_change and unknown events
                    continue;
                }
            }
        }
    }

    /// Get last known price for an asset
    pub fn get_price(&self, asset_id: &str) -> Option<&MarketUpdate> {
        self.prices.get(asset_id)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_book_message() {
        let json = r#"{
            "event_type": "book",
            "asset_id": "12345",
            "market": "0xabc",
            "bids": [{"price": "0.48", "size": "30"}],
            "asks": [{"price": "0.52", "size": "25"}],
            "timestamp": "123456789000",
            "hash": "0x..."
        }"#;

        let event = parse_message(json).unwrap();
        match event {
            MarketEvent::Book(book) => {
                assert_eq!(book.asset_id, "12345");
                assert_eq!(book.bids.len(), 1);
                assert_eq!(book.bids[0].price, "0.48");
                assert_eq!(book.asks.len(), 1);
                assert_eq!(book.asks[0].price, "0.52");
            }
            _ => panic!("Expected Book event"),
        }
    }

    #[test]
    fn test_parse_price_change_message() {
        let json = r#"{
            "market": "0x5f65...",
            "price_changes": [
                {
                    "asset_id": "71321...",
                    "price": "0.5",
                    "size": "200",
                    "side": "BUY",
                    "hash": "56621a...",
                    "best_bid": "0.5",
                    "best_ask": "1"
                }
            ],
            "timestamp": "1757908892351",
            "event_type": "price_change"
        }"#;

        let event = parse_message(json).unwrap();
        match event {
            MarketEvent::PriceChange(pc) => {
                assert_eq!(pc.price_changes.len(), 1);
                assert_eq!(pc.price_changes[0].best_bid, "0.5");
                assert_eq!(pc.price_changes[0].best_ask, "1");
            }
            _ => panic!("Expected PriceChange event"),
        }
    }

    #[test]
    fn test_parse_last_trade_price_message() {
        let json = r#"{
            "asset_id": "114122...",
            "event_type": "last_trade_price",
            "fee_rate_bps": "0",
            "market": "0x6a67...",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "timestamp": "1750428146322"
        }"#;

        let event = parse_message(json).unwrap();
        match event {
            MarketEvent::LastTradePrice(trade) => {
                assert_eq!(trade.price, "0.456");
                assert_eq!(trade.size, "219.217767");
                assert_eq!(trade.side, "BUY");
            }
            _ => panic!("Expected LastTradePrice event"),
        }
    }

    #[test]
    fn test_order_level_decimal() {
        let level = OrderLevel {
            price: "0.55".to_string(),
            size: "100.5".to_string(),
        };
        assert_eq!(
            level.price_decimal(),
            Some(Decimal::from_str("0.55").unwrap())
        );
        assert_eq!(
            level.size_decimal(),
            Some(Decimal::from_str("100.5").unwrap())
        );
    }

    #[test]
    fn test_config_default() {
        let config = WsConfig::default();
        assert_eq!(config.base_url, WS_BASE_URL);
        assert_eq!(config.ping_interval_secs, 10);
        assert_eq!(config.connect_timeout_secs, 10);
    }

    #[test]
    fn test_market_ws_client_initial_state() {
        let client = MarketWsClient::new(WsConfig::default());
        assert!(!client.is_connected());
        assert_eq!(client.reconnect_count(), 0);
    }
}
