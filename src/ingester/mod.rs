//! Signal ingestion from external sources
//!
//! Collects market signals from:
//! - Telegram groups (alpha channels)
//! - Twitter/X (KOL accounts)
//! - On-chain data (whale movements)

pub mod source;
pub mod telegram;
pub mod twitter;
pub mod processor;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod processor_tests;

use crate::error::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Raw signal from any source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSignal {
    /// Source type: "telegram", "twitter", "chain"
    pub source: String,
    /// Original message/event ID
    pub source_id: String,
    /// Raw content
    pub content: String,
    /// Author identifier
    pub author: String,
    /// Author trust score (0.0 - 1.0, based on historical accuracy)
    pub author_trust: f64,
    /// When the signal was captured
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Parsed signal after LLM extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSignal {
    /// Token/asset mentioned (BTC, ETH, SOL, etc.)
    pub token: String,
    /// Direction: bullish, bearish, neutral
    pub direction: SignalDirection,
    /// Timeframe: 5m, 1h, 1d
    pub timeframe: String,
    /// LLM confidence in extraction (0.0 - 1.0)
    pub confidence: f64,
    /// Extracted reasoning
    pub reasoning: String,
    /// Action type: entry, exit, warning
    pub action_type: ActionType,
    /// Original raw signals that contributed
    pub sources: Vec<RawSignal>,
    /// Aggregated score after multi-source validation
    pub agg_score: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalDirection {
    Bullish,
    Bearish,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionType {
    Entry,
    Exit,
    Warning,
    Info,
}

/// Signal source trait
#[async_trait]
pub trait SignalSource: Send + Sync {
    /// Source name
    fn name(&self) -> &str;
    
    /// Start listening and send signals to channel
    async fn run(&self, tx: mpsc::Sender<RawSignal>) -> Result<()>;
}

/// Ingester configuration
#[derive(Debug, Clone, Deserialize)]
pub struct IngesterConfig {
    /// Telegram configuration
    pub telegram: Option<TelegramIngesterConfig>,
    /// Twitter configuration  
    pub twitter: Option<TwitterIngesterConfig>,
    /// Author trust scores (author_id -> score)
    #[serde(default)]
    pub author_trust: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramIngesterConfig {
    /// Telegram API ID
    pub api_id: i32,
    /// Telegram API Hash
    pub api_hash: String,
    /// Session file path
    pub session_file: String,
    /// Chat IDs to monitor
    pub watch_chats: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TwitterIngesterConfig {
    /// Bearer token for Twitter API
    pub bearer_token: Option<String>,
    /// User IDs to monitor
    pub watch_users: Vec<String>,
    /// Keywords to filter
    pub keywords: Vec<String>,
}
