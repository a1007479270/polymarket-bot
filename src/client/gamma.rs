//! Gamma API client for market data
//!
//! Fetches market information, prices, and metadata.

use crate::error::{BotError, Result};
use crate::types::{Market, Outcome};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use tracing::debug;

/// Known crypto series IDs
pub const CRYPTO_SERIES: &[(&str, &str, u64)] = &[
    ("BTC 15m", "btc-up-or-down-15m", 10192),
    ("BTC 4H", "bitcoin-up-or-down-4h", 10323),
    ("ETH 5m", "eth-up-or-down-5m", 10683),
    ("SOL Hourly", "solana-up-or-down-hourly", 10122),
    ("SOL Daily", "solana-up-or-down-daily", 10086),
    ("XRP Hourly", "xrp-up-or-down-hourly", 10123),
];

/// Dynamic search queries for hourly crypto markets
/// These markets are created dynamically with format: bitcoin-up-or-down-{month}-{day}-{hour}pm-et
pub const CRYPTO_SEARCH_QUERIES: &[&str] = &[
    "bitcoin up or down",
    "ethereum up or down", 
    "solana up or down",
    "xrp up or down",
];

/// Gamma API client for market data
#[derive(Clone)]
pub struct GammaClient {
    http: Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct GammaMarket {
    id: String,
    question: String,
    description: Option<String>,
    #[serde(rename = "endDate")]
    end_date: Option<String>,
    volume: Option<String>,
    liquidity: Option<String>,
    active: bool,
    closed: bool,
    outcomes: Option<String>, // JSON string
    #[serde(rename = "outcomePrices")]
    outcome_prices: Option<String>, // JSON string "[0.55, 0.45]"
    #[serde(rename = "clobTokenIds")]
    clob_token_ids: Option<String>, // JSON string
}

impl GammaClient {
    /// Create a new Gamma client
    pub fn new(base_url: &str) -> Result<Self> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Get all active markets
    pub async fn get_markets(&self) -> Result<Vec<Market>> {
        let url = format!("{}/markets", self.base_url);
        let resp: Vec<GammaMarket> = self
            .http
            .get(&url)
            .query(&[("active", "true"), ("closed", "false")])
            .send()
            .await?
            .json()
            .await?;

        Ok(resp.into_iter().filter_map(|m| self.parse_market(m)).collect())
    }

    /// Get a specific market by ID
    pub async fn get_market(&self, market_id: &str) -> Result<Market> {
        let url = format!("{}/markets/{}", self.base_url, market_id);
        let resp: GammaMarket = self.http.get(&url).send().await?.json().await?;

        self.parse_market(resp)
            .ok_or_else(|| BotError::MarketNotFound(market_id.to_string()))
    }

    /// Search markets by keyword
    pub async fn search_markets(&self, query: &str) -> Result<Vec<Market>> {
        let url = format!("{}/markets", self.base_url);
        let resp: Vec<GammaMarket> = self
            .http
            .get(&url)
            .query(&[("_q", query)])
            .send()
            .await?
            .json()
            .await?;

        Ok(resp.into_iter().filter_map(|m| self.parse_market(m)).collect())
    }

    /// Get markets by volume (top markets)
    pub async fn get_top_markets(&self, limit: usize) -> Result<Vec<Market>> {
        let url = format!("{}/markets", self.base_url);
        let resp: Vec<GammaMarket> = self
            .http
            .get(&url)
            .query(&[
                ("active", "true"),
                ("closed", "false"),
                ("_sort", "volume:desc"),
                ("_limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        Ok(resp.into_iter().filter_map(|m| self.parse_market(m)).collect())
    }

    fn parse_market(&self, gm: GammaMarket) -> Option<Market> {
        // Parse outcome prices - API returns string array like ["0.55", "0.45"]
        let prices: Vec<f64> = gm
            .outcome_prices
            .as_ref()
            .and_then(|s| {
                // Try parsing as Vec<String> first (API format)
                if let Ok(string_prices) = serde_json::from_str::<Vec<String>>(s) {
                    let parsed: Vec<f64> = string_prices
                        .iter()
                        .filter_map(|p| p.parse::<f64>().ok())
                        .collect();
                    if !parsed.is_empty() {
                        return Some(parsed);
                    }
                }
                // Fallback: try parsing as Vec<f64> directly
                serde_json::from_str(s).ok()
            })
            .unwrap_or_default();

        // Parse token IDs
        let token_ids: Vec<String> = gm
            .clob_token_ids
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        // Parse outcome names
        let outcome_names: Vec<String> = gm
            .outcomes
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_else(|| vec!["Yes".to_string(), "No".to_string()]);

        // Build outcomes
        let outcomes: Vec<Outcome> = outcome_names
            .into_iter()
            .enumerate()
            .map(|(i, name)| Outcome {
                token_id: token_ids.get(i).cloned().unwrap_or_default(),
                outcome: name,
                price: prices
                    .get(i)
                    .map(|&p| Decimal::try_from(p).unwrap_or(Decimal::ZERO))
                    .unwrap_or(Decimal::ZERO),
            })
            .collect();

        Some(Market {
            id: gm.id,
            question: gm.question,
            description: gm.description,
            end_date: gm.end_date.as_ref().and_then(|s| s.parse().ok()),
            volume: gm
                .volume
                .as_ref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Decimal::ZERO),
            liquidity: gm
                .liquidity
                .as_ref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Decimal::ZERO),
            outcomes,
            active: gm.active,
            closed: gm.closed,
        })
    }

    /// Get active crypto markets (BTC/ETH Up/Down)
    /// Combines static series + dynamic search for hourly markets
    pub async fn get_crypto_markets(&self) -> Result<Vec<Market>> {
        let mut markets = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        // 1. Fetch from known series (static)
        for (name, _slug, series_id) in CRYPTO_SERIES {
            debug!("Fetching {} markets from series {}", name, series_id);
            
            // Get series with events
            let url = format!("{}/series/{}", self.base_url, series_id);
            let resp = self.http.get(&url).send().await?;
            
            if !resp.status().is_success() {
                debug!("Failed to fetch series {}: {}", series_id, resp.status());
                continue;
            }

            let series: SeriesResponse = match resp.json().await {
                Ok(s) => s,
                Err(e) => {
                    debug!("Failed to parse series {}: {}", series_id, e);
                    continue;
                }
            };

            // Get active, non-closed events
            let active_events: Vec<_> = series
                .events
                .into_iter()
                .filter(|e| e.active && !e.closed)
                .take(5) // Limit to 5 most recent events per series
                .collect();

            for event in active_events {
                // Fetch full event data with markets
                let event_url = format!("{}/events/{}", self.base_url, event.id);
                let event_resp = match self.http.get(&event_url).send().await {
                    Ok(r) => r,
                    Err(_) => continue,
                };

                let full_event: EventResponse = match event_resp.json().await {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                // Parse markets from event
                if let Some(event_markets) = full_event.markets {
                    for em in event_markets {
                        if let Some(market) = self.parse_market(em) {
                            if seen_ids.insert(market.id.clone()) {
                                markets.push(market);
                            }
                        }
                    }
                }
            }
        }

        // 2. Dynamic search for hourly markets (catches bitcoin-up-or-down-january-29-5pm-et etc.)
        let dynamic_markets = self.search_crypto_hourly_markets().await?;
        for market in dynamic_markets {
            if seen_ids.insert(market.id.clone()) {
                markets.push(market);
            }
        }

        debug!("Found {} total crypto markets (series + dynamic)", markets.len());
        Ok(markets)
    }

    /// Search for dynamic hourly crypto markets
    /// These are created with slug format: {coin}-up-or-down-{month}-{day}-{hour}pm-et
    pub async fn search_crypto_hourly_markets(&self) -> Result<Vec<Market>> {
        let mut markets = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for query in CRYPTO_SEARCH_QUERIES {
            debug!("Searching for dynamic crypto markets: {}", query);
            
            let url = format!("{}/markets", self.base_url);
            let resp = self.http
                .get(&url)
                .query(&[
                    ("_q", *query),
                    ("active", "true"),
                    ("closed", "false"),
                    ("_limit", "50"), // Get more results to catch hourly markets
                ])
                .send()
                .await;

            let resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    debug!("Search failed for '{}': {}", query, e);
                    continue;
                }
            };

            let results: Vec<GammaMarket> = match resp.json().await {
                Ok(r) => r,
                Err(e) => {
                    debug!("Failed to parse search results for '{}': {}", query, e);
                    continue;
                }
            };

            for gm in results {
                // Filter: must contain "up or down" in question
                let q_lower = gm.question.to_lowercase();
                if !q_lower.contains("up or down") {
                    continue;
                }
                
                // Filter: must be crypto related
                let is_crypto = q_lower.contains("bitcoin") 
                    || q_lower.contains("btc")
                    || q_lower.contains("ethereum")
                    || q_lower.contains("eth")
                    || q_lower.contains("solana")
                    || q_lower.contains("sol")
                    || q_lower.contains("xrp");
                
                if !is_crypto {
                    continue;
                }

                if let Some(market) = self.parse_market(gm) {
                    if seen_ids.insert(market.id.clone()) {
                        debug!("Found dynamic market: {} (liq: ${})", 
                            market.question, market.liquidity);
                        markets.push(market);
                    }
                }
            }
        }

        debug!("Found {} dynamic hourly crypto markets", markets.len());
        Ok(markets)
    }

    /// Get markets ending soon (within N hours) for timing-sensitive strategies
    pub async fn get_markets_ending_soon(&self, hours: u32) -> Result<Vec<Market>> {
        let markets = self.search_crypto_hourly_markets().await?;
        let now = chrono::Utc::now();
        let cutoff = now + chrono::Duration::hours(hours as i64);
        
        Ok(markets
            .into_iter()
            .filter(|m| {
                m.end_date
                    .map(|end| end <= cutoff && end > now)
                    .unwrap_or(false)
            })
            .collect())
    }
}

/// Response structure for series endpoint
#[derive(Debug, Deserialize)]
struct SeriesResponse {
    #[allow(dead_code)]
    title: String,
    events: Vec<SeriesEvent>,
}

#[derive(Debug, Deserialize)]
struct SeriesEvent {
    id: String,
    #[allow(dead_code)]
    title: String,
    active: bool,
    closed: bool,
}

/// Response structure for event endpoint
#[derive(Debug, Deserialize)]
struct EventResponse {
    #[allow(dead_code)]
    title: String,
    markets: Option<Vec<GammaMarket>>,
}
