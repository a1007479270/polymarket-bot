//! Live Trading Binary - Crypto Hourly Markets
//!
//! Uses Rust's dynamic market discovery to find BTC/ETH hourly up/down markets.
//! Integrates Binance data for prediction and executes paper trades.
//!
//! Key features:
//! - Dynamic market discovery via search_crypto_hourly_markets()
//! - ML-based multi-factor prediction (technical + sentiment + orderbook)
//! - Binance kline integration for feature extraction
//! - Risk-controlled position sizing
//! - Real-time logging

use chrono::{DateTime, Duration, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{debug, info, warn};
use tracing_subscriber;

use polymarket_bot::client::gamma::GammaClient;
use polymarket_bot::types::Market;
use polymarket_bot::ml::predictor::{MLPredictor, MLPredictorConfig, MarketDataInput, KlineData};

const GAMMA_API_URL: &str = "https://gamma-api.polymarket.com";
const BINANCE_API_URL: &str = "https://api.binance.com";
const POLL_INTERVAL_SECS: u64 = 30;
const INITIAL_CAPITAL: f64 = 100.0;
const MIN_EDGE: f64 = 0.03; // 3%
const MAX_POSITION_PCT: f64 = 0.05; // 5% of capital
const MAX_TRADES_PER_HOUR: u32 = 5;
const MIN_LIQUIDITY: f64 = 5000.0; // $5k minimum liquidity
const MAX_SETTLEMENT_MINUTES: i64 = 30; // Only trade markets settling within 30 mins

/// Trade record for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Trade {
    id: String,
    timestamp: DateTime<Utc>,
    market_id: String,
    market_question: String,
    side: String,       // "Yes" or "No"
    entry_price: f64,
    amount: f64,        // USDC
    shares: f64,
    predicted_outcome: String,
    prediction_confidence: f64,
    binance_data: BinanceContext,
    status: TradeStatus,
    exit_price: Option<f64>,
    pnl: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TradeStatus {
    Open,
    Won,
    Lost,
    Expired,
}

/// Binance context for the trade
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinanceContext {
    symbol: String,
    current_price: f64,
    price_change_1h_pct: f64,
    volume_24h: f64,
    rsi_14: Option<f64>,
}

/// Binance kline data (reserved for future use)
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BinanceKline {
    open_time: i64,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: String,
    close_time: i64,
}

/// Extended Binance data with klines for ML features
#[derive(Debug, Clone)]
struct ExtendedBinanceData {
    pub context: BinanceContext,
    pub klines: Vec<KlineData>,
}

/// Live trader state
struct LiveTrader {
    gamma: GammaClient,
    http: Client,
    capital: f64,
    trades: Vec<Trade>,
    hourly_trade_count: u32,
    last_hour_reset: DateTime<Utc>,
    log_file: File,
    ml_predictor: MLPredictor,
}

impl LiveTrader {
    async fn new() -> anyhow::Result<Self> {
        let gamma = GammaClient::new(GAMMA_API_URL)?;
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        fs::create_dir_all("logs")?;
        let log_path = format!("logs/live_trading_{}.jsonl", Utc::now().format("%Y%m%d_%H%M%S"));
        let log_file = File::create(&log_path)?;

        // Initialize ML Predictor with default configuration
        let ml_config = MLPredictorConfig::default();
        let ml_predictor = MLPredictor::new(ml_config);

        info!("📁 Log file: {}", log_path);
        info!("🤖 ML Predictor initialized with multi-factor fusion");

        Ok(Self {
            gamma,
            http,
            capital: INITIAL_CAPITAL,
            trades: Vec::new(),
            hourly_trade_count: 0,
            last_hour_reset: Utc::now(),
            log_file,
            ml_predictor,
        })
    }

    fn log(&mut self, msg: &str) {
        let now = Utc::now().format("%H:%M:%S");
        println!("[{}] {}", now, msg);
        let _ = writeln!(self.log_file, "[{}] {}", now, msg);
    }

    fn log_trade(&mut self, trade: &Trade) {
        let json = serde_json::to_string(trade).unwrap_or_default();
        let _ = writeln!(self.log_file, "{}", json);
        let _ = self.log_file.flush();
    }

    /// Get Binance data for a coin
    async fn get_binance_data(&self, symbol: &str) -> anyhow::Result<BinanceContext> {
        // Get current price and 24h stats
        let ticker_url = format!("{}/api/v3/ticker/24hr?symbol={}", BINANCE_API_URL, symbol);
        let ticker: serde_json::Value = self.http.get(&ticker_url).send().await?.json().await?;

        let current_price = ticker["lastPrice"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let volume_24h = ticker["quoteVolume"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        // Get 1H klines for price change calculation
        let klines_url = format!(
            "{}/api/v3/klines?symbol={}&interval=1h&limit=2",
            BINANCE_API_URL, symbol
        );
        let klines: Vec<Vec<serde_json::Value>> = self.http.get(&klines_url).send().await?.json().await?;

        let price_change_1h_pct = if klines.len() >= 2 {
            let prev_close: f64 = klines[0][4]
                .as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(current_price);
            ((current_price - prev_close) / prev_close) * 100.0
        } else {
            0.0
        };

        // Calculate simple RSI from last 14 hours (simplified)
        let rsi = self.calculate_rsi(symbol, 14).await.ok();

        Ok(BinanceContext {
            symbol: symbol.to_string(),
            current_price,
            price_change_1h_pct,
            volume_24h,
            rsi_14: rsi,
        })
    }

    /// Calculate RSI from hourly klines
    async fn calculate_rsi(&self, symbol: &str, periods: usize) -> anyhow::Result<f64> {
        let klines_url = format!(
            "{}/api/v3/klines?symbol={}&interval=1h&limit={}",
            BINANCE_API_URL,
            symbol,
            periods + 1
        );
        let klines: Vec<Vec<serde_json::Value>> = self.http.get(&klines_url).send().await?.json().await?;

        if klines.len() < periods + 1 {
            anyhow::bail!("Not enough klines for RSI");
        }

        let closes: Vec<f64> = klines
            .iter()
            .map(|k| k[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0))
            .collect();

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..closes.len() {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let avg_gain = gains / periods as f64;
        let avg_loss = losses / periods as f64;

        if avg_loss == 0.0 {
            return Ok(100.0);
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        Ok(rsi)
    }

    /// Map market question to Binance symbol
    fn get_binance_symbol(&self, question: &str) -> Option<&'static str> {
        let q = question.to_lowercase();
        if q.contains("bitcoin") || q.contains("btc") {
            Some("BTCUSDT")
        } else if q.contains("ethereum") || q.contains("eth") {
            Some("ETHUSDT")
        } else if q.contains("solana") || q.contains("sol") {
            Some("SOLUSDT")
        } else if q.contains("xrp") {
            Some("XRPUSDT")
        } else {
            None
        }
    }

    /// Get extended Binance data with klines for ML features
    async fn get_extended_binance_data(&self, symbol: &str) -> anyhow::Result<ExtendedBinanceData> {
        // Get basic context
        let context = self.get_binance_data(symbol).await?;

        // Get 50 hourly klines for comprehensive technical analysis
        let klines_url = format!(
            "{}/api/v3/klines?symbol={}&interval=1h&limit=50",
            BINANCE_API_URL, symbol
        );
        let raw_klines: Vec<Vec<serde_json::Value>> = 
            self.http.get(&klines_url).send().await?.json().await?;

        // Convert to KlineData format for ML predictor
        let klines: Vec<KlineData> = raw_klines
            .iter()
            .map(|k| KlineData {
                timestamp: k[0].as_i64().unwrap_or(0),
                open: k[1].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
                high: k[2].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
                low: k[3].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
                close: k[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
                volume: k[5].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            })
            .collect();

        Ok(ExtendedBinanceData { context, klines })
    }

    /// Generate ML-based prediction for a market (multi-factor fusion)
    fn predict(&self, market: &Market, extended_data: &ExtendedBinanceData) -> (String, f64, f64) {
        // Get current Yes/No prices
        let yes_price = market
            .outcomes
            .iter()
            .find(|o| o.outcome.to_lowercase() == "yes")
            .map(|o| o.price.to_string().parse::<f64>().unwrap_or(0.5))
            .unwrap_or(0.5);

        // Build MarketDataInput for ML predictor
        let market_data = MarketDataInput {
            symbol: extended_data.context.symbol.clone(),
            price: extended_data.context.current_price,
            klines: extended_data.klines.clone(),
            orderbook_imbalance: None, // TODO: integrate order book data
            volume_24h: extended_data.context.volume_24h,
            sentiment_score: None, // TODO: integrate Twitter/social sentiment
            question: market.question.clone(),
        };

        // Run ML prediction with multi-factor fusion
        let ml_result = self.ml_predictor.predict(&market_data, yes_price);

        debug!(
            "ML Prediction: up_prob={:.3}, confidence={:.3}, edge={:.3}, agreement={:.3}",
            ml_result.up_probability,
            ml_result.confidence,
            ml_result.edge,
            ml_result.model_agreement
        );
        debug!(
            "Features: RSI={:.1}, MACD={:.3}, BB={:.2}, ADX={:.1}, momentum={:.3}",
            ml_result.features.rsi,
            ml_result.features.macd_signal,
            ml_result.features.bollinger_position,
            ml_result.features.adx,
            ml_result.features.momentum_1h
        );

        (ml_result.recommended_side, ml_result.up_probability, ml_result.edge)
    }

    /// Legacy simple prediction (fallback)
    #[allow(dead_code)]
    fn predict_simple(&self, market: &Market, binance: &BinanceContext) -> (String, f64, f64) {
        let yes_price = market
            .outcomes
            .iter()
            .find(|o| o.outcome.to_lowercase() == "yes")
            .map(|o| o.price.to_string().parse::<f64>().unwrap_or(0.5))
            .unwrap_or(0.5);

        let no_price = 1.0 - yes_price;
        let momentum = binance.price_change_1h_pct;
        let rsi = binance.rsi_14.unwrap_or(50.0);

        let mut up_prob: f64 = 0.5;

        if momentum > 0.5 {
            up_prob += 0.1;
        } else if momentum > 0.2 {
            up_prob += 0.05;
        } else if momentum < -0.5 {
            up_prob -= 0.1;
        } else if momentum < -0.2 {
            up_prob -= 0.05;
        }

        if rsi > 70.0 {
            up_prob -= 0.08;
        } else if rsi < 30.0 {
            up_prob += 0.08;
        }

        up_prob = up_prob.clamp(0.2, 0.8);

        let question = market.question.to_lowercase();
        let (predicted_side, fair_prob, market_price) = if question.contains("go up") {
            if up_prob > 0.5 {
                ("Yes", up_prob, yes_price)
            } else {
                ("No", 1.0 - up_prob, no_price)
            }
        } else if question.contains("go down") {
            if up_prob < 0.5 {
                ("Yes", 1.0 - up_prob, yes_price)
            } else {
                ("No", up_prob, no_price)
            }
        } else {
            if up_prob > 0.5 {
                ("Yes", up_prob, yes_price)
            } else {
                ("No", 1.0 - up_prob, no_price)
            }
        };

        let edge = fair_prob - market_price;
        (predicted_side.to_string(), fair_prob, edge)
    }

    /// Calculate position size (Kelly fraction)
    fn calculate_position_size(&self, edge: f64, win_prob: f64) -> f64 {
        if edge <= 0.0 || win_prob <= 0.0 || win_prob >= 1.0 {
            return 0.0;
        }

        // Kelly: f* = (bp - q) / b where b=1 (even money), p=win_prob, q=1-p
        // For Polymarket: f* = (edge) / (1 - market_price) simplified
        let kelly = edge / (1.0 - win_prob + 0.01); // avoid div by zero

        // Half-Kelly for safety
        let half_kelly = kelly * 0.5;

        // Cap at MAX_POSITION_PCT
        let position_pct = half_kelly.min(MAX_POSITION_PCT);
        let position = self.capital * position_pct;

        // Minimum $1, maximum 10% of remaining capital
        position.max(1.0).min(self.capital * 0.1)
    }

    /// Check hourly trade limit
    fn check_hourly_limit(&mut self) -> bool {
        let now = Utc::now();
        if now - self.last_hour_reset > Duration::hours(1) {
            self.hourly_trade_count = 0;
            self.last_hour_reset = now;
        }
        self.hourly_trade_count < MAX_TRADES_PER_HOUR
    }

    /// Execute a paper trade
    fn execute_trade(
        &mut self,
        market: &Market,
        side: &str,
        amount: f64,
        predicted_outcome: &str,
        confidence: f64,
        binance: &BinanceContext,
    ) -> Trade {
        let price = market
            .outcomes
            .iter()
            .find(|o| o.outcome == side)
            .map(|o| o.price.to_string().parse::<f64>().unwrap_or(0.5))
            .unwrap_or(0.5);

        let shares = amount / price;

        let trade = Trade {
            id: format!("T{}", Utc::now().timestamp_millis()),
            timestamp: Utc::now(),
            market_id: market.id.clone(),
            market_question: market.question.clone(),
            side: side.to_string(),
            entry_price: price,
            amount,
            shares,
            predicted_outcome: predicted_outcome.to_string(),
            prediction_confidence: confidence,
            binance_data: binance.clone(),
            status: TradeStatus::Open,
            exit_price: None,
            pnl: None,
        };

        self.capital -= amount;
        self.hourly_trade_count += 1;
        self.trades.push(trade.clone());
        self.log_trade(&trade);

        trade
    }

    /// Main trading loop
    async fn run(&mut self) -> anyhow::Result<()> {
        self.log("═══════════════════════════════════════════════════════════");
        self.log("🚀 RUST LIVE PAPER TRADING - CRYPTO HOURLY MARKETS");
        self.log(&format!("   Initial Capital: ${:.2}", INITIAL_CAPITAL));
        self.log(&format!("   Min Edge: {:.0}%", MIN_EDGE * 100.0));
        self.log(&format!("   Poll Interval: {}s", POLL_INTERVAL_SECS));
        self.log(&format!("   Max Trades/Hour: {}", MAX_TRADES_PER_HOUR));
        self.log("═══════════════════════════════════════════════════════════");

        let mut interval = interval(TokioDuration::from_secs(POLL_INTERVAL_SECS));

        loop {
            interval.tick().await;

            // Check hourly limit
            if !self.check_hourly_limit() {
                debug!("Hourly trade limit reached, waiting...");
                continue;
            }

            // Discover crypto markets
            self.log("🔍 Scanning for crypto hourly markets...");
            let markets = match self.gamma.get_crypto_markets().await {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to fetch markets: {}", e);
                    continue;
                }
            };

            self.log(&format!("   Found {} crypto markets", markets.len()));

            // Filter by liquidity and find opportunities
            let mut opportunities: Vec<(Market, String, f64, f64, ExtendedBinanceData)> = Vec::new();

            for market in &markets {
                // Skip markets not settling soon (within MAX_SETTLEMENT_MINUTES)
                if let Some(end_date) = market.end_date {
                    let now = Utc::now();
                    let time_to_settlement = end_date - now;
                    let mins_to_settlement = time_to_settlement.num_minutes();
                    
                    // Skip if already expired or too far in the future
                    if mins_to_settlement < 0 || mins_to_settlement > MAX_SETTLEMENT_MINUTES {
                        debug!(
                            "Skipping {} - settles in {} mins (max: {})",
                            market.question, mins_to_settlement, MAX_SETTLEMENT_MINUTES
                        );
                        continue;
                    }
                } else {
                    // No end_date means we can't verify settlement time, skip
                    debug!("Skipping {} - no end_date", market.question);
                    continue;
                }

                // Skip low liquidity
                let liq: f64 = market.liquidity.to_string().parse().unwrap_or(0.0);
                if liq < MIN_LIQUIDITY {
                    continue;
                }

                // Get Binance symbol
                let symbol = match self.get_binance_symbol(&market.question) {
                    Some(s) => s,
                    None => continue,
                };

                // Get extended Binance data (with klines for ML features)
                let extended_data = match self.get_extended_binance_data(symbol).await {
                    Ok(b) => b,
                    Err(e) => {
                        debug!("Binance error for {}: {}", symbol, e);
                        continue;
                    }
                };

                // Get ML prediction (multi-factor fusion)
                let (side, confidence, edge) = self.predict(&market, &extended_data);

                // Check edge
                if edge >= MIN_EDGE {
                    opportunities.push((market.clone(), side, confidence, edge, extended_data));
                }
            }

            if opportunities.is_empty() {
                self.log("   No opportunities above min edge threshold");
                continue;
            }

            // Sort by edge (highest first)
            opportunities.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

            // Take best opportunity
            let (market, side, confidence, edge, extended_data) = &opportunities[0];
            let position_size = self.calculate_position_size(*edge, *confidence);

            if position_size < 1.0 {
                self.log(&format!(
                    "   Best edge: {:.1}% but position too small, skipping",
                    edge * 100.0
                ));
                continue;
            }

            // Execute trade (use context for BinanceContext)
            let trade = self.execute_trade(
                market,
                side,
                position_size,
                side,
                *confidence,
                &extended_data.context,
            );

            self.log("═══════════════════════════════════════════════════════════");
            self.log(&format!("🎯 TRADE EXECUTED: {}", trade.id));
            self.log(&format!("   Market: {}", market.question));
            self.log(&format!("   Side: {} @ ${:.4}", side, trade.entry_price));
            self.log(&format!("   Amount: ${:.2} ({:.2} shares)", trade.amount, trade.shares));
            self.log(&format!("   Edge: {:.1}% | Confidence: {:.1}%", edge * 100.0, confidence * 100.0));
            self.log(&format!(
                "   Binance: {} @ ${:.2} (1H: {:+.2}%, RSI: {:.1})",
                extended_data.context.symbol,
                extended_data.context.current_price,
                extended_data.context.price_change_1h_pct,
                extended_data.context.rsi_14.unwrap_or(50.0)
            ));
            self.log(&format!("   Remaining Capital: ${:.2}", self.capital));
            self.log("═══════════════════════════════════════════════════════════");

            // Stats
            let total_invested: f64 = self.trades.iter().map(|t| t.amount).sum();
            let open_trades = self.trades.iter().filter(|t| matches!(t.status, TradeStatus::Open)).count();
            self.log(&format!(
                "📊 Stats: {} trades | ${:.2} invested | ${:.2} available | {} open",
                self.trades.len(),
                total_invested,
                self.capital,
                open_trades
            ));
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("polymarket_bot=debug".parse()?)
                .add_directive("live_trading=info".parse()?),
        )
        .init();

    let mut trader = LiveTrader::new().await?;
    trader.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test trader with ML predictor
    fn create_test_trader() -> LiveTrader {
        LiveTrader {
            gamma: GammaClient::new("http://test").unwrap(),
            http: Client::new(),
            capital: 100.0,
            trades: vec![],
            hourly_trade_count: 0,
            last_hour_reset: Utc::now(),
            log_file: File::create("/dev/null").unwrap(),
            ml_predictor: MLPredictor::new(MLPredictorConfig::default()),
        }
    }

    #[test]
    fn test_binance_symbol_mapping() {
        let trader = create_test_trader();

        assert_eq!(trader.get_binance_symbol("Will Bitcoin go up?"), Some("BTCUSDT"));
        assert_eq!(trader.get_binance_symbol("ETH price prediction"), Some("ETHUSDT"));
        assert_eq!(trader.get_binance_symbol("Solana hourly"), Some("SOLUSDT"));
        assert_eq!(trader.get_binance_symbol("XRP up or down"), Some("XRPUSDT"));
        assert_eq!(trader.get_binance_symbol("Random market"), None);
    }

    #[test]
    fn test_position_size_calculation() {
        let trader = create_test_trader();

        // Positive edge should give positive position
        let pos = trader.calculate_position_size(0.05, 0.6);
        assert!(pos > 0.0);
        assert!(pos <= 10.0); // Max 10% of $100

        // No edge = no position
        let pos_zero = trader.calculate_position_size(0.0, 0.5);
        assert_eq!(pos_zero, 0.0);

        // Negative edge = no position
        let pos_neg = trader.calculate_position_size(-0.05, 0.4);
        assert_eq!(pos_neg, 0.0);
    }

    #[test]
    fn test_hourly_limit() {
        let mut trader = create_test_trader();

        // Should be under limit initially
        assert!(trader.check_hourly_limit());

        // Simulate hitting limit
        trader.hourly_trade_count = MAX_TRADES_PER_HOUR;
        assert!(!trader.check_hourly_limit());

        // Simulate hour passing
        trader.last_hour_reset = Utc::now() - Duration::hours(2);
        assert!(trader.check_hourly_limit());
        assert_eq!(trader.hourly_trade_count, 0); // Should reset
    }

    #[test]
    fn test_ml_predictor_integration() {
        let trader = create_test_trader();

        // Create mock extended data
        let klines: Vec<KlineData> = (0..50)
            .map(|i| KlineData {
                timestamp: 1706572800000 + i * 3600000, // hourly intervals
                open: 85000.0 + (i as f64) * 10.0,
                high: 85100.0 + (i as f64) * 10.0,
                low: 84900.0 + (i as f64) * 10.0,
                close: 85050.0 + (i as f64) * 10.0,
                volume: 1000.0,
            })
            .collect();

        let extended_data = ExtendedBinanceData {
            context: BinanceContext {
                symbol: "BTCUSDT".to_string(),
                current_price: 85500.0,
                price_change_1h_pct: 0.5,
                volume_24h: 1_000_000.0,
                rsi_14: Some(55.0),
            },
            klines,
        };

        // Create a mock market
        use rust_decimal::Decimal;
        use polymarket_bot::types::Outcome;
        let market = Market {
            id: "test-market-id".to_string(),
            question: "Will Bitcoin go up?".to_string(),
            description: Some("Test market".to_string()),
            outcomes: vec![
                Outcome {
                    token_id: "yes-token-123".to_string(),
                    outcome: "Yes".to_string(),
                    price: Decimal::from_str_exact("0.50").unwrap(),
                },
                Outcome {
                    token_id: "no-token-456".to_string(),
                    outcome: "No".to_string(),
                    price: Decimal::from_str_exact("0.50").unwrap(),
                },
            ],
            liquidity: Decimal::from_str_exact("10000").unwrap(),
            volume: Decimal::from_str_exact("50000").unwrap(),
            end_date: Some(Utc::now() + Duration::hours(1)),
            active: true,
            closed: false,
        };

        // Run ML prediction
        let (side, prob, edge) = trader.predict(&market, &extended_data);

        // Basic sanity checks
        assert!(side == "Yes" || side == "No");
        assert!(prob >= 0.0 && prob <= 1.0);
        // Edge can be positive or negative
        println!("ML Prediction: side={}, prob={:.3}, edge={:.3}", side, prob, edge);
    }
}
