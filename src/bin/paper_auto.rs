//! Multi-Asset Crypto 15m Paper Trading with Binance WebSocket Feed
//! Real-time price streaming for BTC, ETH, SOL, XRP

use polymarket_bot::client::GammaClient;
use polymarket_bot::paper::{PaperTrader, PaperTraderConfig, PositionSide, LlmTrader, PositionContext, MarketContext, TradeDecision};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tracing::{info, warn, error};
use futures_util::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use serde::Deserialize;

/// Asset configuration
#[derive(Clone)]
struct Asset {
    name: &'static str,
    binance: &'static str,      // Binance symbol (lowercase)
    poly_slug: &'static str,    // Polymarket slug prefix
}

const ASSETS: &[Asset] = &[
    Asset { name: "BTC", binance: "btcusdt", poly_slug: "btc-updown-15m" },
    Asset { name: "ETH", binance: "ethusdt", poly_slug: "eth-updown-15m" },
    Asset { name: "SOL", binance: "solusdt", poly_slug: "sol-updown-15m" },
    Asset { name: "XRP", binance: "xrpusdt", poly_slug: "xrp-updown-15m" },
];

/// Shared price state from WebSocket
#[derive(Default)]
struct PriceState {
    prices: HashMap<String, f64>,
    trends: HashMap<String, (f64, Vec<f64>)>, // (trend_pct, recent_prices)
}

#[derive(Deserialize)]
struct BinanceTicker {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "c")]
    price: String,
    #[serde(rename = "P")]
    change_pct: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let config = PaperTraderConfig {
        initial_balance: dec!(1000),
        max_position_pct: dec!(15),
        slippage_pct: dec!(0.5),
        fee_pct: dec!(0.1),
        save_interval: 30,
        state_file: Some("multi_crypto_paper.json".to_string()),
    };
    
    let gamma = GammaClient::new("https://gamma-api.polymarket.com")?;
    let trader = Arc::new(PaperTrader::new(config, gamma.clone()));
    let state = Arc::new(RwLock::new(PriceState::default()));
    
    // LLM Trader for intelligent decisions (optional)
    let llm_trader = std::env::var("DEEPSEEK_API_KEY").ok().map(|key| {
        info!("🤖 LLM trading decisions enabled");
        Arc::new(LlmTrader::new(key))
    });
    
    info!("🚀 Multi-Asset 15m Trading with WebSocket Feed");
    info!("💰 Initial balance: $1000");
    
    // Spawn WebSocket price feed
    let state_ws = state.clone();
    tokio::spawn(async move {
        binance_websocket(state_ws).await;
    });
    
    // Main trading loop - check every 2 seconds (WebSocket updates prices continuously)
    let mut last_slot: u64 = 0;
    let mut last_llm_check: u64 = 0;
    
    loop {
        let prices = {
            let s = state.read().await;
            s.prices.clone()
        };
        
        if !prices.is_empty() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            // LLM decision every 30 seconds
            if let Some(ref llm) = llm_trader {
                if now - last_llm_check >= 30 {
                    last_llm_check = now;
                    llm_decide(&trader, llm, &state, &prices).await;
                }
            }
            
            if let Err(e) = trade_loop(&trader, &state, &prices, &mut last_slot).await {
                error!("Trade loop error: {}", e);
            }
        }
        
        sleep(Duration::from_secs(2)).await;
    }
}

/// LLM-powered trading decision
async fn llm_decide(
    trader: &Arc<PaperTrader>,
    llm: &Arc<LlmTrader>,
    state: &Arc<RwLock<PriceState>>,
    prices: &HashMap<String, f64>,
) {
    let positions = trader.get_positions().await;
    let open_positions: Vec<_> = positions.iter().filter(|p| p.is_open()).collect();
    
    if open_positions.is_empty() {
        return; // No positions to evaluate
    }
    
    // Build position context
    let pos_ctx: Vec<PositionContext> = open_positions.iter().map(|p| {
        let entry_price = (p.cost_basis / p.shares).to_string().parse::<f64>().unwrap_or(0.5);
        let current_price = p.current_price.to_string().parse::<f64>().unwrap_or(0.5);
        let pnl_pct = if entry_price > 0.0 {
            (current_price - entry_price) / entry_price * 100.0
        } else {
            0.0
        };
        
        PositionContext {
            asset: p.market_id.split('-').next().unwrap_or("").to_uppercase(),
            side: if p.market_id.contains("-up") { "UP".to_string() } else { "DOWN".to_string() },
            entry_price,
            current_price,
            unrealized_pnl_pct: pnl_pct,
            shares: p.shares.to_string().parse().unwrap_or(0.0),
            cost_basis: p.cost_basis.to_string().parse().unwrap_or(0.0),
            current_value: p.current_value.to_string().parse().unwrap_or(0.0),
        }
    }).collect();
    
    // Build market context
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let time_in_slot = now % 900;
    
    let mkt_ctx: Vec<MarketContext> = {
        let s = state.read().await;
        let default_trend = (0.0, Vec::new());
        ASSETS.iter().filter_map(|a| {
            let sym = a.binance.to_uppercase();
            let price = prices.get(&sym)?;
            let (change, _) = s.trends.get(&sym).unwrap_or(&default_trend);
            let change_val = *change;
            Some(MarketContext {
                asset: a.name.to_string(),
                binance_price: *price,
                binance_24h_change: change_val,
                poly_up_price: 0.5, // Would need to fetch
                poly_down_price: 0.5,
                window_seconds_remaining: 900 - time_in_slot,
                trend_direction: if change_val > 0.05 { "UP".to_string() } 
                    else if change_val < -0.05 { "DOWN".to_string() } 
                    else { "FLAT".to_string() },
            })
        }).collect()
    };
    
    let balance = trader.get_balance().await.to_string().parse().unwrap_or(1000.0);
    
    info!("🤖 Consulting LLM for {} positions...", pos_ctx.len());
    
    match llm.decide(&pos_ctx, &mkt_ctx, balance).await {
        Ok(decisions) => {
            for decision in decisions {
                match decision {
                    TradeDecision::Sell { position_id, reason } => {
                        // Find matching position
                        for pos in &positions {
                            if pos.is_open() && pos.market_id.to_lowercase().contains(&position_id.to_lowercase()) {
                                info!("🤖 LLM决策: 卖出 {} - {}", position_id, reason);
                                if let Err(e) = trader.sell(&pos.id, format!("LLM: {}", reason)).await {
                                    warn!("卖出失败: {}", e);
                                }
                                break;
                            }
                        }
                    }
                    TradeDecision::SellAll { reason } => {
                        info!("🤖 LLM决策: 全部卖出 - {}", reason);
                        for pos in &positions {
                            if pos.is_open() {
                                let _ = trader.sell(&pos.id, format!("LLM: {}", reason)).await;
                            }
                        }
                    }
                    TradeDecision::Hold => {
                        info!("🤖 LLM决策: 持有");
                    }
                    _ => {}
                }
            }
        }
        Err(e) => {
            warn!("LLM决策失败: {}", e);
        }
    }
}

async fn binance_websocket(state: Arc<RwLock<PriceState>>) {
    // Combined stream for all symbols
    let streams: Vec<String> = ASSETS.iter()
        .map(|a| format!("{}@ticker", a.binance))
        .collect();
    let url = format!("wss://stream.binance.com:9443/stream?streams={}", streams.join("/"));
    
    loop {
        info!("📡 Connecting to Binance WebSocket...");
        
        match connect_async(&url).await {
            Ok((ws, _)) => {
                info!("✅ Binance WebSocket connected");
                let (_, mut read) = ws.split();
                
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            // Parse combined stream format: {"stream":"btcusdt@ticker","data":{...}}
                            if let Ok(wrapper) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some(data) = wrapper.get("data") {
                                    if let Ok(ticker) = serde_json::from_value::<BinanceTicker>(data.clone()) {
                                        let price: f64 = ticker.price.parse().unwrap_or(0.0);
                                        let change: f64 = ticker.change_pct.parse().unwrap_or(0.0);
                                        
                                        let mut s = state.write().await;
                                        s.prices.insert(ticker.symbol.clone(), price);
                                        
                                        // Update trend tracking
                                        let entry = s.trends.entry(ticker.symbol.clone())
                                            .or_insert((0.0, Vec::with_capacity(30)));
                                        entry.0 = change;
                                        entry.1.push(price);
                                        if entry.1.len() > 30 {
                                            entry.1.remove(0);
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Message::Close(_)) => {
                            warn!("WebSocket closed, reconnecting...");
                            break;
                        }
                        Err(e) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                error!("WebSocket connect failed: {}", e);
            }
        }
        
        sleep(Duration::from_secs(3)).await;
    }
}

async fn trade_loop(
    trader: &Arc<PaperTrader>,
    state: &Arc<RwLock<PriceState>>,
    prices: &HashMap<String, f64>,
    last_slot: &mut u64,
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let current_slot = now - (now % 900);
    let time_in_slot = now % 900;
    
    // New window detection + settle previous window
    if current_slot != *last_slot {
        let prev_slot = *last_slot;
        *last_slot = current_slot;
        
        // Settle previous window positions if any
        if prev_slot > 0 {
            settle_previous_window(trader, &client, prev_slot).await;
        }
        
        let price_str: String = ASSETS.iter()
            .filter_map(|a| {
                let sym = a.binance.to_uppercase();
                prices.get(&sym).map(|p| format!("{}: ${:.0}", a.name, p))
            })
            .collect::<Vec<_>>()
            .join(" | ");
        info!("🔄 New 15m window - {}", price_str);
    }
    
    let mut status_parts = Vec::new();
    
    for asset in ASSETS {
        let symbol = asset.binance.to_uppercase();
        let price = match prices.get(&symbol) {
            Some(p) => *p,
            None => continue,
        };
        
        // Get trend from state
        let (trend_pct, trend_dir) = {
            let s = state.read().await;
            if let Some((change, history)) = s.trends.get(&symbol) {
                let dir = if *change > 0.05 { "📈" }
                    else if *change < -0.05 { "📉" }
                    else { "➡️" };
                (*change, dir)
            } else {
                (0.0, "⏳")
            }
        };
        
        // Get Polymarket prices
        let market_slug = format!("{}-{}", asset.poly_slug, current_slot);
        let url = format!("https://gamma-api.polymarket.com/events?slug={}", market_slug);
        
        let resp: Vec<serde_json::Value> = match client.get(&url).send().await {
            Ok(r) => r.json().await.unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        
        if let Some(event) = resp.first() {
            if let Some(market) = event["markets"].as_array().and_then(|m| m.first()) {
                let condition_id = market["conditionId"].as_str().unwrap_or("");
                let question = market["question"].as_str().unwrap_or("");
                
                if let Some(prices_str) = market["outcomePrices"].as_str() {
                    let p: Vec<&str> = prices_str.trim_matches(|c| c == '[' || c == ']' || c == '"')
                        .split("\", \"").collect();
                    
                    if p.len() >= 2 {
                        let up_price: f64 = p[0].parse().unwrap_or(0.5);
                        let down_price: f64 = p[1].parse().unwrap_or(0.5);
                        
                        status_parts.push(format!("{}{} ${:.0} U:{:.0}% D:{:.0}%",
                            trend_dir, asset.name, price, up_price * 100.0, down_price * 100.0));
                        
                        // Trading logic
                        let positions = trader.get_positions().await;
                        let has_position = positions.iter().any(|p|
                            p.market_id.contains(&asset.name.to_lowercase()) &&
                            p.market_id.contains(&current_slot.to_string())
                        );
                        
                        if time_in_slot < 780 && !has_position {
                            trade_asset(trader, asset, condition_id, question,
                                trend_pct, up_price, down_price, current_slot).await;
                        }
                        
                        check_stop_loss(trader, asset, trend_pct, current_slot).await;
                    }
                }
            }
        } else {
            status_parts.push(format!("{}{} ${:.0} (no market)", trend_dir, asset.name, price));
        }
    }
    
    // Compact status every 2 seconds
    info!("📊 {} | {}s", status_parts.join(" | "), time_in_slot);
    
    let summary = trader.get_summary().await;
    info!("💰 ${:.2} | P&L: ${:.2} ({:.2}%) | Trades: {}",
        summary.total_value,
        summary.total_pnl,
        summary.roi_percent,
        summary.trade_count
    );
    
    Ok(())
}

/// Settle positions from previous window by fetching actual results
async fn settle_previous_window(
    trader: &Arc<PaperTrader>,
    client: &reqwest::Client,
    prev_slot: u64,
) {
    info!("📊 Settling positions from slot {}", prev_slot);
    
    for asset in ASSETS {
        let market_slug = format!("{}-{}", asset.poly_slug, prev_slot);
        let url = format!("https://gamma-api.polymarket.com/events?slug={}", market_slug);
        
        let resp: Vec<serde_json::Value> = match client.get(&url).send().await {
            Ok(r) => r.json().await.unwrap_or_default(),
            Err(_) => continue,
        };
        
        if let Some(event) = resp.first() {
            // Check if market is closed/resolved
            let closed = event["closed"].as_bool().unwrap_or(false);
            
            if closed {
                if let Some(market) = event["markets"].as_array().and_then(|m| m.first()) {
                    // Get winning outcome - check outcomePrices (winner = 1.0)
                    if let Some(prices_str) = market["outcomePrices"].as_str() {
                        let p: Vec<&str> = prices_str.trim_matches(|c| c == '[' || c == ']' || c == '"')
                            .split("\", \"").collect();
                        
                        if p.len() >= 2 {
                            let up_price: f64 = p[0].parse().unwrap_or(0.5);
                            let up_won = up_price > 0.9; // If UP price is ~1.0, UP won
                            
                            // Settle this asset's positions
                            let market_id_pattern = format!("{}-15m-{}", asset.name.to_lowercase(), prev_slot);
                            if let Ok(trades) = trader.settle_market(&market_id_pattern, up_won).await {
                                for trade in trades {
                                    let pnl = trade.pnl.unwrap_or(dec!(0));
                                    info!("📋 {} settled: {} ${:.2}", asset.name, trade.reason, pnl);
                                }
                            }
                        }
                    }
                }
            } else {
                info!("⏳ {} market not yet closed", asset.name);
            }
        }
    }
}

async fn trade_asset(
    trader: &PaperTrader,
    asset: &Asset,
    condition_id: &str,
    question: &str,
    trend_pct: f64,
    up_price: f64,
    down_price: f64,
    current_slot: u64,
) {
    // Follow trend + look for mispricing
    if trend_pct > 0.08 && up_price < 0.55 {
        let amount = if trend_pct > 0.15 { dec!(80) } else { dec!(50) };
        buy_position(trader, asset, "UP", condition_id, question, up_price, amount, current_slot).await;
    }
    else if trend_pct < -0.08 && down_price < 0.55 {
        let amount = if trend_pct < -0.15 { dec!(80) } else { dec!(50) };
        buy_position(trader, asset, "DOWN", condition_id, question, down_price, amount, current_slot).await;
    }
    // Extreme mispricing
    else if up_price < 0.12 && trend_pct > -0.10 {
        buy_position(trader, asset, "UP", condition_id, question, up_price, dec!(60), current_slot).await;
    }
    else if down_price < 0.12 && trend_pct < 0.10 {
        buy_position(trader, asset, "DOWN", condition_id, question, down_price, dec!(60), current_slot).await;
    }
}

async fn check_stop_loss(trader: &PaperTrader, asset: &Asset, trend_pct: f64, current_slot: u64) {
    let positions = trader.get_positions().await;
    let slot_str = current_slot.to_string();
    let asset_lower = asset.name.to_lowercase();
    
    for pos in &positions {
        if pos.market_id.contains(&asset_lower) && pos.market_id.contains(&slot_str) {
            let is_up = pos.market_id.contains("-up");
            let is_down = pos.market_id.contains("-down");
            
            let should_stop = if is_up && trend_pct < -0.12 {
                info!("⚠️ {} STOP: UP but crashing {:.2}%", asset.name, trend_pct);
                true
            } else if is_down && trend_pct > 0.12 {
                info!("⚠️ {} STOP: DOWN but mooning {:.2}%", asset.name, trend_pct);
                true
            } else { false };
            
            if should_stop {
                if let Ok(record) = trader.sell(&pos.id, format!("{} stop-loss", asset.name)).await {
                    let pnl = record.pnl.unwrap_or(dec!(0));
                    info!("🛑 {} CLOSED P&L: ${:.2}", asset.name, pnl);
                }
            }
        }
    }
}

async fn buy_position(
    trader: &PaperTrader,
    asset: &Asset,
    side: &str,
    condition_id: &str,
    question: &str,
    price: f64,
    amount: Decimal,
    current_slot: u64,
) {
    let market_id = format!("{}-15m-{}-{}", asset.name.to_lowercase(), current_slot, side.to_lowercase());
    
    let mock_market = polymarket_bot::types::Market {
        id: market_id,
        question: question.to_string(),
        description: None,
        outcomes: vec![
            polymarket_bot::types::Outcome {
                token_id: condition_id.to_string(),
                outcome: "Yes".to_string(),
                price: Decimal::from_f64_retain(price).unwrap_or(dec!(0.1)),
            },
        ],
        volume: dec!(0),
        liquidity: dec!(0),
        end_date: None,
        active: true,
        closed: false,
    };
    
    match trader.buy(&mock_market, PositionSide::Yes, amount,
        format!("{} {} @ {:.1}%", asset.name, side, price * 100.0)).await
    {
        Ok(_) => info!("🎰 {} BUY {} @ {:.1}% - ${}", asset.name, side, price * 100.0, amount),
        Err(e) => warn!("{} buy failed: {}", asset.name, e),
    }
}
