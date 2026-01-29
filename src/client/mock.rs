//! Mock API client for testing
//!
//! Provides mock implementations of CLOB and Gamma clients for:
//! - Unit tests without network calls
//! - Integration tests with controlled responses
//! - Dry run simulations

use crate::client::{OrderBook, OrderBookLevel};
use crate::error::Result;
use crate::types::{Market, Order, OrderStatus, Position, Side, Outcome};
use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{Utc, Duration};

/// Trait for CLOB operations (allows mocking)
#[async_trait]
pub trait ClobClientTrait: Send + Sync {
    async fn get_balance(&self) -> Result<Decimal>;
    async fn get_order_book(&self, token_id: &str) -> Result<OrderBook>;
    async fn place_order(&self, order: &Order) -> Result<OrderStatus>;
    async fn cancel_order(&self, order_id: &str) -> Result<()>;
    async fn get_open_orders(&self) -> Result<Vec<OrderStatus>>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
}

/// Trait for Gamma operations (allows mocking)
#[async_trait]
pub trait GammaClientTrait: Send + Sync {
    async fn get_top_markets(&self, limit: usize) -> Result<Vec<Market>>;
    async fn get_market(&self, market_id: &str) -> Result<Market>;
    async fn get_crypto_markets(&self) -> Result<Vec<Market>>;
}

/// Mock state for tracking simulated trades
#[derive(Debug, Clone)]
pub struct MockState {
    pub balance: Decimal,
    pub positions: HashMap<String, Position>,
    pub orders: Vec<MockOrder>,
    pub trades_executed: u32,
    pub total_volume: Decimal,
}

#[derive(Debug, Clone)]
pub struct MockOrder {
    pub order_id: String,
    pub token_id: String,
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
    pub status: String,
    pub timestamp: chrono::DateTime<Utc>,
}

impl Default for MockState {
    fn default() -> Self {
        Self {
            balance: dec!(1000), // Start with $1000
            positions: HashMap::new(),
            orders: Vec::new(),
            trades_executed: 0,
            total_volume: Decimal::ZERO,
        }
    }
}

/// Mock CLOB client for testing
pub struct MockClobClient {
    state: Arc<RwLock<MockState>>,
    order_books: HashMap<String, OrderBook>,
    simulate_failures: bool,
    latency_ms: u64,
}

impl MockClobClient {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MockState::default())),
            order_books: Self::default_order_books(),
            simulate_failures: false,
            latency_ms: 0,
        }
    }

    pub fn with_balance(self, balance: Decimal) -> Self {
        self.state.write().unwrap().balance = balance;
        self
    }

    pub fn with_failures(mut self) -> Self {
        self.simulate_failures = true;
        self
    }

    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn state(&self) -> Arc<RwLock<MockState>> {
        self.state.clone()
    }

    fn default_order_books() -> HashMap<String, OrderBook> {
        let mut books = HashMap::new();
        
        // Default order book with spread
        let default_book = OrderBook {
            bids: vec![
                OrderBookLevel { price: dec!(0.54), size: dec!(500) },
                OrderBookLevel { price: dec!(0.53), size: dec!(1000) },
                OrderBookLevel { price: dec!(0.52), size: dec!(2000) },
            ],
            asks: vec![
                OrderBookLevel { price: dec!(0.56), size: dec!(500) },
                OrderBookLevel { price: dec!(0.57), size: dec!(1000) },
                OrderBookLevel { price: dec!(0.58), size: dec!(2000) },
            ],
        };
        
        books.insert("default".to_string(), default_book);
        books
    }

    pub fn set_order_book(&mut self, token_id: &str, book: OrderBook) {
        self.order_books.insert(token_id.to_string(), book);
    }

    async fn simulate_latency(&self) {
        if self.latency_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        }
    }
}

impl Default for MockClobClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ClobClientTrait for MockClobClient {
    async fn get_balance(&self) -> Result<Decimal> {
        self.simulate_latency().await;
        if self.simulate_failures {
            return Err(crate::error::BotError::Api("Mock failure".into()));
        }
        Ok(self.state.read().unwrap().balance)
    }

    async fn get_order_book(&self, token_id: &str) -> Result<OrderBook> {
        self.simulate_latency().await;
        if self.simulate_failures {
            return Err(crate::error::BotError::Api("Mock failure".into()));
        }
        Ok(self.order_books
            .get(token_id)
            .or(self.order_books.get("default"))
            .cloned()
            .unwrap_or_else(|| OrderBook {
                bids: vec![OrderBookLevel { price: dec!(0.50), size: dec!(1000) }],
                asks: vec![OrderBookLevel { price: dec!(0.52), size: dec!(1000) }],
            }))
    }

    async fn place_order(&self, order: &Order) -> Result<OrderStatus> {
        self.simulate_latency().await;
        if self.simulate_failures {
            return Err(crate::error::BotError::Api("Mock failure".into()));
        }

        let mut state = self.state.write().unwrap();
        let order_id = format!("mock_order_{}", state.trades_executed + 1);
        
        // Update balance (simulate immediate fill)
        let cost = order.price * order.size;
        match order.side {
            Side::Buy => {
                if state.balance < cost {
                    return Err(crate::error::BotError::Execution("Insufficient balance".into()));
                }
                state.balance -= cost;
            }
            Side::Sell => {
                state.balance += cost;
            }
        }

        // Record the order
        state.orders.push(MockOrder {
            order_id: order_id.clone(),
            token_id: order.token_id.clone(),
            side: order.side,
            price: order.price,
            size: order.size,
            status: "FILLED".to_string(),
            timestamp: Utc::now(),
        });
        
        state.trades_executed += 1;
        state.total_volume += cost;

        Ok(OrderStatus {
            order_id,
            status: "FILLED".to_string(),
            filled_size: order.size,
            remaining_size: Decimal::ZERO,
            avg_price: Some(order.price),
        })
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        self.simulate_latency().await;
        let mut state = self.state.write().unwrap();
        if let Some(order) = state.orders.iter_mut().find(|o| o.order_id == order_id) {
            order.status = "CANCELLED".to_string();
        }
        Ok(())
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderStatus>> {
        self.simulate_latency().await;
        let state = self.state.read().unwrap();
        Ok(state.orders
            .iter()
            .filter(|o| o.status == "OPEN" || o.status == "PARTIAL")
            .map(|o| OrderStatus {
                order_id: o.order_id.clone(),
                status: o.status.clone(),
                filled_size: Decimal::ZERO,
                remaining_size: o.size,
                avg_price: None,
            })
            .collect())
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        self.simulate_latency().await;
        let state = self.state.read().unwrap();
        Ok(state.positions.values().cloned().collect())
    }
}

/// Mock Gamma client for market data
pub struct MockGammaClient {
    markets: Vec<Market>,
    simulate_failures: bool,
}

impl MockGammaClient {
    pub fn new() -> Self {
        Self {
            markets: Self::default_markets(),
            simulate_failures: false,
        }
    }

    pub fn with_markets(mut self, markets: Vec<Market>) -> Self {
        self.markets = markets;
        self
    }

    pub fn with_failures(mut self) -> Self {
        self.simulate_failures = true;
        self
    }

    fn default_markets() -> Vec<Market> {
        vec![
            Market {
                id: "btc_100k_2026".to_string(),
                question: "Will Bitcoin reach $100,000 in 2026?".to_string(),
                description: Some("Bitcoin price prediction market".to_string()),
                end_date: Some(Utc::now() + Duration::days(365)),
                volume: dec!(500000),
                liquidity: dec!(100000),
                outcomes: vec![
                    Outcome {
                        token_id: "btc_yes".to_string(),
                        outcome: "Yes".to_string(),
                        price: dec!(0.65),
                    },
                    Outcome {
                        token_id: "btc_no".to_string(),
                        outcome: "No".to_string(),
                        price: dec!(0.35),
                    },
                ],
                active: true,
                closed: false,
            },
            Market {
                id: "eth_5k_2026".to_string(),
                question: "Will Ethereum reach $5,000 in 2026?".to_string(),
                description: Some("Ethereum price prediction".to_string()),
                end_date: Some(Utc::now() + Duration::days(365)),
                volume: dec!(300000),
                liquidity: dec!(80000),
                outcomes: vec![
                    Outcome {
                        token_id: "eth_yes".to_string(),
                        outcome: "Yes".to_string(),
                        price: dec!(0.55),
                    },
                    Outcome {
                        token_id: "eth_no".to_string(),
                        outcome: "No".to_string(),
                        price: dec!(0.45),
                    },
                ],
                active: true,
                closed: false,
            },
            Market {
                id: "btc_up_24h".to_string(),
                question: "Will Bitcoin go up in the next 24 hours?".to_string(),
                description: Some("Crypto HF market".to_string()),
                end_date: Some(Utc::now() + Duration::hours(24)),
                volume: dec!(50000),
                liquidity: dec!(20000),
                outcomes: vec![
                    Outcome {
                        token_id: "btc24_yes".to_string(),
                        outcome: "Yes".to_string(),
                        price: dec!(0.52),
                    },
                    Outcome {
                        token_id: "btc24_no".to_string(),
                        outcome: "No".to_string(),
                        price: dec!(0.48),
                    },
                ],
                active: true,
                closed: false,
            },
        ]
    }
}

impl Default for MockGammaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GammaClientTrait for MockGammaClient {
    async fn get_top_markets(&self, limit: usize) -> Result<Vec<Market>> {
        if self.simulate_failures {
            return Err(crate::error::BotError::Api("Mock failure".into()));
        }
        Ok(self.markets.iter().take(limit).cloned().collect())
    }

    async fn get_market(&self, market_id: &str) -> Result<Market> {
        if self.simulate_failures {
            return Err(crate::error::BotError::Api("Mock failure".into()));
        }
        self.markets
            .iter()
            .find(|m| m.id == market_id)
            .cloned()
            .ok_or_else(|| crate::error::BotError::Api(format!("Market {} not found", market_id)))
    }

    async fn get_crypto_markets(&self) -> Result<Vec<Market>> {
        if self.simulate_failures {
            return Err(crate::error::BotError::Api("Mock failure".into()));
        }
        Ok(self.markets
            .iter()
            .filter(|m| m.question.to_lowercase().contains("bitcoin") 
                     || m.question.to_lowercase().contains("ethereum")
                     || m.question.to_lowercase().contains("btc")
                     || m.question.to_lowercase().contains("eth"))
            .cloned()
            .collect())
    }
}

/// Builder for creating test scenarios
pub struct MockScenarioBuilder {
    clob: MockClobClient,
    gamma: MockGammaClient,
}

impl MockScenarioBuilder {
    pub fn new() -> Self {
        Self {
            clob: MockClobClient::new(),
            gamma: MockGammaClient::new(),
        }
    }

    pub fn with_balance(mut self, balance: Decimal) -> Self {
        self.clob = self.clob.with_balance(balance);
        self
    }

    pub fn with_markets(mut self, markets: Vec<Market>) -> Self {
        self.gamma = self.gamma.with_markets(markets);
        self
    }

    pub fn with_order_book(mut self, token_id: &str, book: OrderBook) -> Self {
        self.clob.set_order_book(token_id, book);
        self
    }

    pub fn simulate_network_issues(mut self) -> Self {
        self.clob = self.clob.with_failures();
        self.gamma = self.gamma.with_failures();
        self
    }

    pub fn with_latency(mut self, ms: u64) -> Self {
        self.clob = self.clob.with_latency(ms);
        self
    }

    pub fn build(self) -> (MockClobClient, MockGammaClient) {
        (self.clob, self.gamma)
    }
}

impl Default for MockScenarioBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_clob_balance() {
        let client = MockClobClient::new().with_balance(dec!(5000));
        assert_eq!(client.get_balance().await.unwrap(), dec!(5000));
    }

    #[tokio::test]
    async fn test_mock_clob_place_order() {
        let client = MockClobClient::new().with_balance(dec!(1000));
        
        let order = Order {
            token_id: "test".to_string(),
            side: Side::Buy,
            price: dec!(0.55),
            size: dec!(100),
            order_type: crate::types::OrderType::GTC,
        };
        
        let result = client.place_order(&order).await.unwrap();
        assert_eq!(result.status, "FILLED");
        
        // Check balance updated
        let new_balance = client.get_balance().await.unwrap();
        assert_eq!(new_balance, dec!(945)); // 1000 - (0.55 * 100)
    }

    #[tokio::test]
    async fn test_mock_clob_insufficient_balance() {
        let client = MockClobClient::new().with_balance(dec!(10));
        
        let order = Order {
            token_id: "test".to_string(),
            side: Side::Buy,
            price: dec!(0.55),
            size: dec!(100), // Would cost $55
            order_type: crate::types::OrderType::GTC,
        };
        
        assert!(client.place_order(&order).await.is_err());
    }

    #[tokio::test]
    async fn test_mock_gamma_markets() {
        let client = MockGammaClient::new();
        let markets = client.get_top_markets(10).await.unwrap();
        assert!(!markets.is_empty());
        assert!(markets.iter().any(|m| m.question.contains("Bitcoin")));
    }

    #[tokio::test]
    async fn test_mock_failure_simulation() {
        let client = MockClobClient::new().with_failures();
        assert!(client.get_balance().await.is_err());
    }

    #[tokio::test]
    async fn test_scenario_builder() {
        let (clob, gamma) = MockScenarioBuilder::new()
            .with_balance(dec!(2000))
            .build();
        
        assert_eq!(clob.get_balance().await.unwrap(), dec!(2000));
        assert!(!gamma.get_top_markets(5).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_trade_tracking() {
        let client = MockClobClient::new().with_balance(dec!(1000));
        
        // Execute multiple trades
        for i in 0..3 {
            let order = Order {
                token_id: format!("token_{}", i),
                side: Side::Buy,
                price: dec!(0.10),
                size: dec!(10),
                order_type: crate::types::OrderType::GTC,
            };
            client.place_order(&order).await.unwrap();
        }
        
        let state = client.state();
        let state = state.read().unwrap();
        assert_eq!(state.trades_executed, 3);
        assert_eq!(state.total_volume, dec!(3)); // 3 * (0.10 * 10)
    }
}
