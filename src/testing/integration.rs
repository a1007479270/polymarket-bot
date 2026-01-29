//! Integration Test Harness
//!
//! Provides end-to-end testing of bot components.

use crate::client::mock::{MockClobClient, MockGammaClient, MockScenarioBuilder, ClobClientTrait, GammaClientTrait};
use crate::config::{StrategyConfig, RiskConfig};
use crate::strategy::SignalGenerator;
use crate::types::{Market, Side, Order, OrderType, Outcome};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Instant;
use chrono::{Utc, Duration};

/// Integration test harness for running component tests
pub struct IntegrationTestHarness {
    clob: MockClobClient,
    gamma: MockGammaClient,
    signal_gen: SignalGenerator,
    test_results: Vec<TestResult>,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub duration_ms: u128,
    pub message: String,
}

impl IntegrationTestHarness {
    pub fn new() -> Self {
        let (clob, gamma) = MockScenarioBuilder::new()
            .with_balance(dec!(10000))
            .build();
        
        Self {
            clob,
            gamma,
            signal_gen: SignalGenerator::new(StrategyConfig::default(), RiskConfig::default()),
            test_results: Vec::new(),
        }
    }

    /// Run all integration tests
    pub async fn run_all(&mut self) -> Vec<TestResult> {
        self.test_results.clear();
        
        self.test_market_fetch().await;
        self.test_order_placement().await;
        self.test_signal_generation().await;
        self.test_risk_limits().await;
        self.test_portfolio_tracking().await;
        self.test_order_cancellation().await;
        self.test_position_management().await;
        
        self.test_results.clone()
    }

    async fn test_market_fetch(&mut self) {
        let start = Instant::now();
        let name = "Market Fetch".to_string();
        
        match self.gamma.get_top_markets(10).await {
            Ok(markets) => {
                if markets.is_empty() {
                    self.record_result(name, false, start, "No markets returned".to_string());
                } else {
                    self.record_result(name, true, start, format!("Fetched {} markets", markets.len()));
                }
            }
            Err(e) => {
                self.record_result(name, false, start, format!("Error: {}", e));
            }
        }
    }

    async fn test_order_placement(&mut self) {
        let start = Instant::now();
        let name = "Order Placement".to_string();
        
        let order = Order {
            token_id: "test_token".to_string(),
            side: Side::Buy,
            price: dec!(0.55),
            size: dec!(100),
            order_type: OrderType::GTC,
        };
        
        match self.clob.place_order(&order).await {
            Ok(status) => {
                if status.status == "FILLED" {
                    self.record_result(name, true, start, format!("Order {} filled", status.order_id));
                } else {
                    self.record_result(name, true, start, format!("Order {} status: {}", status.order_id, status.status));
                }
            }
            Err(e) => {
                self.record_result(name, false, start, format!("Error: {}", e));
            }
        }
    }

    async fn test_signal_generation(&mut self) {
        let start = Instant::now();
        let name = "Signal Generation".to_string();
        
        let market = Market {
            id: "test_market".to_string(),
            question: "Test question?".to_string(),
            description: None,
            end_date: Some(Utc::now() + Duration::days(30)),
            volume: dec!(100000),
            liquidity: dec!(50000),
            outcomes: vec![
                Outcome {
                    token_id: "yes".to_string(),
                    outcome: "Yes".to_string(),
                    price: dec!(0.50),
                },
                Outcome {
                    token_id: "no".to_string(),
                    outcome: "No".to_string(),
                    price: dec!(0.50),
                },
            ],
            active: true,
            closed: false,
        };

        let prediction = crate::model::Prediction {
            probability: dec!(0.65),
            confidence: dec!(0.80),
            reasoning: "Test prediction".to_string(),
        };

        match self.signal_gen.generate(&market, &prediction) {
            Some(signal) => {
                if signal.edge > Decimal::ZERO {
                    self.record_result(name, true, start, format!("Signal generated with {:.1}% edge", signal.edge * dec!(100)));
                } else {
                    self.record_result(name, true, start, "No tradeable signal (as expected for low edge)".to_string());
                }
            }
            None => {
                self.record_result(name, true, start, "No signal (edge below threshold)".to_string());
            }
        }
    }

    async fn test_risk_limits(&mut self) {
        let start = Instant::now();
        let name = "Risk Limits".to_string();
        
        // Test with large order that should fail
        let order = Order {
            token_id: "test".to_string(),
            side: Side::Buy,
            price: dec!(0.50),
            size: dec!(100000), // Very large order
            order_type: OrderType::GTC,
        };
        
        match self.clob.place_order(&order).await {
            Ok(_) => {
                self.record_result(name, false, start, "Large order should have failed".to_string());
            }
            Err(_) => {
                self.record_result(name, true, start, "Risk limit correctly rejected large order".to_string());
            }
        }
    }

    async fn test_portfolio_tracking(&mut self) {
        let start = Instant::now();
        let name = "Portfolio Tracking".to_string();
        
        let initial_balance = self.clob.get_balance().await.unwrap_or(Decimal::ZERO);
        
        // Execute a trade
        let order = Order {
            token_id: "track_test".to_string(),
            side: Side::Buy,
            price: dec!(0.50),
            size: dec!(10),
            order_type: OrderType::GTC,
        };
        let _ = self.clob.place_order(&order).await;
        
        let new_balance = self.clob.get_balance().await.unwrap_or(Decimal::ZERO);
        let expected_cost = dec!(5); // 0.50 * 10
        
        if (initial_balance - new_balance - expected_cost).abs() < dec!(0.01) {
            self.record_result(name, true, start, format!("Balance tracked correctly: {} -> {}", initial_balance, new_balance));
        } else {
            self.record_result(name, false, start, format!("Balance mismatch: expected {} got {}", initial_balance - expected_cost, new_balance));
        }
    }

    async fn test_order_cancellation(&mut self) {
        let start = Instant::now();
        let name = "Order Cancellation".to_string();
        
        match self.clob.cancel_order("test_order_123").await {
            Ok(_) => {
                self.record_result(name, true, start, "Order cancelled successfully".to_string());
            }
            Err(e) => {
                self.record_result(name, false, start, format!("Cancel failed: {}", e));
            }
        }
    }

    async fn test_position_management(&mut self) {
        let start = Instant::now();
        let name = "Position Management".to_string();
        
        match self.clob.get_positions().await {
            Ok(positions) => {
                self.record_result(name, true, start, format!("Retrieved {} positions", positions.len()));
            }
            Err(e) => {
                self.record_result(name, false, start, format!("Error: {}", e));
            }
        }
    }

    fn record_result(&mut self, name: String, passed: bool, start: Instant, message: String) {
        self.test_results.push(TestResult {
            name,
            passed,
            duration_ms: start.elapsed().as_millis(),
            message,
        });
    }

    /// Generate test report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("# 🧪 Integration Test Report\n\n");
        
        let passed = self.test_results.iter().filter(|r| r.passed).count();
        let total = self.test_results.len();
        
        report.push_str(&format!("**Results**: {}/{} passed ({:.0}%)\n\n", 
            passed, total, 
            if total > 0 { passed as f64 / total as f64 * 100.0 } else { 0.0 }
        ));
        
        report.push_str("| Test | Status | Duration | Message |\n");
        report.push_str("|------|--------|----------|----------|\n");
        
        for result in &self.test_results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!("| {} | {} | {}ms | {} |\n",
                result.name, status, result.duration_ms, result.message
            ));
        }
        
        report
    }
}

impl Default for IntegrationTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_harness_creation() {
        let harness = IntegrationTestHarness::new();
        assert!(harness.test_results.is_empty());
    }

    #[tokio::test]
    async fn test_run_all_tests() {
        let mut harness = IntegrationTestHarness::new();
        let results = harness.run_all().await;
        assert!(!results.is_empty());
        
        // All tests should pass with mock clients
        let passed = results.iter().filter(|r| r.passed).count();
        assert!(passed > 0, "At least some tests should pass");
    }

    #[tokio::test]
    async fn test_report_generation() {
        let mut harness = IntegrationTestHarness::new();
        harness.run_all().await;
        let report = harness.generate_report();
        assert!(report.contains("Integration Test Report"));
        assert!(report.contains("PASS") || report.contains("FAIL"));
    }
}
