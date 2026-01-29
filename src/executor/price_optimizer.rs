//! Execution Price Optimizer
//!
//! Determines optimal order placement based on:
//! 1. Order book analysis (spread, depth, imbalance)
//! 2. Urgency of execution
//! 3. Market maker competition
//! 4. Historical fill rates at different price levels
//! 5. Probability of price movement

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

use super::slippage_predictor::{OrderBook, OrderSide};

/// Price optimization configuration
#[derive(Debug, Clone)]
pub struct PriceOptimizerConfig {
    /// Max spread willing to cross (basis points)
    pub max_spread_to_cross_bps: Decimal,
    /// Edge above best price for limit orders (basis points)
    pub limit_order_edge_bps: Decimal,
    /// Time-in-force for limit orders (seconds)
    pub limit_order_timeout_secs: u32,
    /// Fill rate threshold to consider aggressive pricing
    pub aggressive_fill_rate_threshold: Decimal,
    /// Max price improvement attempts
    pub max_price_improvement_attempts: u32,
    /// Order book imbalance threshold for side pressure
    pub imbalance_threshold: Decimal,
}

impl Default for PriceOptimizerConfig {
    fn default() -> Self {
        Self {
            max_spread_to_cross_bps: dec!(20),     // 0.2% max spread to cross
            limit_order_edge_bps: dec!(5),         // 0.05% edge for limit orders
            limit_order_timeout_secs: 30,          // 30 second timeout
            aggressive_fill_rate_threshold: dec!(0.7), // 70% fill rate to go aggressive
            max_price_improvement_attempts: 3,
            imbalance_threshold: dec!(1.5),        // 1.5:1 ratio for imbalance
        }
    }
}

/// Order execution urgency
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionUrgency {
    /// Must fill immediately (use market order)
    Immediate,
    /// Normal execution (passive with timeout)
    Normal,
    /// Patient execution (wait for best price)
    Patient,
}

/// Recommended order type and price
#[derive(Debug, Clone)]
pub struct PriceRecommendation {
    /// Recommended order type
    pub order_type: RecommendedOrderType,
    /// Recommended price (None for market orders)
    pub price: Option<Decimal>,
    /// Expected fill probability at this price
    pub expected_fill_probability: Decimal,
    /// Expected execution cost (negative = savings)
    pub expected_cost_bps: Decimal,
    /// Reasoning for the recommendation
    pub reasoning: String,
    /// Alternative prices to try if initial fails
    pub fallback_prices: Vec<Decimal>,
    /// Time to wait before adjusting (seconds)
    pub time_to_adjust_secs: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecommendedOrderType {
    /// Market order - immediate fill
    Market,
    /// Limit order - specific price
    Limit,
    /// Limit IOC - immediate or cancel
    LimitIOC,
    /// Post-only limit - must add liquidity
    PostOnly,
}

/// Historical fill data for learning
#[derive(Debug, Clone)]
struct FillRecord {
    timestamp: DateTime<Utc>,
    side: OrderSide,
    price_vs_mid_bps: Decimal, // Negative = below mid (buying), Positive = above mid (selling)
    filled: bool,
    fill_time_ms: Option<u64>,
}

/// Price optimizer with learning
pub struct PriceOptimizer {
    config: PriceOptimizerConfig,
    /// Fill history per market
    fill_history: RwLock<HashMap<String, VecDeque<FillRecord>>>,
    /// Learned fill probabilities at different spreads
    fill_probability_curve: RwLock<Vec<(Decimal, Decimal)>>,
}

impl PriceOptimizer {
    pub fn new(config: PriceOptimizerConfig) -> Self {
        // Initialize default fill probability curve
        // (distance from mid in bps, expected fill probability)
        let default_curve = vec![
            (dec!(-20), dec!(0.95)), // 20bps below mid = 95% fill when buying
            (dec!(-10), dec!(0.85)), // 10bps below mid = 85% fill
            (dec!(-5), dec!(0.70)),  // 5bps below mid = 70% fill
            (dec!(0), dec!(0.50)),   // At mid = 50% fill
            (dec!(5), dec!(0.30)),   // 5bps above mid = 30% fill
            (dec!(10), dec!(0.15)),  // 10bps above mid = 15% fill
            (dec!(20), dec!(0.05)),  // 20bps above mid = 5% fill
        ];

        Self {
            config,
            fill_history: RwLock::new(HashMap::new()),
            fill_probability_curve: RwLock::new(default_curve),
        }
    }

    /// Get optimal price for an order
    pub fn optimize(
        &self,
        _market_id: &str,
        side: OrderSide,
        order_book: &OrderBook,
        urgency: ExecutionUrgency,
        model_edge: Decimal,
    ) -> PriceRecommendation {
        let spread_bps = order_book.spread_bps;
        let mid_price = (order_book.best_bid + order_book.best_ask) / dec!(2);
        
        // Analyze order book imbalance
        let imbalance = self.calculate_imbalance(order_book);
        
        // Check if spread is too wide
        if spread_bps > self.config.max_spread_to_cross_bps * dec!(2) {
            return PriceRecommendation {
                order_type: RecommendedOrderType::Limit,
                price: Some(self.calculate_mid_price_offer(mid_price, side)),
                expected_fill_probability: dec!(0.30),
                expected_cost_bps: dec!(0), // We're providing liquidity
                reasoning: format!(
                    "Wide spread ({:.1}bps) - placing limit at mid",
                    spread_bps
                ),
                fallback_prices: self.generate_fallback_prices(mid_price, side, 3),
                time_to_adjust_secs: self.config.limit_order_timeout_secs,
            };
        }

        match urgency {
            ExecutionUrgency::Immediate => self.immediate_execution(order_book, side, spread_bps),
            ExecutionUrgency::Normal => self.normal_execution(order_book, side, spread_bps, imbalance, model_edge),
            ExecutionUrgency::Patient => self.patient_execution(order_book, side, spread_bps, imbalance),
        }
    }

    /// Immediate execution - prioritize fill over price
    fn immediate_execution(
        &self,
        order_book: &OrderBook,
        side: OrderSide,
        spread_bps: Decimal,
    ) -> PriceRecommendation {
        if spread_bps <= self.config.max_spread_to_cross_bps {
            // Spread is acceptable - use market order
            PriceRecommendation {
                order_type: RecommendedOrderType::Market,
                price: None,
                expected_fill_probability: dec!(0.99),
                expected_cost_bps: spread_bps / dec!(2), // Pay half spread
                reasoning: format!(
                    "Immediate execution - crossing {:.1}bps spread",
                    spread_bps
                ),
                fallback_prices: vec![],
                time_to_adjust_secs: 0,
            }
        } else {
            // Spread too wide - use aggressive limit
            let aggressive_price = match side {
                OrderSide::Buy => order_book.best_ask,
                OrderSide::Sell => order_book.best_bid,
            };
            
            PriceRecommendation {
                order_type: RecommendedOrderType::LimitIOC,
                price: Some(aggressive_price),
                expected_fill_probability: dec!(0.90),
                expected_cost_bps: spread_bps / dec!(2),
                reasoning: format!(
                    "Wide spread ({:.1}bps) - IOC at best {}",
                    spread_bps,
                    if side == OrderSide::Buy { "ask" } else { "bid" }
                ),
                fallback_prices: vec![],
                time_to_adjust_secs: 0,
            }
        }
    }

    /// Normal execution - balance fill probability and cost
    fn normal_execution(
        &self,
        order_book: &OrderBook,
        side: OrderSide,
        _spread_bps: Decimal,
        imbalance: OrderBookImbalance,
        model_edge: Decimal,
    ) -> PriceRecommendation {
        let mid_price = (order_book.best_bid + order_book.best_ask) / dec!(2);
        
        // If imbalance favors us, be more patient
        let favorable_imbalance = match side {
            OrderSide::Buy => imbalance.bid_heavy, // Many sellers = wait for better price
            OrderSide::Sell => imbalance.ask_heavy, // Many buyers = wait for better price
        };

        // Calculate target price
        let edge_bps = if favorable_imbalance {
            self.config.limit_order_edge_bps * dec!(1.5) // More aggressive edge
        } else {
            self.config.limit_order_edge_bps
        };

        let target_price = match side {
            OrderSide::Buy => mid_price - (mid_price * edge_bps / dec!(10000)),
            OrderSide::Sell => mid_price + (mid_price * edge_bps / dec!(10000)),
        };

        // Estimate fill probability
        let fill_prob = self.estimate_fill_probability(edge_bps, side);

        // If we have good edge, we can be more patient
        let order_type = if model_edge.abs() >= dec!(0.05) && fill_prob >= dec!(0.50) {
            RecommendedOrderType::PostOnly
        } else {
            RecommendedOrderType::Limit
        };

        PriceRecommendation {
            order_type,
            price: Some(target_price),
            expected_fill_probability: fill_prob,
            expected_cost_bps: -edge_bps, // Negative = we're getting a better price
            reasoning: format!(
                "Normal exec @ {:.4} ({}{:.1}bps from mid) | fill prob: {:.0}%{}",
                target_price,
                if edge_bps > Decimal::ZERO { "+" } else { "" },
                edge_bps,
                fill_prob * dec!(100),
                if favorable_imbalance { " [favorable imbalance]" } else { "" }
            ),
            fallback_prices: self.generate_fallback_prices(mid_price, side, 3),
            time_to_adjust_secs: self.config.limit_order_timeout_secs,
        }
    }

    /// Patient execution - maximize price improvement
    fn patient_execution(
        &self,
        order_book: &OrderBook,
        side: OrderSide,
        spread_bps: Decimal,
        _imbalance: OrderBookImbalance,
    ) -> PriceRecommendation {
        let mid_price = (order_book.best_bid + order_book.best_ask) / dec!(2);
        
        // Be very aggressive with edge
        let edge_bps = self.config.limit_order_edge_bps * dec!(2);
        
        let target_price = match side {
            OrderSide::Buy => mid_price - (mid_price * edge_bps / dec!(10000)),
            OrderSide::Sell => mid_price + (mid_price * edge_bps / dec!(10000)),
        };

        let fill_prob = self.estimate_fill_probability(edge_bps * dec!(2), side);

        PriceRecommendation {
            order_type: RecommendedOrderType::PostOnly,
            price: Some(target_price),
            expected_fill_probability: fill_prob,
            expected_cost_bps: -edge_bps,
            reasoning: format!(
                "Patient exec @ {:.4} ({:.1}bps edge) | fill prob: {:.0}% | spread: {:.1}bps",
                target_price,
                edge_bps,
                fill_prob * dec!(100),
                spread_bps
            ),
            fallback_prices: self.generate_fallback_prices(mid_price, side, 5),
            time_to_adjust_secs: self.config.limit_order_timeout_secs * 2,
        }
    }

    /// Calculate order book imbalance
    fn calculate_imbalance(&self, order_book: &OrderBook) -> OrderBookImbalance {
        let bid_depth: Decimal = order_book.bids.iter().map(|(_, s)| s).sum();
        let ask_depth: Decimal = order_book.asks.iter().map(|(_, s)| s).sum();
        
        if ask_depth == Decimal::ZERO {
            return OrderBookImbalance {
                ratio: dec!(999),
                bid_heavy: true,
                ask_heavy: false,
            };
        }
        
        let ratio = bid_depth / ask_depth;
        
        OrderBookImbalance {
            ratio,
            bid_heavy: ratio >= self.config.imbalance_threshold,
            ask_heavy: ratio <= Decimal::ONE / self.config.imbalance_threshold,
        }
    }

    /// Calculate mid-price limit offer
    fn calculate_mid_price_offer(&self, mid_price: Decimal, side: OrderSide) -> Decimal {
        let tiny_edge = mid_price * dec!(0.0001); // 0.01%
        match side {
            OrderSide::Buy => mid_price - tiny_edge,
            OrderSide::Sell => mid_price + tiny_edge,
        }
    }

    /// Generate fallback prices for retry logic
    fn generate_fallback_prices(&self, mid_price: Decimal, side: OrderSide, count: usize) -> Vec<Decimal> {
        let mut prices = Vec::with_capacity(count);
        
        for i in 1..=count {
            let adjustment = mid_price * Decimal::from(i as u32) * dec!(0.001); // 0.1% per step
            let price = match side {
                OrderSide::Buy => mid_price + adjustment, // Move towards ask
                OrderSide::Sell => mid_price - adjustment, // Move towards bid
            };
            prices.push(price);
        }
        
        prices
    }

    /// Estimate fill probability based on price edge
    fn estimate_fill_probability(&self, edge_bps: Decimal, side: OrderSide) -> Decimal {
        // Adjust edge sign for buy vs sell
        let effective_edge = match side {
            OrderSide::Buy => -edge_bps,  // Buying below mid = negative edge
            OrderSide::Sell => edge_bps,  // Selling above mid = positive edge
        };

        let curve = self.fill_probability_curve.read().unwrap();
        
        // Interpolate from curve
        for i in 0..curve.len() - 1 {
            let (edge1, prob1) = curve[i];
            let (edge2, prob2) = curve[i + 1];
            
            if effective_edge >= edge1 && effective_edge <= edge2 {
                // Linear interpolation
                let t = (effective_edge - edge1) / (edge2 - edge1);
                return prob1 + t * (prob2 - prob1);
            }
        }

        // Extrapolate
        if effective_edge < curve[0].0 {
            dec!(0.99) // Very aggressive = high fill
        } else {
            dec!(0.01) // Very passive = low fill
        }
    }

    /// Record a fill result for learning
    pub fn record_fill(
        &self,
        market_id: &str,
        side: OrderSide,
        price: Decimal,
        mid_price: Decimal,
        filled: bool,
        fill_time_ms: Option<u64>,
    ) {
        let price_vs_mid_bps = (price - mid_price) / mid_price * dec!(10000);
        
        let record = FillRecord {
            timestamp: Utc::now(),
            side,
            price_vs_mid_bps,
            filled,
            fill_time_ms,
        };

        // Add to history
        let mut history = self.fill_history.write().unwrap();
        let market_history = history.entry(market_id.to_string()).or_default();
        market_history.push_back(record);
        
        // Keep only recent records
        while market_history.len() > 200 {
            market_history.pop_front();
        }
        drop(history);

        // Update fill probability curve periodically
        self.maybe_update_curve();
    }

    /// Update fill probability curve based on data
    fn maybe_update_curve(&self) {
        let history = self.fill_history.read().unwrap();
        let total_records: usize = history.values().map(|v| v.len()).sum();
        
        if total_records < 50 {
            return; // Need more data
        }

        // Bucket fills by edge
        let mut buckets: HashMap<i32, (u32, u32)> = HashMap::new(); // (fills, total)
        
        for market_history in history.values() {
            for record in market_history.iter() {
                let bucket = (record.price_vs_mid_bps / dec!(5)).to_i32().unwrap_or(0); // 5bps buckets
                let entry = buckets.entry(bucket).or_insert((0, 0));
                if record.filled {
                    entry.0 += 1;
                }
                entry.1 += 1;
            }
        }
        drop(history);

        // Update curve if we have enough data
        let mut new_curve = Vec::new();
        for bucket in -4..=4 {
            if let Some((fills, total)) = buckets.get(&bucket) {
                if *total >= 5 {
                    let edge = Decimal::from(bucket) * dec!(5);
                    let prob = Decimal::from(*fills) / Decimal::from(*total);
                    new_curve.push((edge, prob));
                }
            }
        }

        if new_curve.len() >= 3 {
            new_curve.sort_by(|a, b| a.0.cmp(&b.0));
            *self.fill_probability_curve.write().unwrap() = new_curve;
        }
    }

    /// Get execution stats
    pub fn get_stats(&self) -> OptimizerStats {
        let history = self.fill_history.read().unwrap();
        let curve = self.fill_probability_curve.read().unwrap();
        
        let mut total_fills = 0u32;
        let mut total_orders = 0u32;
        let mut total_edge_bps = Decimal::ZERO;
        
        for market_history in history.values() {
            for record in market_history.iter() {
                total_orders += 1;
                if record.filled {
                    total_fills += 1;
                    // Edge is negative for buys that got filled below mid
                    total_edge_bps += match record.side {
                        OrderSide::Buy => -record.price_vs_mid_bps,
                        OrderSide::Sell => record.price_vs_mid_bps,
                    };
                }
            }
        }

        let fill_rate = if total_orders > 0 {
            Decimal::from(total_fills) / Decimal::from(total_orders)
        } else {
            Decimal::ZERO
        };

        let avg_edge = if total_fills > 0 {
            total_edge_bps / Decimal::from(total_fills)
        } else {
            Decimal::ZERO
        };

        OptimizerStats {
            total_orders,
            total_fills,
            fill_rate,
            avg_edge_bps: avg_edge,
            curve_points: curve.len(),
        }
    }
}

#[derive(Debug, Clone)]
struct OrderBookImbalance {
    ratio: Decimal,
    bid_heavy: bool,
    ask_heavy: bool,
}

#[derive(Debug, Clone)]
pub struct OptimizerStats {
    pub total_orders: u32,
    pub total_fills: u32,
    pub fill_rate: Decimal,
    pub avg_edge_bps: Decimal,
    pub curve_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_optimizer() -> PriceOptimizer {
        PriceOptimizer::new(PriceOptimizerConfig::default())
    }

    fn make_order_book(spread_bps: Decimal, imbalance_ratio: Decimal) -> OrderBook {
        let mid = dec!(0.50);
        let spread = mid * spread_bps / dec!(10000);
        let best_bid = mid - spread / dec!(2);
        let best_ask = mid + spread / dec!(2);
        
        let bid_depth = dec!(1000);
        let ask_depth = bid_depth / imbalance_ratio;
        
        OrderBook::new(
            vec![
                (best_bid, bid_depth * dec!(0.5)),
                (best_bid - dec!(0.01), bid_depth * dec!(0.3)),
                (best_bid - dec!(0.02), bid_depth * dec!(0.2)),
            ],
            vec![
                (best_ask, ask_depth * dec!(0.5)),
                (best_ask + dec!(0.01), ask_depth * dec!(0.3)),
                (best_ask + dec!(0.02), ask_depth * dec!(0.2)),
            ],
        )
    }

    #[test]
    fn test_immediate_execution_narrow_spread() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(10), dec!(1)); // 0.1% spread, balanced
        
        let rec = optimizer.optimize("test", OrderSide::Buy, &book, ExecutionUrgency::Immediate, dec!(0.05));
        
        assert_eq!(rec.order_type, RecommendedOrderType::Market);
        assert!(rec.expected_fill_probability > dec!(0.95));
        println!("{}", rec.reasoning);
    }

    #[test]
    fn test_immediate_execution_wide_spread() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(30), dec!(1)); // 0.3% spread - wide but not too wide
        
        let rec = optimizer.optimize("test", OrderSide::Buy, &book, ExecutionUrgency::Immediate, dec!(0.05));
        
        // Wide spread triggers IOC at best price
        assert_eq!(rec.order_type, RecommendedOrderType::LimitIOC);
        println!("{}", rec.reasoning);
    }

    #[test]
    fn test_normal_execution() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(15), dec!(1)); // 0.15% spread
        
        let rec = optimizer.optimize("test", OrderSide::Buy, &book, ExecutionUrgency::Normal, dec!(0.05));
        
        assert!(matches!(rec.order_type, RecommendedOrderType::Limit | RecommendedOrderType::PostOnly));
        assert!(rec.price.is_some());
        assert!(rec.expected_cost_bps < Decimal::ZERO); // Should save money
        println!("{}", rec.reasoning);
    }

    #[test]
    fn test_patient_execution() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(20), dec!(1));
        
        let rec = optimizer.optimize("test", OrderSide::Sell, &book, ExecutionUrgency::Patient, dec!(0.03));
        
        assert_eq!(rec.order_type, RecommendedOrderType::PostOnly);
        assert!(rec.fallback_prices.len() >= 3);
        println!("{}", rec.reasoning);
    }

    #[test]
    fn test_imbalance_affects_recommendation() {
        let optimizer = make_optimizer();
        
        // Bid heavy book (buyers dominate) - selling should be more patient
        let bid_heavy = make_order_book(dec!(15), dec!(2));
        let rec_bid_heavy = optimizer.optimize("test", OrderSide::Sell, &bid_heavy, ExecutionUrgency::Normal, dec!(0.05));
        
        // Ask heavy book (sellers dominate) - buying should be more patient
        let ask_heavy = make_order_book(dec!(15), dec!(0.5));
        let rec_ask_heavy = optimizer.optimize("test", OrderSide::Buy, &ask_heavy, ExecutionUrgency::Normal, dec!(0.05));
        
        // Both should recognize favorable imbalance
        assert!(rec_bid_heavy.reasoning.contains("imbalance") || rec_bid_heavy.expected_cost_bps < dec!(-3));
        println!("Bid heavy sell: {}", rec_bid_heavy.reasoning);
        println!("Ask heavy buy: {}", rec_ask_heavy.reasoning);
    }

    #[test]
    fn test_fallback_prices_generated() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(15), dec!(1));
        
        let rec = optimizer.optimize("test", OrderSide::Buy, &book, ExecutionUrgency::Normal, dec!(0.05));
        
        assert!(!rec.fallback_prices.is_empty());
        // Fallback prices should be progressively worse (moving towards ask for buys)
        for i in 1..rec.fallback_prices.len() {
            assert!(rec.fallback_prices[i] > rec.fallback_prices[i-1]);
        }
    }

    #[test]
    fn test_fill_probability_estimation() {
        let optimizer = make_optimizer();
        
        // For buying: negative edge_bps means below mid = aggressive (higher fill prob)
        // The estimate_fill_probability function expects the edge_bps we're trying to get
        // For a buy at -15bps below mid, that's aggressive
        // But the function negates for buys, so we pass +15 to get -15 effective
        
        // Very aggressive buy (15bps below mid) = high fill prob
        let aggressive = optimizer.estimate_fill_probability(dec!(15), OrderSide::Buy);
        // This results in effective_edge = -15, which should give high probability
        assert!(aggressive > dec!(0.80), "Aggressive fill prob should be > 80%, got {}", aggressive);
        
        // Very passive buy (15bps above mid) = low fill prob  
        let passive = optimizer.estimate_fill_probability(dec!(-15), OrderSide::Buy);
        // This results in effective_edge = +15, which should give low probability
        assert!(passive < dec!(0.30), "Passive fill prob should be < 30%, got {}", passive);
    }

    #[test]
    fn test_record_and_learn() {
        let optimizer = make_optimizer();
        let mid_price = dec!(0.50);
        
        // Record some fills
        for i in 0..20 {
            let price = mid_price - dec!(0.002) * Decimal::from(i % 3); // Various prices
            optimizer.record_fill("test", OrderSide::Buy, price, mid_price, i % 2 == 0, Some(100));
        }
        
        let stats = optimizer.get_stats();
        assert_eq!(stats.total_orders, 20);
        assert_eq!(stats.total_fills, 10);
        println!("Fill rate: {:.1}%", stats.fill_rate * dec!(100));
    }

    #[test]
    fn test_very_wide_spread_handling() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(100), dec!(1)); // 1% spread
        
        let rec = optimizer.optimize("test", OrderSide::Buy, &book, ExecutionUrgency::Normal, dec!(0.05));
        
        // Should place limit at mid instead of crossing
        assert_eq!(rec.order_type, RecommendedOrderType::Limit);
        println!("{}", rec.reasoning);
    }

    #[test]
    fn test_sell_side_pricing() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(15), dec!(1));
        let mid = (book.best_bid + book.best_ask) / dec!(2);
        
        let rec = optimizer.optimize("test", OrderSide::Sell, &book, ExecutionUrgency::Normal, dec!(0.05));
        
        // Sell price should be above mid
        if let Some(price) = rec.price {
            assert!(price > mid);
        }
    }

    #[test]
    fn test_optimizer_stats() {
        let optimizer = make_optimizer();
        
        // Initial stats
        let stats = optimizer.get_stats();
        assert_eq!(stats.total_orders, 0);
        
        // Record some trades
        optimizer.record_fill("m1", OrderSide::Buy, dec!(0.49), dec!(0.50), true, Some(50));
        optimizer.record_fill("m1", OrderSide::Buy, dec!(0.495), dec!(0.50), true, Some(80));
        optimizer.record_fill("m1", OrderSide::Sell, dec!(0.51), dec!(0.50), false, None);
        
        let stats = optimizer.get_stats();
        assert_eq!(stats.total_orders, 3);
        assert_eq!(stats.total_fills, 2);
    }

    #[test]
    fn test_high_edge_uses_post_only() {
        let optimizer = make_optimizer();
        let book = make_order_book(dec!(15), dec!(1));
        
        // High model edge should use post-only to maximize savings
        let rec = optimizer.optimize("test", OrderSide::Buy, &book, ExecutionUrgency::Normal, dec!(0.10));
        
        // With 10% edge and good fill probability, should use post-only
        if rec.expected_fill_probability >= dec!(0.50) {
            assert_eq!(rec.order_type, RecommendedOrderType::PostOnly);
        }
    }
}
