//! Market evaluator for paper trading decisions

use crate::client::GammaClient;
use crate::error::Result;
use crate::types::Market;
use crate::scanner::{RSI, StochRSI};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Confidence level for trading signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    Low,      // 30-50%
    Medium,   // 50-70%
    High,     // 70-85%
    VeryHigh, // 85%+
}

impl ConfidenceLevel {
    pub fn from_score(score: Decimal) -> Self {
        if score >= dec!(0.85) {
            ConfidenceLevel::VeryHigh
        } else if score >= dec!(0.70) {
            ConfidenceLevel::High
        } else if score >= dec!(0.50) {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }
}

impl std::fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfidenceLevel::Low => write!(f, "🟡 Low"),
            ConfidenceLevel::Medium => write!(f, "🟠 Medium"),
            ConfidenceLevel::High => write!(f, "🟢 High"),
            ConfidenceLevel::VeryHigh => write!(f, "💎 Very High"),
        }
    }
}

/// Evaluation result for a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Market ID
    pub market_id: String,
    /// Market question
    pub question: String,
    /// Current YES price
    pub yes_price: Decimal,
    /// Current NO price
    pub no_price: Decimal,
    /// 24h volume
    pub volume_24h: Decimal,
    /// Liquidity available
    pub liquidity: Decimal,
    /// Technical indicators
    pub indicators: TechnicalIndicators,
    /// Market characteristics
    pub characteristics: MarketCharacteristics,
    /// Trading signal
    pub signal: TradingSignal,
    /// Overall confidence score (0-1)
    pub confidence_score: Decimal,
    /// Confidence level
    pub confidence_level: ConfidenceLevel,
    /// Reasons for the signal
    pub reasons: Vec<String>,
    /// Warnings/risks
    pub warnings: Vec<String>,
    /// Suggested position size (% of portfolio)
    pub suggested_size_pct: Decimal,
    /// Evaluation timestamp
    pub evaluated_at: DateTime<Utc>,
}

/// Technical indicators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    pub rsi_14: Option<Decimal>,
    pub stoch_rsi_k: Option<Decimal>,
    pub stoch_rsi_d: Option<Decimal>,
    pub price_momentum: Option<Decimal>, // % change over period
    pub volatility: Option<Decimal>,     // Standard deviation
}

/// Market characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketCharacteristics {
    pub is_crypto: bool,
    pub is_political: bool,
    pub is_sports: bool,
    pub time_to_resolution: Option<Duration>,
    pub has_active_orders: bool,
    pub spread_pct: Decimal,
    pub price_efficiency: Decimal, // How close YES + NO is to 1.0
}

impl Default for MarketCharacteristics {
    fn default() -> Self {
        Self {
            is_crypto: false,
            is_political: false,
            is_sports: false,
            time_to_resolution: None,
            has_active_orders: false,
            spread_pct: dec!(0),
            price_efficiency: dec!(1),
        }
    }
}

/// Trading signal recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    StrongBuyYes,
    BuyYes,
    Hold,
    BuyNo,
    StrongBuyNo,
    Avoid, // High risk, stay away
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingSignal::StrongBuyYes => write!(f, "📈 Strong Buy YES"),
            TradingSignal::BuyYes => write!(f, "🟢 Buy YES"),
            TradingSignal::Hold => write!(f, "⏸️ Hold"),
            TradingSignal::BuyNo => write!(f, "🔴 Buy NO"),
            TradingSignal::StrongBuyNo => write!(f, "📉 Strong Buy NO"),
            TradingSignal::Avoid => write!(f, "⚠️ Avoid"),
        }
    }
}

/// Market evaluator
pub struct MarketEvaluator {
    #[allow(dead_code)]
    client: GammaClient,
    price_history: HashMap<String, Vec<(DateTime<Utc>, Decimal)>>,
}

impl MarketEvaluator {
    pub fn new(client: GammaClient) -> Self {
        Self {
            client,
            price_history: HashMap::new(),
        }
    }

    /// Evaluate a market for trading
    pub async fn evaluate(&mut self, market: &Market) -> Result<EvaluationResult> {
        let yes_price = market.outcomes.first()
            .map(|o| o.price)
            .unwrap_or(dec!(0.5));
        
        let no_price = market.outcomes.get(1)
            .map(|o| o.price)
            .unwrap_or(dec!(0.5));

        // Calculate characteristics
        let characteristics = self.analyze_characteristics(market, yes_price, no_price);
        
        // Get technical indicators
        let indicators = self.calculate_indicators(&market.id, yes_price);
        
        // Generate signal
        let (signal, confidence_score, reasons, warnings) = 
            self.generate_signal(&characteristics, &indicators, yes_price);
        
        let confidence_level = ConfidenceLevel::from_score(confidence_score);
        
        // Suggest position size based on confidence
        let suggested_size_pct = match confidence_level {
            ConfidenceLevel::VeryHigh => dec!(5),
            ConfidenceLevel::High => dec!(3),
            ConfidenceLevel::Medium => dec!(2),
            ConfidenceLevel::Low => dec!(1),
        };

        Ok(EvaluationResult {
            market_id: market.id.clone(),
            question: market.question.clone(),
            yes_price,
            no_price,
            volume_24h: market.volume,
            liquidity: market.liquidity,
            indicators,
            characteristics,
            signal,
            confidence_score,
            confidence_level,
            reasons,
            warnings,
            suggested_size_pct,
            evaluated_at: Utc::now(),
        })
    }

    /// Evaluate multiple markets and rank by opportunity
    pub async fn evaluate_batch(&mut self, markets: &[Market]) -> Vec<EvaluationResult> {
        let mut results = Vec::new();
        
        for market in markets {
            if let Ok(eval) = self.evaluate(market).await {
                results.push(eval);
            }
        }

        // Sort by confidence score descending
        results.sort_by(|a, b| b.confidence_score.cmp(&a.confidence_score));
        results
    }

    /// Record price for history tracking
    pub fn record_price(&mut self, market_id: &str, price: Decimal) {
        let history = self.price_history
            .entry(market_id.to_string())
            .or_insert_with(Vec::new);
        
        history.push((Utc::now(), price));
        
        // Keep last 100 data points
        if history.len() > 100 {
            history.remove(0);
        }
    }

    fn analyze_characteristics(
        &self,
        market: &Market,
        yes_price: Decimal,
        no_price: Decimal,
    ) -> MarketCharacteristics {
        let question_lower = market.question.to_lowercase();
        
        let is_crypto = question_lower.contains("bitcoin") 
            || question_lower.contains("btc")
            || question_lower.contains("ethereum")
            || question_lower.contains("eth")
            || question_lower.contains("crypto");
        
        let is_political = question_lower.contains("election")
            || question_lower.contains("president")
            || question_lower.contains("vote")
            || question_lower.contains("trump")
            || question_lower.contains("biden");
        
        let is_sports = question_lower.contains("win ")
            || question_lower.contains("championship")
            || question_lower.contains("super bowl")
            || question_lower.contains("nba")
            || question_lower.contains("nfl");

        // Price efficiency: how close YES + NO is to 1.0
        let total = yes_price + no_price;
        let price_efficiency = if total > dec!(0) {
            dec!(1) - (total - dec!(1)).abs()
        } else {
            dec!(0)
        };

        // Simple spread estimate
        let spread_pct = if yes_price > dec!(0) {
            ((yes_price - no_price).abs() / yes_price) * dec!(100)
        } else {
            dec!(0)
        };

        MarketCharacteristics {
            is_crypto,
            is_political,
            is_sports,
            time_to_resolution: None, // Would need end_date parsing
            has_active_orders: market.active,
            spread_pct,
            price_efficiency,
        }
    }

    fn calculate_indicators(
        &self,
        market_id: &str,
        _current_price: Decimal,
    ) -> TechnicalIndicators {
        let mut indicators = TechnicalIndicators::default();

        // Get price history
        if let Some(history) = self.price_history.get(market_id) {
            if history.len() >= 14 {
                let prices: Vec<f64> = history.iter()
                    .map(|(_, p)| p.to_string().parse().unwrap_or(0.5))
                    .collect();

                // Calculate RSI
                let mut rsi = RSI::new(14);
                for price in &prices {
                    rsi.update(*price);
                }
                let rsi_val = rsi.value();
                if rsi_val > 0.0 {
                    indicators.rsi_14 = Decimal::from_f64_retain(rsi_val);
                }

                // Calculate StochRSI
                let mut stoch = StochRSI::new(14, 14, 3, 3);
                let mut last_result = None;
                for price in &prices {
                    last_result = Some(stoch.update(*price));
                }
                if let Some(result) = last_result {
                    if result.ready {
                        indicators.stoch_rsi_k = Decimal::from_f64_retain(result.k);
                        indicators.stoch_rsi_d = Decimal::from_f64_retain(result.d);
                    }
                }

                // Calculate momentum (% change from start to now)
                if let (Some(first), Some(last)) = (prices.first(), prices.last()) {
                    if *first > 0.0 {
                        let momentum = ((last - first) / first) * 100.0;
                        indicators.price_momentum = Decimal::from_f64_retain(momentum);
                    }
                }
            }
        }

        indicators
    }

    fn generate_signal(
        &self,
        characteristics: &MarketCharacteristics,
        indicators: &TechnicalIndicators,
        yes_price: Decimal,
    ) -> (TradingSignal, Decimal, Vec<String>, Vec<String>) {
        let mut score = dec!(0.5); // Start neutral
        let mut reasons = Vec::new();
        let mut warnings = Vec::new();

        // Price-based signals
        if yes_price < dec!(0.20) {
            score += dec!(0.15);
            reasons.push("YES price is low (potential value)".to_string());
        } else if yes_price > dec!(0.80) {
            score -= dec!(0.10);
            reasons.push("YES price is high (less upside)".to_string());
        }

        // RSI signals
        if let Some(rsi) = indicators.rsi_14 {
            if rsi < dec!(30) {
                score += dec!(0.20);
                reasons.push(format!("RSI oversold: {:.1}", rsi));
            } else if rsi > dec!(70) {
                score -= dec!(0.15);
                reasons.push(format!("RSI overbought: {:.1}", rsi));
            }
        }

        // StochRSI signals
        if let (Some(k), Some(d)) = (indicators.stoch_rsi_k, indicators.stoch_rsi_d) {
            if k < dec!(20) && d < dec!(20) {
                score += dec!(0.15);
                reasons.push("StochRSI deeply oversold".to_string());
            } else if k > dec!(80) && d > dec!(80) {
                score -= dec!(0.10);
                reasons.push("StochRSI overbought".to_string());
            }
            
            // Bullish crossover
            if k > d && k < dec!(30) {
                score += dec!(0.10);
                reasons.push("StochRSI bullish crossover".to_string());
            }
        }

        // Momentum
        if let Some(momentum) = indicators.price_momentum {
            if momentum < dec!(-10) {
                score += dec!(0.10);
                reasons.push(format!("Price dropped {:.1}% (potential reversal)", momentum));
            } else if momentum > dec!(20) {
                score -= dec!(0.05);
                warnings.push("Strong recent rally (may be late)".to_string());
            }
        }

        // Market characteristics adjustments
        if characteristics.is_crypto {
            score += dec!(0.05); // Crypto markets have clearer patterns
        }
        
        if characteristics.price_efficiency < dec!(0.95) {
            warnings.push("Market inefficiency detected".to_string());
        }

        if !characteristics.has_active_orders {
            score -= dec!(0.20);
            warnings.push("Market may be inactive".to_string());
        }

        // Clamp score
        score = score.max(dec!(0)).min(dec!(1));

        // Determine signal
        let signal = if score >= dec!(0.85) {
            TradingSignal::StrongBuyYes
        } else if score >= dec!(0.65) {
            TradingSignal::BuyYes
        } else if score >= dec!(0.45) {
            TradingSignal::Hold
        } else if score >= dec!(0.25) {
            TradingSignal::BuyNo
        } else if score >= dec!(0.15) {
            TradingSignal::StrongBuyNo
        } else {
            TradingSignal::Avoid
        };

        // Add warnings for low confidence
        if score > dec!(0.40) && score < dec!(0.60) {
            warnings.push("Mixed signals - proceed with caution".to_string());
        }

        (signal, score, reasons, warnings)
    }

    /// Format evaluation as readable report
    pub fn format_report(eval: &EvaluationResult) -> String {
        let mut report = String::new();
        
        report.push_str("📊 **Market Evaluation**\n");
        report.push_str("━━━━━━━━━━━━━━━━━━━━━\n");
        report.push_str(&format!("**{}**\n\n", eval.question));
        
        report.push_str("**Prices:**\n");
        report.push_str(&format!("• YES: {:.1}¢\n", eval.yes_price * dec!(100)));
        report.push_str(&format!("• NO: {:.1}¢\n", eval.no_price * dec!(100)));
        report.push_str(&format!("• Volume 24h: ${:.0}\n\n", eval.volume_24h));
        
        if eval.indicators.rsi_14.is_some() || eval.indicators.stoch_rsi_k.is_some() {
            report.push_str("**Technical Indicators:**\n");
            if let Some(rsi) = eval.indicators.rsi_14 {
                report.push_str(&format!("• RSI(14): {:.1}\n", rsi));
            }
            if let (Some(k), Some(d)) = (eval.indicators.stoch_rsi_k, eval.indicators.stoch_rsi_d) {
                report.push_str(&format!("• StochRSI: K={:.1}, D={:.1}\n", k, d));
            }
            if let Some(mom) = eval.indicators.price_momentum {
                report.push_str(&format!("• Momentum: {:.1}%\n", mom));
            }
            report.push('\n');
        }
        
        report.push_str(&format!("**Signal:** {}\n", eval.signal));
        report.push_str(&format!("**Confidence:** {} ({:.0}%)\n\n", 
            eval.confidence_level, eval.confidence_score * dec!(100)));
        
        if !eval.reasons.is_empty() {
            report.push_str("**Reasons:**\n");
            for reason in &eval.reasons {
                report.push_str(&format!("✓ {}\n", reason));
            }
            report.push('\n');
        }
        
        if !eval.warnings.is_empty() {
            report.push_str("**Warnings:**\n");
            for warning in &eval.warnings {
                report.push_str(&format!("⚠️ {}\n", warning));
            }
            report.push('\n');
        }
        
        report.push_str(&format!("**Suggested Size:** {:.1}% of portfolio\n", 
            eval.suggested_size_pct));
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Outcome;

    fn create_test_market(yes_price: Decimal) -> Market {
        Market {
            id: "test123".to_string(),
            question: "Will Bitcoin reach 100k by end of 2025?".to_string(),
            description: None,
            end_date: None,
            outcomes: vec![
                Outcome {
                    token_id: "yes".to_string(),
                    outcome: "Yes".to_string(),
                    price: yes_price,
                },
                Outcome {
                    token_id: "no".to_string(),
                    outcome: "No".to_string(),
                    price: dec!(1) - yes_price,
                },
            ],
            volume: dec!(50000),
            liquidity: dec!(10000),
            active: true,
            closed: false,
        }
    }

    fn create_test_client() -> GammaClient {
        GammaClient::new("https://gamma-api.polymarket.com").unwrap()
    }

    #[tokio::test]
    async fn test_evaluate_low_price() {
        let client = create_test_client();
        let mut evaluator = MarketEvaluator::new(client);
        
        let market = create_test_market(dec!(0.15)); // Low YES price
        let result = evaluator.evaluate(&market).await.unwrap();
        
        assert!(result.confidence_score > dec!(0.5));
        assert!(result.reasons.iter().any(|r| r.contains("low")));
    }

    #[tokio::test]
    async fn test_evaluate_high_price() {
        let client = create_test_client();
        let mut evaluator = MarketEvaluator::new(client);
        
        let market = create_test_market(dec!(0.85)); // High YES price
        let result = evaluator.evaluate(&market).await.unwrap();
        
        assert!(result.confidence_score < dec!(0.6));
    }

    #[tokio::test]
    async fn test_evaluate_with_history() {
        let client = create_test_client();
        let mut evaluator = MarketEvaluator::new(client);
        
        // Add some price history (declining prices = oversold)
        for i in 0..20 {
            let price = dec!(0.6) - Decimal::from(i) * dec!(0.02);
            evaluator.record_price("test123", price);
        }
        
        let market = create_test_market(dec!(0.20));
        let result = evaluator.evaluate(&market).await.unwrap();
        
        // Should have RSI calculated
        assert!(result.indicators.rsi_14.is_some() || result.indicators.price_momentum.is_some());
    }

    #[test]
    fn test_confidence_levels() {
        assert_eq!(ConfidenceLevel::from_score(dec!(0.90)), ConfidenceLevel::VeryHigh);
        assert_eq!(ConfidenceLevel::from_score(dec!(0.75)), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_score(dec!(0.55)), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::from_score(dec!(0.35)), ConfidenceLevel::Low);
    }

    #[test]
    fn test_format_report() {
        let eval = EvaluationResult {
            market_id: "test".to_string(),
            question: "Test market?".to_string(),
            yes_price: dec!(0.65),
            no_price: dec!(0.35),
            volume_24h: dec!(10000),
            liquidity: dec!(5000),
            indicators: TechnicalIndicators {
                rsi_14: Some(dec!(35)),
                stoch_rsi_k: Some(dec!(25)),
                stoch_rsi_d: Some(dec!(30)),
                ..Default::default()
            },
            characteristics: MarketCharacteristics::default(),
            signal: TradingSignal::BuyYes,
            confidence_score: dec!(0.72),
            confidence_level: ConfidenceLevel::High,
            reasons: vec!["RSI oversold".to_string()],
            warnings: vec![],
            suggested_size_pct: dec!(3),
            evaluated_at: Utc::now(),
        };
        
        let report = MarketEvaluator::format_report(&eval);
        assert!(report.contains("Test market"));
        assert!(report.contains("RSI"));
        assert!(report.contains("Buy YES"));
    }

    #[test]
    fn test_crypto_detection() {
        let client = create_test_client();
        let evaluator = MarketEvaluator::new(client);
        
        let market = Market {
            id: "btc_test".to_string(),
            question: "Will Bitcoin hit 150k?".to_string(),
            description: None,
            end_date: None,
            outcomes: vec![],
            volume: dec!(0),
            liquidity: dec!(0),
            active: true,
            closed: false,
        };
        
        let characteristics = evaluator.analyze_characteristics(&market, dec!(0.5), dec!(0.5));
        assert!(characteristics.is_crypto);
    }
}
