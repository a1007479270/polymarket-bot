//! Signal aggregation and final decision engine
//!
//! Combines multiple signal sources into a unified trading decision:
//! 1. LLM model predictions
//! 2. Technical analysis signals
//! 3. Sentiment analysis
//! 4. Arbitrage opportunities
//! 5. Copy trading signals
//!
//! Uses weighted voting and consensus to improve signal quality.

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

/// A single signal source's opinion
#[derive(Debug, Clone)]
pub struct SignalSource {
    pub name: String,
    pub signal_type: SignalType,
    pub direction: SignalDirection,
    pub confidence: Decimal,
    pub edge: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalType {
    /// LLM-based prediction
    LlmPrediction,
    /// Technical analysis (trend, momentum)
    Technical,
    /// Sentiment from social media
    Sentiment,
    /// Arbitrage opportunity
    Arbitrage,
    /// Copy trading signal
    CopyTrade,
    /// Orderbook analysis
    OrderFlow,
    /// External data source
    External,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalDirection {
    /// Strong buy signal
    StrongBuy,
    /// Moderate buy signal
    Buy,
    /// Neutral - no position
    Neutral,
    /// Moderate sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl SignalDirection {
    /// Convert to numeric value for aggregation
    pub fn to_value(&self) -> Decimal {
        match self {
            Self::StrongBuy => dec!(2),
            Self::Buy => dec!(1),
            Self::Neutral => dec!(0),
            Self::Sell => dec!(-1),
            Self::StrongSell => dec!(-2),
        }
    }

    /// Create from numeric value
    pub fn from_value(value: Decimal) -> Self {
        if value >= dec!(1.5) {
            Self::StrongBuy
        } else if value >= dec!(0.5) {
            Self::Buy
        } else if value <= dec!(-1.5) {
            Self::StrongSell
        } else if value <= dec!(-0.5) {
            Self::Sell
        } else {
            Self::Neutral
        }
    }

    /// Check if bullish
    pub fn is_bullish(&self) -> bool {
        matches!(self, Self::StrongBuy | Self::Buy)
    }

    /// Check if bearish
    pub fn is_bearish(&self) -> bool {
        matches!(self, Self::StrongSell | Self::Sell)
    }
}

/// Aggregated decision from multiple signals
#[derive(Debug, Clone)]
pub struct AggregatedDecision {
    /// Final direction recommendation
    pub direction: SignalDirection,
    /// Combined confidence (0-1)
    pub confidence: Decimal,
    /// Weighted score (-2 to +2)
    pub score: Decimal,
    /// Number of sources agreeing
    pub consensus_count: usize,
    /// Total sources considered
    pub total_sources: usize,
    /// Agreement percentage
    pub agreement_pct: Decimal,
    /// Suggested position size multiplier
    pub size_multiplier: Decimal,
    /// Should we act on this decision?
    pub should_act: bool,
    /// Reasons for the decision
    pub reasons: Vec<String>,
    /// Contributing signals
    pub contributing_signals: Vec<String>,
}

/// Configuration for signal aggregation
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// Minimum sources required to make a decision
    pub min_sources: usize,
    /// Minimum agreement percentage to act
    pub min_agreement_pct: Decimal,
    /// Minimum confidence to act
    pub min_confidence: Decimal,
    /// Signal expiry time (seconds)
    pub signal_expiry_secs: i64,
    /// Weights for each signal type
    pub weights: SignalWeights,
    /// Require LLM signal for final decision
    pub require_llm: bool,
    /// Conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,
}

#[derive(Debug, Clone)]
pub struct SignalWeights {
    pub llm: Decimal,
    pub technical: Decimal,
    pub sentiment: Decimal,
    pub arbitrage: Decimal,
    pub copy_trade: Decimal,
    pub order_flow: Decimal,
    pub external: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictStrategy {
    /// Side with the majority
    Majority,
    /// Go with highest confidence
    HighestConfidence,
    /// Stay neutral on conflict
    Conservative,
    /// Weight by historical accuracy
    HistoricalAccuracy,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            min_sources: 2,
            min_agreement_pct: dec!(0.60), // 60% must agree
            min_confidence: dec!(0.55),
            signal_expiry_secs: 300, // 5 minutes
            weights: SignalWeights::default(),
            require_llm: true,
            conflict_strategy: ConflictStrategy::Majority,
        }
    }
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            llm: dec!(0.35),         // 35% - primary source
            technical: dec!(0.20),   // 20%
            sentiment: dec!(0.15),   // 15%
            arbitrage: dec!(0.10),   // 10%
            copy_trade: dec!(0.10),  // 10%
            order_flow: dec!(0.05),  // 5%
            external: dec!(0.05),    // 5%
        }
    }
}

/// Signal aggregator - combines multiple signals into decisions
pub struct SignalAggregator {
    config: AggregatorConfig,
    /// Historical accuracy per signal type
    accuracy_history: HashMap<SignalType, AccuracyTracker>,
}

#[derive(Debug, Clone, Default)]
struct AccuracyTracker {
    wins: u32,
    losses: u32,
}

impl AccuracyTracker {
    fn accuracy(&self) -> Decimal {
        let total = self.wins + self.losses;
        if total == 0 {
            dec!(0.5) // Default 50% if no history
        } else {
            Decimal::from(self.wins) / Decimal::from(total)
        }
    }

    fn record(&mut self, won: bool) {
        if won {
            self.wins += 1;
        } else {
            self.losses += 1;
        }
    }
}

impl SignalAggregator {
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            config,
            accuracy_history: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(AggregatorConfig::default())
    }

    /// Aggregate multiple signals into a decision
    pub fn aggregate(&self, market_id: &str, signals: &[SignalSource]) -> AggregatedDecision {
        let now = Utc::now();
        let mut reasons = Vec::new();

        // Filter expired signals
        let active_signals: Vec<_> = signals
            .iter()
            .filter(|s| {
                let age = now - s.timestamp;
                age <= Duration::seconds(self.config.signal_expiry_secs)
            })
            .collect();

        if active_signals.is_empty() {
            return self.no_decision("No active signals");
        }

        // Check minimum sources
        if active_signals.len() < self.config.min_sources {
            return self.no_decision(&format!(
                "Insufficient sources: {} < {} required",
                active_signals.len(),
                self.config.min_sources
            ));
        }

        // Check for required LLM signal
        if self.config.require_llm {
            let has_llm = active_signals
                .iter()
                .any(|s| s.signal_type == SignalType::LlmPrediction);
            if !has_llm {
                return self.no_decision("Required LLM signal missing");
            }
        }

        // Calculate weighted scores
        let mut weighted_sum = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        let mut confidence_sum = Decimal::ZERO;
        let mut contributing: Vec<String> = Vec::new();

        for signal in &active_signals {
            let base_weight = self.get_weight(signal.signal_type);
            
            // Adjust weight by confidence
            let confidence_adj = signal.confidence;
            
            // Adjust by historical accuracy if using that strategy
            let accuracy_adj = if self.config.conflict_strategy == ConflictStrategy::HistoricalAccuracy {
                self.accuracy_history
                    .get(&signal.signal_type)
                    .map(|t| t.accuracy())
                    .unwrap_or(dec!(0.5))
            } else {
                dec!(1)
            };

            let final_weight = base_weight * confidence_adj * accuracy_adj;
            let signal_value = signal.direction.to_value();

            weighted_sum += signal_value * final_weight;
            total_weight += final_weight;
            confidence_sum += signal.confidence;

            contributing.push(format!(
                "{}: {:?} ({:.0}%)",
                signal.name,
                signal.direction,
                signal.confidence * dec!(100)
            ));
        }

        // Calculate final score
        let score = if total_weight > Decimal::ZERO {
            weighted_sum / total_weight
        } else {
            Decimal::ZERO
        };

        let direction = SignalDirection::from_value(score);
        let avg_confidence = confidence_sum / Decimal::from(active_signals.len() as u32);

        // Calculate consensus
        let (consensus_count, agreement_pct) = self.calculate_consensus(&active_signals, direction);
        
        reasons.push(format!(
            "Score: {:.2}, Direction: {:?}",
            score, direction
        ));
        reasons.push(format!(
            "Consensus: {}/{} ({:.0}%)",
            consensus_count,
            active_signals.len(),
            agreement_pct * dec!(100)
        ));

        // Determine if we should act
        let should_act = self.should_act(
            direction,
            avg_confidence,
            agreement_pct,
            &active_signals,
        );

        // Calculate size multiplier based on signal strength
        let size_multiplier = self.calculate_size_multiplier(
            score.abs(),
            avg_confidence,
            agreement_pct,
        );

        if !should_act {
            reasons.push(self.get_block_reason(direction, avg_confidence, agreement_pct));
        }

        AggregatedDecision {
            direction,
            confidence: avg_confidence,
            score,
            consensus_count,
            total_sources: active_signals.len(),
            agreement_pct,
            size_multiplier,
            should_act,
            reasons,
            contributing_signals: contributing,
        }
    }

    fn no_decision(&self, reason: &str) -> AggregatedDecision {
        AggregatedDecision {
            direction: SignalDirection::Neutral,
            confidence: Decimal::ZERO,
            score: Decimal::ZERO,
            consensus_count: 0,
            total_sources: 0,
            agreement_pct: Decimal::ZERO,
            size_multiplier: Decimal::ZERO,
            should_act: false,
            reasons: vec![reason.to_string()],
            contributing_signals: vec![],
        }
    }

    fn get_weight(&self, signal_type: SignalType) -> Decimal {
        match signal_type {
            SignalType::LlmPrediction => self.config.weights.llm,
            SignalType::Technical => self.config.weights.technical,
            SignalType::Sentiment => self.config.weights.sentiment,
            SignalType::Arbitrage => self.config.weights.arbitrage,
            SignalType::CopyTrade => self.config.weights.copy_trade,
            SignalType::OrderFlow => self.config.weights.order_flow,
            SignalType::External => self.config.weights.external,
        }
    }

    fn calculate_consensus(
        &self,
        signals: &[&SignalSource],
        final_direction: SignalDirection,
    ) -> (usize, Decimal) {
        let is_bullish = final_direction.is_bullish();
        let is_bearish = final_direction.is_bearish();

        let agreeing = signals
            .iter()
            .filter(|s| {
                if is_bullish {
                    s.direction.is_bullish()
                } else if is_bearish {
                    s.direction.is_bearish()
                } else {
                    s.direction == SignalDirection::Neutral
                }
            })
            .count();

        let pct = if signals.is_empty() {
            Decimal::ZERO
        } else {
            Decimal::from(agreeing as u32) / Decimal::from(signals.len() as u32)
        };

        (agreeing, pct)
    }

    fn should_act(
        &self,
        direction: SignalDirection,
        confidence: Decimal,
        agreement_pct: Decimal,
        signals: &[&SignalSource],
    ) -> bool {
        // Must not be neutral
        if direction == SignalDirection::Neutral {
            return false;
        }

        // Check minimum confidence
        if confidence < self.config.min_confidence {
            return false;
        }

        // Check consensus requirement
        if agreement_pct < self.config.min_agreement_pct {
            return false;
        }

        // Check for strong disagreement
        let has_strong_opposite = signals.iter().any(|s| {
            (direction.is_bullish() && s.direction == SignalDirection::StrongSell)
                || (direction.is_bearish() && s.direction == SignalDirection::StrongBuy)
        });

        if has_strong_opposite && self.config.conflict_strategy == ConflictStrategy::Conservative {
            return false;
        }

        true
    }

    fn get_block_reason(
        &self,
        direction: SignalDirection,
        confidence: Decimal,
        agreement_pct: Decimal,
    ) -> String {
        if direction == SignalDirection::Neutral {
            return "Signals cancel out (neutral)".to_string();
        }
        if confidence < self.config.min_confidence {
            return format!(
                "Low confidence: {:.0}% < {:.0}% required",
                confidence * dec!(100),
                self.config.min_confidence * dec!(100)
            );
        }
        if agreement_pct < self.config.min_agreement_pct {
            return format!(
                "Insufficient consensus: {:.0}% < {:.0}% required",
                agreement_pct * dec!(100),
                self.config.min_agreement_pct * dec!(100)
            );
        }
        "Strong disagreement in signals".to_string()
    }

    fn calculate_size_multiplier(
        &self,
        score_magnitude: Decimal,
        confidence: Decimal,
        agreement_pct: Decimal,
    ) -> Decimal {
        // Base multiplier from score magnitude (0-2 range -> 0.5-1.0)
        let score_mult = dec!(0.5) + (score_magnitude / dec!(4));

        // Confidence factor
        let conf_mult = confidence;

        // Agreement factor
        let agree_mult = agreement_pct;

        // Combined (geometric mean approximation)
        let combined = score_mult * conf_mult * agree_mult;
        
        // Normalize to 0.2 - 1.0 range
        (combined * dec!(2)).min(dec!(1)).max(dec!(0.2))
    }

    /// Record outcome for accuracy tracking
    pub fn record_outcome(&mut self, signal_type: SignalType, won: bool) {
        self.accuracy_history
            .entry(signal_type)
            .or_default()
            .record(won);
    }

    /// Get accuracy stats for a signal type
    pub fn get_accuracy(&self, signal_type: SignalType) -> Option<Decimal> {
        self.accuracy_history.get(&signal_type).map(|t| t.accuracy())
    }

    /// Get all accuracy stats
    pub fn get_all_accuracies(&self) -> HashMap<SignalType, Decimal> {
        self.accuracy_history
            .iter()
            .map(|(t, tracker)| (*t, tracker.accuracy()))
            .collect()
    }
}

/// Builder for creating signal sources
pub struct SignalBuilder {
    source: SignalSource,
}

impl SignalBuilder {
    pub fn new(name: &str, signal_type: SignalType) -> Self {
        Self {
            source: SignalSource {
                name: name.to_string(),
                signal_type,
                direction: SignalDirection::Neutral,
                confidence: dec!(0.5),
                edge: None,
                timestamp: Utc::now(),
                metadata: HashMap::new(),
            },
        }
    }

    pub fn direction(mut self, direction: SignalDirection) -> Self {
        self.source.direction = direction;
        self
    }

    pub fn confidence(mut self, confidence: Decimal) -> Self {
        self.source.confidence = confidence;
        self
    }

    pub fn edge(mut self, edge: Decimal) -> Self {
        self.source.edge = Some(edge);
        self
    }

    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.source.timestamp = timestamp;
        self
    }

    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.source.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn build(self) -> SignalSource {
        self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(
        name: &str,
        signal_type: SignalType,
        direction: SignalDirection,
        confidence: Decimal,
    ) -> SignalSource {
        SignalBuilder::new(name, signal_type)
            .direction(direction)
            .confidence(confidence)
            .build()
    }

    #[test]
    fn test_direction_conversion() {
        assert_eq!(SignalDirection::from_value(dec!(1.8)), SignalDirection::StrongBuy);
        assert_eq!(SignalDirection::from_value(dec!(0.8)), SignalDirection::Buy);
        assert_eq!(SignalDirection::from_value(dec!(0.2)), SignalDirection::Neutral);
        assert_eq!(SignalDirection::from_value(dec!(-0.8)), SignalDirection::Sell);
        assert_eq!(SignalDirection::from_value(dec!(-1.8)), SignalDirection::StrongSell);
    }

    #[test]
    fn test_aggregator_basic() {
        let aggregator = SignalAggregator::with_defaults();
        
        let signals = vec![
            make_signal("llm", SignalType::LlmPrediction, SignalDirection::Buy, dec!(0.75)),
            make_signal("tech", SignalType::Technical, SignalDirection::Buy, dec!(0.70)),
            make_signal("sentiment", SignalType::Sentiment, SignalDirection::Buy, dec!(0.65)),
        ];

        let decision = aggregator.aggregate("test_market", &signals);
        
        assert!(decision.direction.is_bullish());
        assert!(decision.should_act);
        assert!(decision.agreement_pct >= dec!(0.9)); // All agree
    }

    #[test]
    fn test_aggregator_conflict() {
        let aggregator = SignalAggregator::with_defaults();
        
        let signals = vec![
            make_signal("llm", SignalType::LlmPrediction, SignalDirection::Buy, dec!(0.70)),
            make_signal("tech", SignalType::Technical, SignalDirection::Sell, dec!(0.70)),
            make_signal("sentiment", SignalType::Sentiment, SignalDirection::Neutral, dec!(0.50)),
        ];

        let decision = aggregator.aggregate("test_market", &signals);
        
        // Should have low agreement due to conflict
        assert!(decision.agreement_pct < dec!(0.6));
    }

    #[test]
    fn test_aggregator_insufficient_sources() {
        let aggregator = SignalAggregator::with_defaults();
        
        let signals = vec![
            make_signal("llm", SignalType::LlmPrediction, SignalDirection::StrongBuy, dec!(0.90)),
        ];

        let decision = aggregator.aggregate("test_market", &signals);
        
        assert!(!decision.should_act);
        assert!(decision.reasons[0].contains("Insufficient"));
    }

    #[test]
    fn test_aggregator_no_llm() {
        let aggregator = SignalAggregator::with_defaults();
        
        let signals = vec![
            make_signal("tech", SignalType::Technical, SignalDirection::Buy, dec!(0.80)),
            make_signal("sentiment", SignalType::Sentiment, SignalDirection::Buy, dec!(0.75)),
        ];

        let decision = aggregator.aggregate("test_market", &signals);
        
        assert!(!decision.should_act);
        assert!(decision.reasons[0].contains("LLM"));
    }

    #[test]
    fn test_aggregator_expired_signal() {
        let aggregator = SignalAggregator::with_defaults();
        
        let old_time = Utc::now() - Duration::seconds(600); // 10 minutes ago
        
        let signals = vec![
            SignalBuilder::new("llm", SignalType::LlmPrediction)
                .direction(SignalDirection::Buy)
                .confidence(dec!(0.80))
                .timestamp(old_time)
                .build(),
            make_signal("tech", SignalType::Technical, SignalDirection::Buy, dec!(0.75)),
        ];

        let decision = aggregator.aggregate("test_market", &signals);
        
        // LLM signal expired, so should fail
        assert!(!decision.should_act);
    }

    #[test]
    fn test_aggregator_strong_consensus() {
        let aggregator = SignalAggregator::with_defaults();
        
        let signals = vec![
            make_signal("llm", SignalType::LlmPrediction, SignalDirection::StrongBuy, dec!(0.90)),
            make_signal("tech", SignalType::Technical, SignalDirection::StrongBuy, dec!(0.85)),
            make_signal("sentiment", SignalType::Sentiment, SignalDirection::Buy, dec!(0.80)),
            make_signal("orderflow", SignalType::OrderFlow, SignalDirection::Buy, dec!(0.75)),
        ];

        let decision = aggregator.aggregate("test_market", &signals);
        
        assert!(decision.should_act);
        assert!(decision.direction == SignalDirection::StrongBuy || decision.direction == SignalDirection::Buy);
        assert!(decision.size_multiplier > dec!(0.5)); // Should be high
    }

    #[test]
    fn test_accuracy_tracking() {
        let mut aggregator = SignalAggregator::with_defaults();
        
        // Record some outcomes
        aggregator.record_outcome(SignalType::LlmPrediction, true);
        aggregator.record_outcome(SignalType::LlmPrediction, true);
        aggregator.record_outcome(SignalType::LlmPrediction, false);
        
        let accuracy = aggregator.get_accuracy(SignalType::LlmPrediction);
        assert!(accuracy.is_some());
        
        // 2 wins, 1 loss = 66.7%
        let acc = accuracy.unwrap();
        assert!(acc > dec!(0.6) && acc < dec!(0.7));
    }

    #[test]
    fn test_size_multiplier() {
        let aggregator = SignalAggregator::with_defaults();
        
        // Strong signal
        let strong_signals = vec![
            make_signal("llm", SignalType::LlmPrediction, SignalDirection::StrongBuy, dec!(0.90)),
            make_signal("tech", SignalType::Technical, SignalDirection::StrongBuy, dec!(0.85)),
        ];
        let strong_decision = aggregator.aggregate("test", &strong_signals);
        
        // Weak signal
        let weak_signals = vec![
            make_signal("llm", SignalType::LlmPrediction, SignalDirection::Buy, dec!(0.60)),
            make_signal("tech", SignalType::Technical, SignalDirection::Neutral, dec!(0.55)),
        ];
        let weak_decision = aggregator.aggregate("test", &weak_signals);
        
        // Strong signal should have higher size multiplier
        assert!(strong_decision.size_multiplier > weak_decision.size_multiplier);
    }

    #[test]
    fn test_signal_builder() {
        let signal = SignalBuilder::new("test", SignalType::LlmPrediction)
            .direction(SignalDirection::Buy)
            .confidence(dec!(0.75))
            .edge(dec!(0.05))
            .metadata("model", "gpt-4")
            .build();

        assert_eq!(signal.name, "test");
        assert_eq!(signal.direction, SignalDirection::Buy);
        assert_eq!(signal.confidence, dec!(0.75));
        assert_eq!(signal.edge, Some(dec!(0.05)));
        assert_eq!(signal.metadata.get("model"), Some(&"gpt-4".to_string()));
    }
}
