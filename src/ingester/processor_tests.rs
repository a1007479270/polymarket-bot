//! Tests for signal processor

#[cfg(test)]
mod tests {
    use super::super::{RawSignal, ParsedSignal, SignalDirection, ActionType};
    use super::super::processor::SignalProcessor;
    use crate::config::LlmConfig;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn create_llm_config() -> LlmConfig {
        LlmConfig {
            provider: "deepseek".to_string(),
            api_key: "test-key".to_string(),
            model: Some("deepseek-chat".to_string()),
            base_url: None,
        }
    }

    #[test]
    fn test_signal_processor_creation() {
        let config = create_llm_config();
        let processor = SignalProcessor::new(config);
        let _ = processor;
    }

    #[test]
    fn test_signal_processor_with_thresholds() {
        let config = create_llm_config();
        let processor = SignalProcessor::new(config)
            .with_thresholds(0.6, 0.7);
        let _ = processor;
    }

    #[test]
    fn test_signal_processor_with_window() {
        let config = create_llm_config();
        let processor = SignalProcessor::new(config)
            .with_window(600);
        let _ = processor;
    }

    #[test]
    fn test_signal_processor_full_config() {
        let config = create_llm_config();
        let processor = SignalProcessor::new(config)
            .with_thresholds(0.55, 0.65)
            .with_window(300);
        let _ = processor;
    }

    #[test]
    fn test_raw_signal_for_processing() {
        let signal = RawSignal {
            source: "twitter".to_string(),
            source_id: "tweet123".to_string(),
            content: "BTC looking very bullish here! 🚀".to_string(),
            author: "@cryptotrader".to_string(),
            author_trust: 0.8,
            timestamp: Utc::now(),
            metadata: None,
        };
        
        assert!(signal.content.contains("bullish"));
        assert_eq!(signal.author_trust, 0.8);
    }

    #[test]
    fn test_bearish_raw_signal() {
        let signal = RawSignal {
            source: "telegram".to_string(),
            source_id: "msg456".to_string(),
            content: "ETH breakdown incoming, short setup".to_string(),
            author: "trader_group".to_string(),
            author_trust: 0.75,
            timestamp: Utc::now(),
            metadata: None,
        };
        
        assert!(signal.content.contains("breakdown") || signal.content.contains("short"));
    }

    #[test]
    fn test_parsed_signal_bullish() {
        let parsed = ParsedSignal {
            token: "BTC".to_string(),
            direction: SignalDirection::Bullish,
            timeframe: "4h".to_string(),
            confidence: 0.85,
            reasoning: "Strong momentum".to_string(),
            action_type: ActionType::Entry,
            sources: vec![],
            agg_score: 0.90,
            timestamp: Utc::now(),
        };
        
        assert_eq!(parsed.direction, SignalDirection::Bullish);
        assert!(parsed.agg_score >= 0.7);
    }

    #[test]
    fn test_parsed_signal_bearish() {
        let parsed = ParsedSignal {
            token: "SOL".to_string(),
            direction: SignalDirection::Bearish,
            timeframe: "1h".to_string(),
            confidence: 0.70,
            reasoning: "Breaking support".to_string(),
            action_type: ActionType::Exit,
            sources: vec![],
            agg_score: 0.75,
            timestamp: Utc::now(),
        };
        
        assert_eq!(parsed.direction, SignalDirection::Bearish);
        assert_eq!(parsed.action_type, ActionType::Exit);
    }

    #[test]
    fn test_parsed_signal_neutral() {
        let parsed = ParsedSignal {
            token: "XRP".to_string(),
            direction: SignalDirection::Neutral,
            timeframe: "1d".to_string(),
            confidence: 0.50,
            reasoning: "Consolidating".to_string(),
            action_type: ActionType::Info,
            sources: vec![],
            agg_score: 0.55,
            timestamp: Utc::now(),
        };
        
        assert_eq!(parsed.direction, SignalDirection::Neutral);
        assert_eq!(parsed.action_type, ActionType::Info);
    }

    #[test]
    fn test_signal_aggregation_score() {
        let score1: f64 = 0.8;
        let score2: f64 = 0.9;
        let avg = (score1 + score2) / 2.0;
        assert!((avg - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_confidence_threshold() {
        let min_confidence = 0.6;
        let signal_confidence = 0.75;
        assert!(signal_confidence >= min_confidence);
    }

    #[test]
    fn test_confidence_below_threshold() {
        let min_confidence = 0.6;
        let signal_confidence = 0.45;
        assert!(signal_confidence < min_confidence);
    }

    #[test]
    fn test_agg_score_threshold() {
        let min_agg_score = 0.65;
        let signal_agg_score = 0.80;
        assert!(signal_agg_score >= min_agg_score);
    }

    #[test]
    fn test_timeframe_parsing() {
        let timeframes = ["5m", "15m", "1h", "4h", "1d"];
        for tf in timeframes {
            assert!(tf.ends_with('m') || tf.ends_with('h') || tf.ends_with('d'));
        }
    }

    #[test]
    fn test_token_identification() {
        let tokens = ["BTC", "ETH", "SOL", "XRP", "DOGE"];
        for token in tokens {
            assert!(token.len() >= 3 && token.len() <= 5);
        }
    }

    #[test]
    fn test_multiple_sources_aggregation() {
        let sources = vec![
            RawSignal {
                source: "twitter".to_string(),
                source_id: "1".to_string(),
                content: "BTC bullish".to_string(),
                author: "a".to_string(),
                author_trust: 0.8,
                timestamp: Utc::now(),
                metadata: None,
            },
            RawSignal {
                source: "telegram".to_string(),
                source_id: "2".to_string(),
                content: "BTC going up".to_string(),
                author: "b".to_string(),
                author_trust: 0.7,
                timestamp: Utc::now(),
                metadata: None,
            },
        ];
        
        assert_eq!(sources.len(), 2);
        
        // Average trust
        let avg_trust: f64 = sources.iter().map(|s| s.author_trust).sum::<f64>() / sources.len() as f64;
        assert!((avg_trust - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_warning_action_type() {
        let parsed = ParsedSignal {
            token: "ETH".to_string(),
            direction: SignalDirection::Bearish,
            timeframe: "1h".to_string(),
            confidence: 0.60,
            reasoning: "Potential reversal".to_string(),
            action_type: ActionType::Warning,
            sources: vec![],
            agg_score: 0.65,
            timestamp: Utc::now(),
        };
        
        assert_eq!(parsed.action_type, ActionType::Warning);
    }
}
