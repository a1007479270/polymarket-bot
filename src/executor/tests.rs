//! Tests for executor module

#[cfg(test)]
mod tests {
    use crate::config::RiskConfig;
    use crate::types::{Order, OrderType, Side, Signal};
    use chrono::Utc;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    #[test]
    fn test_risk_config_default() {
        let config = RiskConfig::default();
        assert_eq!(config.max_position_pct, dec!(0.05));
        assert_eq!(config.max_exposure_pct, dec!(0.50));
        assert_eq!(config.max_open_positions, 10);
    }

    #[test]
    fn test_order_creation() {
        let order = Order {
            token_id: "token1".to_string(),
            side: Side::Buy,
            price: dec!(0.55),
            size: dec!(100),
            order_type: OrderType::GTC,
        };
        
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.price, dec!(0.55));
    }

    #[test]
    fn test_order_sell() {
        let order = Order {
            token_id: "token1".to_string(),
            side: Side::Sell,
            price: dec!(0.65),
            size: dec!(50),
            order_type: OrderType::FOK,
        };
        
        assert_eq!(order.side, Side::Sell);
        assert_eq!(order.order_type, OrderType::FOK);
    }

    #[test]
    fn test_signal_size_calculation() {
        let signal = Signal {
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.70),
            market_probability: dec!(0.55),
            edge: dec!(0.15),
            confidence: dec!(0.80),
            suggested_size: dec!(0.05), // 5% of portfolio
            timestamp: Utc::now(),
        };
        
        let portfolio_value = dec!(1000);
        let size_usd = signal.suggested_size * portfolio_value;
        assert_eq!(size_usd, dec!(50)); // $50
        
        let size_shares = size_usd / signal.market_probability;
        assert!(size_shares > dec!(90)); // ~90.9 shares
    }

    #[test]
    fn test_position_sizing_max() {
        let config = RiskConfig::default();
        let portfolio = dec!(10000);
        let max_position = config.max_position_pct * portfolio;
        assert_eq!(max_position, dec!(500)); // 5% of 10k
    }

    #[test]
    fn test_exposure_limit() {
        let config = RiskConfig::default();
        let portfolio = dec!(10000);
        let max_exposure = config.max_exposure_pct * portfolio;
        assert_eq!(max_exposure, dec!(5000)); // 50% of 10k
    }

    #[test]
    fn test_daily_loss_limit() {
        let config = RiskConfig::default();
        let portfolio = dec!(10000);
        let max_loss = config.max_daily_loss_pct * portfolio;
        assert_eq!(max_loss, dec!(1000)); // 10% of 10k
    }

    #[test]
    fn test_min_balance_reserve() {
        let config = RiskConfig::default();
        assert_eq!(config.min_balance_reserve, dec!(100));
    }

    #[test]
    fn test_position_check_small() {
        let config = RiskConfig::default();
        let portfolio = dec!(1000);
        let position_size = dec!(40); // 4%
        let pct = position_size / portfolio;
        assert!(pct < config.max_position_pct);
    }

    #[test]
    fn test_position_check_too_large() {
        let config = RiskConfig::default();
        let portfolio = dec!(1000);
        let position_size = dec!(100); // 10%
        let pct = position_size / portfolio;
        assert!(pct > config.max_position_pct); // Exceeds 5% limit
    }

    #[test]
    fn test_multiple_positions_exposure() {
        let config = RiskConfig::default();
        let positions = vec![dec!(100), dec!(150), dec!(200)]; // 3 positions
        let total_exposure: Decimal = positions.iter().sum();
        assert_eq!(total_exposure, dec!(450));
        
        let portfolio = dec!(1000);
        let exposure_pct = total_exposure / portfolio;
        assert!(exposure_pct < config.max_exposure_pct); // 45% < 50%
    }

    #[test]
    fn test_exposure_exceeds_limit() {
        let config = RiskConfig::default();
        let positions = vec![dec!(200), dec!(200), dec!(200)]; // 3 large positions
        let total_exposure: Decimal = positions.iter().sum();
        
        let portfolio = dec!(1000);
        let exposure_pct = total_exposure / portfolio;
        assert!(exposure_pct > config.max_exposure_pct); // 60% > 50%
    }

    #[test]
    fn test_order_type_gtd() {
        let order = Order {
            token_id: "t".to_string(),
            side: Side::Buy,
            price: dec!(0.5),
            size: dec!(10),
            order_type: OrderType::GTD,
        };
        assert_eq!(order.order_type, OrderType::GTD);
    }

    #[test]
    fn test_signal_edge_positive() {
        let signal = Signal {
            market_id: "m".to_string(),
            token_id: "t".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.75),
            market_probability: dec!(0.60),
            edge: dec!(0.15),
            confidence: dec!(0.85),
            suggested_size: dec!(0.03),
            timestamp: Utc::now(),
        };
        
        assert!(signal.edge > Decimal::ZERO);
    }

    #[test]
    fn test_signal_edge_negative() {
        let signal = Signal {
            market_id: "m".to_string(),
            token_id: "t".to_string(),
            side: Side::Sell,
            model_probability: dec!(0.25),
            market_probability: dec!(0.40),
            edge: dec!(-0.15),
            confidence: dec!(0.80),
            suggested_size: dec!(0.02),
            timestamp: Utc::now(),
        };
        
        assert!(signal.edge < Decimal::ZERO);
    }

    #[test]
    fn test_max_open_positions() {
        let config = RiskConfig::default();
        assert_eq!(config.max_open_positions, 10);
        
        let current_positions = 8;
        assert!(current_positions < config.max_open_positions);
    }

    #[test]
    fn test_positions_at_limit() {
        let config = RiskConfig::default();
        let current_positions = 10;
        assert!(current_positions >= config.max_open_positions);
    }
}
