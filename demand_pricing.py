#!/usr/bin/env python3
"""
Demand-Driven Dynamic Pricing System
Adjusts surge multipliers based on predicted demand vs. available supply
"""

import logging
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('demand_pricing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemandPricingEngine:
    """Dynamic pricing based on demand forecasts"""
    
    def __init__(self):
        """Load demand model and configuration"""
        try:
            self.demand_model = joblib.load('demand_model.joblib')
            self.demand_scaler = joblib.load('demand_scaler.joblib')
            with open('demand_features.json', 'r') as f:
                features_data = json.load(f)
                # Handle both list and dict formats
                self.demand_features = features_data if isinstance(features_data, list) else features_data.get('features', [])
            logger.info("[OK] Demand pricing engine initialized")
        except Exception as e:
            logger.error(f"Failed to load demand model: {e}")
            raise
    
    def predict_demand(self, grid_id, hour, day_of_week, trip_distance=12.0, surge_multiplier=1.0):
        """
        Predict demand for a given grid cell and time
        
        Args:
            grid_id: Grid cell identifier (e.g., "1285_7745")
            hour: Hour of day (0-23)
            day_of_week: Day of week (0-6, Monday-Sunday)
            trip_distance: Average trip distance in km
            surge_multiplier: Current surge multiplier for context
        
        Returns:
            dict with predicted_demand, confidence, recommended_surge
        """
        try:
            # Feature engineering
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            is_weekend = 1 if day_of_week >= 5 else 0
            is_rush_hour = 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
            
            # Default lag values (would be populated from historical data)
            demand_lag_1h = 8.0  # Average demand
            demand_lag_24h = 8.5
            
            # Prepare feature vector
            X = pd.DataFrame([{
                'hour': hour,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,
                'demand_lag_1h': demand_lag_1h,
                'demand_lag_24h': demand_lag_24h,
                'trip_distance': trip_distance,
                'surge_multiplier': surge_multiplier
            }])
            
            # Scale features
            X_scaled = self.demand_scaler.transform(X)
            
            # Predict demand
            predicted_demand = self.demand_model.predict(X_scaled)[0]
            predicted_demand = max(0, predicted_demand)  # Ensure non-negative
            
            # Calculate recommended surge multiplier based on demand
            # Use demand percentile to determine surge
            base_demand = 8.0  # Average baseline demand
            demand_ratio = predicted_demand / base_demand if base_demand > 0 else 1.0
            
            # Map demand ratio to surge multiplier
            if demand_ratio > 1.5:  # Very high demand
                recommended_surge = 1.8
                confidence = "Very High"
            elif demand_ratio > 1.2:  # High demand
                recommended_surge = 1.5
                confidence = "High"
            elif demand_ratio > 0.8:  # Normal demand
                recommended_surge = 1.0 + (demand_ratio - 0.8) * 0.625  # Linear interpolation
                confidence = "Medium"
            else:  # Low demand
                recommended_surge = 0.9
                confidence = "Low"
            
            # Ensure surge bounds
            recommended_surge = max(0.8, min(1.8, recommended_surge))
            
            return {
                'grid_id': grid_id,
                'predicted_demand': round(predicted_demand, 2),
                'demand_ratio': round(demand_ratio, 3),
                'recommended_surge': round(recommended_surge, 2),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error predicting demand: {e}")
            return {
                'grid_id': grid_id,
                'error': str(e),
                'recommended_surge': 1.0  # Default surge on error
            }
    
    def calculate_dynamic_surge(self, current_hour, day_of_week, available_vehicles, 
                               total_demand_estimate, grid_id='default'):
        """
        Calculate final surge multiplier combining time-based and demand-based factors
        
        Args:
            current_hour: Current hour (0-23)
            day_of_week: Current day of week (0-6)
            available_vehicles: Number of available vehicles in area
            total_demand_estimate: Estimated total demand
            grid_id: Grid cell for demand prediction
        
        Returns:
            dict with final surge multiplier and breakdown
        """
        try:
            # Time-based surge (existing logic)
            is_rush_hour = (7 <= current_hour <= 10) or (17 <= current_hour <= 20)
            is_weekend_night = day_of_week >= 5 and current_hour >= 22
            
            time_surge = 1.3 if is_rush_hour else 1.0
            time_surge = max(time_surge, 1.2) if is_weekend_night else time_surge
            
            # Demand-based surge
            demand_forecast = self.predict_demand(
                grid_id, current_hour, day_of_week,
                trip_distance=12.0, surge_multiplier=time_surge
            )
            demand_surge = demand_forecast.get('recommended_surge', 1.0)
            
            # Supply-demand ratio
            supply_ratio = available_vehicles / max(total_demand_estimate, 1)
            
            if supply_ratio < 0.3:  # Very few vehicles for demand
                supply_surge = 1.6
            elif supply_ratio < 0.5:
                supply_surge = 1.4
            elif supply_ratio < 0.8:
                supply_surge = 1.2
            else:
                supply_surge = 1.0
            
            # Combine surges: demand-based is primary, time & supply as modifiers
            final_surge = demand_surge * (supply_surge ** 0.3)  # Supply has smaller weight
            final_surge = max(0.8, min(1.8, final_surge))  # Bound between 0.8 and 1.8
            
            return {
                'final_surge': round(final_surge, 2),
                'demand_surge': round(demand_surge, 2),
                'time_surge': round(time_surge, 2),
                'supply_surge': round(supply_surge, 2),
                'supply_ratio': round(supply_ratio, 3),
                'breakdown': f"Demand({demand_surge:.2f}) × Supply({supply_surge:.2f})^0.3 = {final_surge:.2f}",
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error calculating dynamic surge: {e}")
            return {
                'final_surge': 1.0,
                'error': str(e)
            }


# Example usage for testing
if __name__ == "__main__":
    logger.info("Initializing Demand Pricing Engine...")
    engine = DemandPricingEngine()
    
    # Test predictions
    test_cases = [
        ("1285_7745", 8, 1, "Morning rush hour"),   # Monday 8am
        ("1286_7746", 18, 4, "Evening rush hour"),  # Friday 6pm
        ("1287_7747", 14, 5, "Saturday afternoon"), # Saturday 2pm
        ("1288_7748", 23, 6, "Sunday night"),       # Sunday 11pm
    ]
    
    logger.info("\nTesting demand predictions:")
    for grid_id, hour, day, label in test_cases:
        result = engine.predict_demand(grid_id, hour, day)
        logger.info(f"\n{label} ({grid_id} @ {hour}:00, day={day}):")
        logger.info(f"  Predicted demand: {result['predicted_demand']} trips/hour")
        logger.info(f"  Recommended surge: {result['recommended_surge']}x")
        logger.info(f"  Confidence: {result['confidence']}")
    
    # Test dynamic pricing
    logger.info("\n\nTesting dynamic pricing calculations:")
    pricing_results = engine.calculate_dynamic_surge(
        current_hour=18,
        day_of_week=4,  # Friday
        available_vehicles=5,
        total_demand_estimate=12,
        grid_id="1286_7746"
    )
    logger.info(f"\nDynamic Surge Calculation (Friday 6pm):")
    logger.info(f"  Final surge: {pricing_results['final_surge']}x")
    logger.info(f"  Breakdown: {pricing_results['breakdown']}")
    logger.info(f"  Supply ratio: {pricing_results['supply_ratio']}")
