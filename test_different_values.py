"""
HOW TO TEST WITH DIFFERENT VALUES
===================================
This file demonstrates how to test your vehicle matching system with different 
input values, test cases, and scenarios.
"""

import pandas as pd
import numpy as np
import joblib
import json

print("=" * 80)
print("TESTING WITH DIFFERENT VALUES - COMPLETE GUIDE")
print("=" * 80)

# Load models once (reused for all tests)
eta_model = joblib.load('eta_model.joblib')
demand_model = joblib.load('demand_model.joblib')

# ============================================================================
# METHOD 1: TEST WITH DIFFERENT SINGLE VALUES
# ============================================================================
print("\n\n[METHOD 1] Testing ETA Model with Different Single Values")
print("-" * 80)

# Define test cases with different input values
test_cases_eta = [
    {
        'name': 'Short trip (2km, morning)',
        'distance': 2.0,
        'distance_squared': 4.0,
        'hour': 8,
        'hour_sin': np.sin(2 * np.pi * 8 / 24),
        'hour_cos': np.cos(2 * np.pi * 8 / 24),
        'is_rush_hour': 1,
        'is_weekend': 0,
        'surge_multiplier': 1.5,
    },
    {
        'name': 'Medium trip (10km, afternoon)',
        'distance': 10.0,
        'distance_squared': 100.0,
        'hour': 14,
        'hour_sin': np.sin(2 * np.pi * 14 / 24),
        'hour_cos': np.cos(2 * np.pi * 14 / 24),
        'is_rush_hour': 0,
        'is_weekend': 0,
        'surge_multiplier': 1.0,
    },
    {
        'name': 'Long trip (25km, night)',
        'distance': 25.0,
        'distance_squared': 625.0,
        'hour': 22,
        'hour_sin': np.sin(2 * np.pi * 22 / 24),
        'hour_cos': np.cos(2 * np.pi * 22 / 24),
        'is_rush_hour': 0,
        'is_weekend': 0,
        'surge_multiplier': 1.2,
    },
    {
        'name': 'Weekend night (8km, 11pm, weekend)',
        'distance': 8.0,
        'distance_squared': 64.0,
        'hour': 23,
        'hour_sin': np.sin(2 * np.pi * 23 / 24),
        'hour_cos': np.cos(2 * np.pi * 23 / 24),
        'is_rush_hour': 0,
        'is_weekend': 1,
        'surge_multiplier': 1.8,
    },
]

print("\nTesting ETA model with different scenarios:")
print(f"{'Scenario':<30} {'Distance':<12} {'Hour':<8} {'Prediction':<15}")
print("-" * 65)

for test_case in test_cases_eta:
    name = test_case.pop('name')
    
    # Create input array in correct feature order
    X = np.array([[
        test_case['distance'],
        test_case['distance_squared'],
        test_case['hour'],
        test_case['hour_sin'],
        test_case['hour_cos'],
        test_case['is_rush_hour'],
        test_case['is_weekend'],
        test_case['surge_multiplier']
    ]])
    
    pred = eta_model.predict(X)[0]
    print(f"{name:<30} {test_case['distance']:<12.1f} {test_case['hour']:<8} {pred:<15.1f} minutes")


# ============================================================================
# METHOD 2: TEST WITH MULTIPLE VALUES (LOOPS)
# ============================================================================
print("\n\n[METHOD 2] Testing with Multiple Values (Loop Through Options)")
print("-" * 80)

print("\nTesting ETA prediction at different hours of the day:")
print(f"{'Hour':<8} {'Time Period':<20} {'ETA (minutes)':<15}")
print("-" * 43)

hours_to_test = [0, 6, 8, 12, 18, 22]
time_names = {0: 'Midnight', 6: 'Early Morning', 8: 'Morning Rush', 
              12: 'Noon', 18: 'Evening Rush', 22: 'Late Night'}

for hour in hours_to_test:
    X = np.array([[
        10.0,  # 10km trip
        100.0,  # distance squared
        hour,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        1 if 7 <= hour <= 9 or 18 <= hour <= 20 else 0,  # rush hour
        0,  # weekday
        1.0  # baseline surge
    ]])
    
    pred = eta_model.predict(X)[0]
    print(f"{hour:<8} {time_names[hour]:<20} {pred:<15.1f}")


# ============================================================================
# METHOD 3: TEST WITH DIFFERENT RANGES
# ============================================================================
print("\n\n[METHOD 3] Testing Across Value Ranges")
print("-" * 80)

print("\nHow ETA changes with distance (fixed 10am, weekday):")
print(f"{'Distance (km)':<15} {'ETA (minutes)':<15} {'Speed (km/h)':<15}")
print("-" * 45)

distances = [1, 2, 5, 10, 15, 20, 30, 40]

for dist in distances:
    X = np.array([[
        dist,
        dist ** 2,
        10,  # 10am
        np.sin(2 * np.pi * 10 / 24),
        np.cos(2 * np.pi * 10 / 24),
        0,  # not rush hour
        0,  # weekday
        1.0  # baseline surge
    ]])
    
    eta = eta_model.predict(X)[0]
    speed = (dist / eta) * 60 if eta > 0 else 0
    print(f"{dist:<15} {eta:<15.1f} {speed:<15.1f}")


# ============================================================================
# METHOD 4: TEST PRICING WITH DIFFERENT VALUES
# ============================================================================
print("\n\n[METHOD 4] Testing Pricing with Different Parameters")
print("-" * 80)

# Define pricing parameters for different vehicle types
vehicle_pricing = {
    'Auto': {'base_fare': 30, 'distance_rate': 12, 'time_rate': 1},
    'Mini': {'base_fare': 50, 'distance_rate': 10, 'time_rate': 1.5},
    'Sedan': {'base_fare': 80, 'distance_rate': 15, 'time_rate': 2},
    'SUV': {'base_fare': 120, 'distance_rate': 20, 'time_rate': 2.5},
}

# Test different trip scenarios
trip_scenarios = [
    {'distance': 2, 'duration': 10, 'surge': 1.0, 'name': 'Short low-demand'},
    {'distance': 5, 'duration': 15, 'surge': 1.3, 'name': 'Medium normal'},
    {'distance': 10, 'duration': 25, 'surge': 1.5, 'name': 'Long rush hour'},
    {'distance': 15, 'duration': 35, 'surge': 1.8, 'name': 'Very long peak'},
]

for scenario in trip_scenarios:
    print(f"\n{scenario['name'].upper()}: {scenario['distance']}km, "
          f"{scenario['duration']}min, {scenario['surge']:.1f}x surge")
    print(f"{'Vehicle Type':<12} {'Base Fare':<12} {'Distance':<12} {'Time':<12} {'Final Fare':<12}")
    print("-" * 60)
    
    for vehicle_type, rates in vehicle_pricing.items():
        base = rates['base_fare']
        distance_charge = scenario['distance'] * rates['distance_rate']
        time_charge = scenario['duration'] * rates['time_rate']
        subtotal = base + distance_charge + time_charge
        final_fare = subtotal * scenario['surge']
        
        print(f"{vehicle_type:<12} Rs {base:<11.0f} Rs {distance_charge:<11.0f} "
              f"Rs {time_charge:<11.0f} Rs {final_fare:<11.2f}")


# ============================================================================
# METHOD 5: TEST WITH DATA FROM CSV
# ============================================================================
print("\n\n[METHOD 5] Testing With Real Data from CSV")
print("-" * 80)

df = pd.read_csv('trip_data.csv')

# Test with first 5 records
print("\nETA predictions for first 5 trips in dataset:")
print(f"{'Trip ID':<10} {'Distance':<12} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
print("-" * 56)

# Create features
df_test = df.copy()
df_test['distance_squared'] = df_test['trip_distance'] ** 2
df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)
df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)
df_test['is_rush_hour'] = ((df_test['hour'] >= 7) & (df_test['hour'] <= 9) | 
                           (df_test['hour'] >= 18) & (df_test['hour'] <= 20)).astype(int)

with open('model_features.json') as f:
    features_list = json.load(f)

X_test = df_test[features_list].head(5)
preds = eta_model.predict(X_test)

for idx in range(5):
    actual = df.iloc[idx]['trip_duration']
    predicted = preds[idx]
    error = abs(actual - predicted)
    distance = df.iloc[idx]['trip_distance']
    
    print(f"{df.iloc[idx]['trip_id']:<10} {distance:<12.1f} "
          f"{actual:<12.1f} {predicted:<12.1f} {error:<10.2f}")


# ============================================================================
# METHOD 6: TEST WITH EDGE CASES
# ============================================================================
print("\n\n[METHOD 6] Testing Edge Cases (Boundary Values)")
print("-" * 80)

edge_cases = [
    {'name': 'Minimum distance (0.5km)', 'distance': 0.5},
    {'name': 'Maximum distance (50km)', 'distance': 50.0},
    {'name': 'Very short trip (0.1km)', 'distance': 0.1},
    {'name': 'Very long trip (80km)', 'distance': 80.0},
]

print("\nEdge case predictions:")
print(f"{'Test Case':<30} {'Distance':<12} {'ETA (min)':<12}")
print("-" * 54)

for edge_case in edge_cases:
    dist = edge_case['distance']
    X = np.array([[
        dist,
        dist ** 2,
        12,
        np.sin(2 * np.pi * 12 / 24),
        np.cos(2 * np.pi * 12 / 24),
        0,
        0,
        1.0
    ]])
    
    try:
        pred = eta_model.predict(X)[0]
        print(f"{edge_case['name']:<30} {dist:<12.1f} {pred:<12.1f}")
    except Exception as e:
        print(f"{edge_case['name']:<30} {dist:<12.1f} ERROR: {str(e)[:20]}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY - WAYS TO TEST WITH DIFFERENT VALUES")
print("=" * 80)
print("""
1. SINGLE TEST CASES: Define specific scenarios in a list of dictionaries
   - Each dict contains all parameters for one test
   - Loop through and test each scenario
   
2. LOOP THROUGH OPTIONS: Test multiple values of one parameter
   - hours_to_test = [0, 6, 12, 18, 24]
   - distances = [1, 5, 10, 20, 50]
   - Loop and predict for each
   
3. RANGE TESTING: See how predictions change across a range
   - Test distances from 1 to 50 km
   - Observe how ETA changes with distance
   
4. PRICING SCENARIOS: Test different trip types
   - Different distances, durations, surge multipliers
   - Different vehicle types
   
5. REAL DATA: Use CSV data for realistic scenarios
   - Pull actual trips from trip_data.csv
   - Test predictions against actual values
   
6. EDGE CASES: Test boundary conditions
   - Minimum/maximum values
   - Unusual combinations
   
CUSTOMIZATION TIPS:
- Modify test_cases_eta list to add more scenarios
- Change hours_to_test to test different times
- Modify distances list to test different ranges
- Add more vehicle types to pricing tests
- Filter CSV data for specific conditions
""")
print("=" * 80)
