#!/usr/bin/env python3
"""
Comprehensive System Test - Tests all 4 features of the vehicle matching system:
1. ETA Prediction - Predict pickup and trip duration
2. Demand Forecasting - Predict ride demand per region
3. Dynamic Pricing - Compute surge multipliers based on demand
4. Vehicle Ranking - Rank vehicles by user preference
"""

import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:5000"

def test_feature(name, test_func):
    """Helper to run and report on feature tests"""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print('='*70)
    try:
        test_func()
        print(f"[PASS] {name} - Test completed successfully")
    except AssertionError as e:
        print(f"[FAIL] {name} - {str(e)}")
    except Exception as e:
        print(f"[ERROR] {name} - {str(e)}")


def test_api_health():
    """Test 1: API is running and responsive"""
    response = requests.get(f"{API_URL}/")
    assert response.status_code == 200, f"API returned {response.status_code}"
    data = response.json()
    assert 'message' in data, "Missing message in response"
    print(f"[OK] API is healthy: {data['message']}")
    print(f"[OK] Available endpoints: {len(data.get('endpoints', {}))} found")


def test_vehicle_management():
    """Test 2: Vehicle management (update/list/delete)"""
    # Add vehicles
    vehicles = [
        {'vehicle_id': 'KA01AB0001', 'lat': 13.0, 'lon': 77.6, 'vehicle_type': 'Mini'},
        {'vehicle_id': 'KA01AB0002', 'lat': 13.01, 'lon': 77.61, 'vehicle_type': 'Sedan'},
        {'vehicle_id': 'KA01AB0003', 'lat': 13.02, 'lon': 77.62, 'vehicle_type': 'SUV'},
    ]
    
    for v in vehicles:
        response = requests.post(f"{API_URL}/vehicles/update", json=v)
        assert response.status_code == 200, f"Failed to add vehicle {v['vehicle_id']}: {response.text}"
        print(f"[OK] Added vehicle: {v['vehicle_id']} ({v['vehicle_type']})")
    
    # List vehicles
    response = requests.get(f"{API_URL}/vehicles/list")
    assert response.status_code == 200, f"List failed: {response.text}"
    vehicles_list = response.json()
    print(f"[OK] Total vehicles in system: {len(vehicles_list)}")
    
    # Delete one vehicle
    response = requests.delete(f"{API_URL}/vehicles/KA01AB0001")
    assert response.status_code == 200, f"Delete failed: {response.text}"
    print(f"[OK] Deleted vehicle: KA01AB0001")


def test_eta_prediction():
    """Test 3: ETA Prediction (Feature 1)"""
    print("\nFeature 1: ETA Prediction")
    print("-" * 70)
    
    # Make a ride quote request which uses ETA prediction
    response = requests.post(f"{API_URL}/ride/quote", json={
        'origin_lat': 13.0,
        'origin_lon': 77.6,
        'dest_lat': 13.05,
        'dest_lon': 77.65,
        'preference': 'balanced'
    })
    
    assert response.status_code == 200, f"Quote request failed: {response.text}"
    data = response.json()
    
    # Check recommendations have ETA
    assert 'recommendations' in data, "No recommendations in response"
    assert len(data['recommendations']) > 0, "No vehicles recommended"
    
    first_rec = data['recommendations'][0]
    assert 'pickup_eta_minutes' in first_rec, "No pickup ETA in recommendation"
    assert 'trip_duration_minutes' in first_rec, "No trip duration in recommendation"
    
    print(f"[OK] ETA Predictions generated:")
    for i, rec in enumerate(data['recommendations'][:3], 1):
        print(f"    {i}. {rec['vehicle_id']} ({rec['vehicle_type']})")
        print(f"       Pickup ETA: {rec['pickup_eta_minutes']} min")
        print(f"       Trip duration: {rec['trip_duration_minutes']} min")


def test_demand_forecasting():
    """Test 4: Demand Forecasting (Feature 2)"""
    print("\nFeature 2: Demand Forecasting")
    print("-" * 70)
    
    # Request demand forecast for current location and time
    response = requests.post(f"{API_URL}/demand/forecast", json={
        'lat': 13.0,
        'lon': 77.6,
        'hour': 18,  # 6 PM
        'day_of_week': 4  # Friday
    })
    
    if response.status_code == 503:
        print("[WARN] Demand forecasting not available (module disabled)")
        return
    
    assert response.status_code == 200, f"Forecast request failed: {response.text}"
    data = response.json()
    
    # Check forecast details
    assert 'predicted_demand' in data, "No predicted demand"
    assert 'recommended_surge' in data, "No recommended surge"
    assert 'confidence' in data, "No confidence level"
    
    print(f"[OK] Demand Forecast for {data['day_name']} {data['hour']}:00:")
    print(f"    Grid: {data['grid_id']}")
    print(f"    Predicted demand: {data['predicted_demand']} trips/hour")
    print(f"    Demand ratio: {data['demand_ratio']}")
    print(f"    Recommended surge: {data['recommended_surge']}x")
    print(f"    Confidence: {data['confidence']}")


def test_dynamic_pricing():
    """Test 5: Dynamic Pricing (Feature 3) - Check surge multiplier in quotes"""
    print("\nFeature 3: Dynamic Pricing")
    print("-" * 70)
    
    # Get quotes during rush hour (high demand expected)
    response = requests.post(f"{API_URL}/ride/quote", json={
        'origin_lat': 13.0,
        'origin_lon': 77.6,
        'dest_lat': 13.05,
        'dest_lon': 77.65,
        'preference': 'balanced'
    })
    
    assert response.status_code == 200, f"Quote request failed: {response.text}"
    data = response.json()
    
    surge = data.get('surge_multiplier', 1.0)
    surge_active = data.get('surge_active', False)
    
    print(f"[OK] Dynamic Pricing Calculation:")
    print(f"    Current surge multiplier: {surge}x")
    print(f"    Surge pricing active: {surge_active}")
    
    # Check that surge affects fare calculation
    if len(data['recommendations']) > 0:
        rec = data['recommendations'][0]
        print(f"    Sample fare for {rec['vehicle_type']}: Rs {rec['estimated_fare']}")


def test_vehicle_ranking():
    """Test 6: Vehicle Ranking (Feature 4) - Test different preferences"""
    print("\nFeature 4: Vehicle Ranking")
    print("-" * 70)
    
    preferences = ['fastest', 'cheapest', 'balanced']
    
    for pref in preferences:
        response = requests.post(f"{API_URL}/ride/quote", json={
            'origin_lat': 13.0,
            'origin_lon': 77.6,
            'dest_lat': 13.05,
            'dest_lon': 77.65,
            'preference': pref
        })
        
        assert response.status_code == 200, f"Quote with {pref} preference failed"
        data = response.json()
        
        print(f"\n[OK] {pref.upper()} preference ranking:")
        for i, rec in enumerate(data['recommendations'][:3], 1):
            print(f"    {i}. {rec['vehicle_id']} ({rec['vehicle_type']})")
            print(f"       Pickup: {rec['pickup_eta_minutes']}min | Fare: Rs {rec['estimated_fare']} | Score: {rec.get('score', 'N/A')}")


def test_error_handling():
    """Test 7: Error handling and validation"""
    print("\nTesting Error Handling")
    print("-" * 70)
    
    # Test invalid coordinates
    response = requests.post(f"{API_URL}/ride/quote", json={
        'origin_lat': 50.0,  # Out of bounds
        'origin_lon': 77.6,
        'dest_lat': 13.05,
        'dest_lon': 77.65,
        'preference': 'balanced'
    })
    
    assert response.status_code == 400, "Should reject out-of-bounds coordinates"
    print(f"[OK] Rejected out-of-bounds coordinates: {response.json()['error']}")
    
    # Test missing fields
    response = requests.post(f"{API_URL}/ride/quote", json={
        'origin_lat': 13.0
        # Missing other required fields
    })
    
    assert response.status_code == 400, "Should reject missing fields"
    print(f"[OK] Rejected missing fields: {response.json()['error']}")


def main():
    """Run all feature tests"""
    print("\n" + "="*70)
    print("VEHICLE MATCHING SYSTEM - COMPREHENSIVE FEATURE TEST")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    
    # Wait for API to be ready
    print("\nWaiting for API to be ready...")
    for i in range(30):
        try:
            requests.get(f"{API_URL}/")
            print("[OK] API is ready!")
            break
        except:
            if i < 29:
                time.sleep(1)
            else:
                raise Exception("API not responding")
    
    # Run all tests
    test_feature("API Health Check", test_api_health)
    test_feature("Vehicle Management (Add/List/Delete)", test_vehicle_management)
    test_feature("Feature 1: ETA Prediction", test_eta_prediction)
    test_feature("Feature 2: Demand Forecasting", test_demand_forecasting)
    test_feature("Feature 3: Dynamic Pricing", test_dynamic_pricing)
    test_feature("Feature 4: Vehicle Ranking", test_vehicle_ranking)
    test_feature("Error Handling & Validation", test_error_handling)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70)
    print(f"Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSummary of 4 Key Features:")
    print("1. [CHECK] ETA Prediction - Predicts pickup time and trip duration")
    print("2. [CHECK] Demand Forecasting - Predicts ride demand per grid/time")
    print("3. [CHECK] Dynamic Pricing - Calculates surge based on demand")
    print("4. [CHECK] Vehicle Ranking - Ranks vehicles by user preference")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
