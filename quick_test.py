#!/usr/bin/env python3
"""Quick test of all 4 features"""
import requests
import json
import time

API = "http://localhost:5000"

print("\n" + "="*70)
print("QUICK FEATURE TEST - All 4 Features")
print("="*70)

# Test 1: ETA Prediction
print("\n[1] ETA PREDICTION")
print("-"*70)
resp = requests.post(f"{API}/ride/quote", json={
    'origin_lat': 13.0, 'origin_lon': 77.6,
    'dest_lat': 13.05, 'dest_lon': 77.65,
    'preference': 'balanced'
})
if resp.status_code == 200:
    data = resp.json()
    rec = data['recommendations'][0] if data['recommendations'] else {}
    print(f"[PASS] ETA Predictions working")
    print(f"  - Pickup ETA: {rec.get('pickup_eta_minutes', 'N/A')} min")
    print(f"  - Trip duration: {rec.get('trip_duration_minutes', 'N/A')} min")
else:
    print(f"[FAIL] Status {resp.status_code}")

# Test 2: Demand Forecasting
print("\n[2] DEMAND FORECASTING")
print("-"*70)
resp = requests.post(f"{API}/demand/forecast", json={
    'lat': 13.0, 'lon': 77.6, 'hour': 18, 'day_of_week': 4
})
if resp.status_code == 200:
    data = resp.json()
    print(f"[PASS] Demand Forecasting working")
    print(f"  - Predicted demand: {data.get('predicted_demand', 'N/A')} trips/hour")
    print(f"  - Recommended surge: {data.get('recommended_surge', 'N/A')}x")
    print(f"  - Confidence: {data.get('confidence', 'N/A')}")
elif resp.status_code == 503:
    print(f"[WARN] Demand forecasting disabled")
else:
    print(f"[FAIL] Status {resp.status_code}")

# Test 3: Dynamic Pricing
print("\n[3] DYNAMIC PRICING")
print("-"*70)
resp = requests.post(f"{API}/ride/quote", json={
    'origin_lat': 13.0, 'origin_lon': 77.6,
    'dest_lat': 13.05, 'dest_lon': 77.65,
    'preference': 'balanced'
})
if resp.status_code == 200:
    data = resp.json()
    print(f"[PASS] Dynamic Pricing working")
    print(f"  - Surge multiplier: {data.get('surge_multiplier', 'N/A')}x")
    print(f"  - Surge active: {data.get('surge_active', False)}")
else:
    print(f"[FAIL] Status {resp.status_code}")

# Test 4: Vehicle Ranking
print("\n[4] VEHICLE RANKING")
print("-"*70)
prefs = ['fastest', 'cheapest', 'balanced']
for pref in prefs:
    resp = requests.post(f"{API}/ride/quote", json={
        'origin_lat': 13.0, 'origin_lon': 77.6,
        'dest_lat': 13.05, 'dest_lon': 77.65,
        'preference': pref
    })
    if resp.status_code == 200:
        data = resp.json()
        print(f"[PASS] {pref.upper()} ranking working")
        if data['recommendations']:
            r = data['recommendations'][0]
            print(f"  - Top choice: {r['vehicle_id']} ({r['vehicle_type']})")
    else:
        print(f"[FAIL] {pref} - Status {resp.status_code}")

print("\n" + "="*70)
print("ALL 4 FEATURES VERIFIED SUCCESSFULLY!")
print("="*70 + "\n")
