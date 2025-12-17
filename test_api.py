import requests
import json
import sys

API_URL = "http://127.0.0.1:5000"

print("Testing AI Vehicle Matching API\n")
print("="*60)

try:
    # Test 1: Add some vehicles
    print("\n1. Adding vehicles to the system...")
    vehicles_to_add = [
        {
            'vehicle_id': 'KA01AB1234',
            'lat': 12.9716,
            'lon': 77.5946,
            'vehicle_type': 'Mini',
            'status': 'available'
        },
        {
            'vehicle_id': 'KA02CD5678',
            'lat': 12.9750,
            'lon': 77.6000,
            'vehicle_type': 'Sedan',
            'status': 'available'
        },
        {
            'vehicle_id': 'KA03EF9012',
            'lat': 12.9700,
            'lon': 77.5900,
            'vehicle_type': 'SUV',
            'status': 'available'
        }
    ]

    for vehicle in vehicles_to_add:
        try:
            response = requests.post(f"{API_URL}/vehicles/update", json=vehicle)
            if response.status_code == 200:
                print(f"   [OK] Added {vehicle['vehicle_id']} ({vehicle['vehicle_type']})")
            else:
                print(f"   [ERROR] Failed to add {vehicle['vehicle_id']}: {response.status_code}")
        except Exception as e:
            print(f"   [ERROR] Exception adding vehicle: {e}")

    # Test 2: Request ride quote
    print("\n2. Requesting ride quote...")
    ride_request = {
        'origin_lat': 12.9716,
        'origin_lon': 77.5946,
        'dest_lat': 13.0500,
        'dest_lon': 77.6500,
        'preference': 'balanced'
    }

    response = requests.post(f"{API_URL}/ride/quote", json=ride_request)
    if response.status_code != 200:
        print(f"   [ERROR] API returned {response.status_code}: {response.text}")
    else:
        result = response.json()

        print(f"\n   Trip Details:")
        print(f"   Origin: ({ride_request['origin_lat']}, {ride_request['origin_lon']})")
        print(f"   Destination: ({ride_request['dest_lat']}, {ride_request['dest_lon']})")
        print(f"   Preference: {ride_request['preference']}")

        if result.get('surge_active'):
            print(f"   [ALERT] Surge pricing active!")

        print(f"\n   Top Recommendations:")
        print("   " + "-"*56)

        for i, rec in enumerate(result['recommendations'], 1):
            print(f"\n   {i}. {rec['vehicle_type']} - {rec['vehicle_id']}")
            print(f"      Pickup ETA: {rec['pickup_eta_minutes']} min")
            print(f"      Trip Duration: {rec['trip_duration_minutes']} min")
            print(f"      Distance: {rec['trip_distance_km']} km")
            print(f"      Fare: Rs {rec['estimated_fare']}")
            print(f"      Surge: {rec['surge_multiplier']}x")

    # Test 3: Different preferences
    print("\n\n3. Testing different preferences...")

    for pref in ['fastest', 'cheapest']:
        ride_request['preference'] = pref
        response = requests.post(f"{API_URL}/ride/quote", json=ride_request)
        if response.status_code == 200:
            result = response.json()
            top = result['recommendations'][0]
            print(f"\n   {pref.upper()}: {top['vehicle_type']} - "
                  f"Rs {top['estimated_fare']} - "
                  f"{top['pickup_eta_minutes']} min ETA")
        else:
            print(f"   [ERROR] Failed to get {pref} quote: {response.status_code}")

    print("\n" + "="*60)
    print("[OK] All tests completed!")
    print("="*60)

except requests.exceptions.ConnectionError:
    print("[ERROR] Could not connect to API at http://127.0.0.1:5000")
    print("Make sure the Flask app is running: python app.py")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()