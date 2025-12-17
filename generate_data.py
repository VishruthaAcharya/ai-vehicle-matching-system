import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

print("Generating dataset")

LAT_MIN, LAT_MAX = 12.85, 13.15
LON_MIN, LON_MAX = 77.45, 77.75


VEHICLE_TYPES = ['Mini', 'Sedan', 'SUV', 'Auto']
BASE_FARES = {'Mini': 50, 'Sedan': 80, 'SUV': 120, 'Auto': 30}

PER_KM = {'Mini': 10, 'Sedan': 15, 'SUV': 20, 'Auto': 12}
PER_MIN = {'Mini': 1.5, 'Sedan': 2, 'SUV': 2.5, 'Auto': 1}

USER_PREFERENCES = ['cheap', 'fast', 'balanced']

def generate_coordinate():
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)
    return lat, lon

def calculate_distance(lat1, lon1, lat2, lon2):
    lat_diff = (lat2 - lat1) * 111
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    return max(np.sqrt(lat_diff**2 + lon_diff**2), 0.5)

def calculate_duration(distance, hour, is_weekend):
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        speed = 15
    elif 22 <= hour or hour <= 6:
        speed = 35
    else:
        speed = 25

    if is_weekend:
        speed *= 1.2

    speed *= random.uniform(0.8, 1.2)
    return max((distance / speed) * 60, 3)

def calculate_fare(distance, duration, vehicle_type, surge):
    fare = (
        BASE_FARES[vehicle_type]
        + distance * PER_KM[vehicle_type]
        + duration * PER_MIN[vehicle_type]
    ) * surge

    # adding noise 
    fare *= random.uniform(0.95, 1.1)
    return round(fare, 2)

# synthetic vehicles
NUM_VEHICLES = 3000
vehicles = []

for i in range(NUM_VEHICLES):
    lat, lon = generate_coordinate()
    vehicles.append({
        "vehicle_id": f"VEH_{i+1:04d}",
        "vehicle_type": random.choice(VEHICLE_TYPES),
        "vehicle_lat": round(lat, 6),
        "vehicle_lon": round(lon, 6)
    })

vehicles_df = pd.DataFrame(vehicles)


# trips

NUM_TRIPS = 10000
start_date = datetime(2024, 1, 1)
records = []

for i in range(NUM_TRIPS):
    vehicle = vehicles_df.sample(1).iloc[0]

    # Timestamp
    days_offset = random.randint(0, 180)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    timestamp = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
    is_weekend = timestamp.weekday() >= 5

    # Origin & destination
    origin_lat, origin_lon = generate_coordinate()
    dest_lat, dest_lon = generate_coordinate()

    # Distances
    pickup_distance = calculate_distance(
        vehicle.vehicle_lat, vehicle.vehicle_lon,
        origin_lat, origin_lon
    )

    trip_distance = calculate_distance(
        origin_lat, origin_lon,
        dest_lat, dest_lon
    )

    # Durations
    pickup_eta = calculate_duration(pickup_distance, hour, is_weekend)
    trip_duration = calculate_duration(trip_distance, hour, is_weekend)

    # Surge
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        surge = random.uniform(1.2, 2.0)
    elif is_weekend and 20 <= hour <= 23:
        surge = random.uniform(1.1, 1.5)
    else:
        surge = random.uniform(1.0, 1.2)

    # Fare
    fare = calculate_fare(
        trip_distance,
        trip_duration,
        vehicle.vehicle_type,
        surge
    )

    # User preference
    preference = random.choice(USER_PREFERENCES)

    records.append({
        "trip_id": f"TRIP_{i+1:06d}",
        "timestamp": timestamp,
        "hour": hour,
        "day_of_week": timestamp.weekday(),
        "is_weekend": is_weekend,

        "vehicle_id": vehicle.vehicle_id,
        "vehicle_type": vehicle.vehicle_type,
        "vehicle_lat": vehicle.vehicle_lat,
        "vehicle_lon": vehicle.vehicle_lon,

        "origin_lat": round(origin_lat, 6),
        "origin_lon": round(origin_lon, 6),
        "dest_lat": round(dest_lat, 6),
        "dest_lon": round(dest_lon, 6),

        "pickup_distance": round(pickup_distance, 2),
        "pickup_eta": round(pickup_eta, 2),

        "trip_distance": round(trip_distance, 2),
        "trip_duration": round(trip_duration, 2),

        "surge_multiplier": round(surge, 2),
        "fare": fare,

        "user_preference": preference
    })



df = pd.DataFrame(records)


df["lat_bin"] = (df["origin_lat"] * 100).astype(int)
df["lon_bin"] = (df["origin_lon"] * 100).astype(int)
df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

df.to_csv("ai_vehicle_matching_dataset.csv", index=False)

df.to_csv("trip.csv", index=False)

print("✓ Dataset created: ai_vehicle_matching_dataset.csv")
print("✓ Dataset created: trip.csv")
print("✓ Records:", len(df))
print("\nSample:")
print(df.head())
