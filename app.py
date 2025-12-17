from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import random
import sqlite3
import logging
import os
from functools import wraps

# Try to import demand pricing engine (optional feature)
try:
    from demand_pricing import DemandPricingEngine
    DEMAND_PRICING_ENABLED = True
    logger_temp = logging.getLogger(__name__)
except ImportError:
    DEMAND_PRICING_ENABLED = False
    logger_temp = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
DB_PATH = 'vehicles.db'
MODEL_PATH = 'eta_model.joblib'
FEATURES_PATH = 'model_features.json'
DEBUG_MODE = os.getenv('FLASK_ENV', 'production') == 'development'

# Coordinate boundaries (Bangalore area)
LAT_MIN, LAT_MAX = 12.8, 13.2
LON_MIN, LON_MAX = 77.4, 77.8

# Vehicle types and pricing
VEHICLE_TYPES = ['Mini', 'Sedan', 'SUV', 'Auto']
BASE_FARES = {'Mini': 50, 'Sedan': 80, 'SUV': 120, 'Auto': 30}
PER_KM = {'Mini': 10, 'Sedan': 15, 'SUV': 20, 'Auto': 12}
PER_MIN = {'Mini': 1.5, 'Sedan': 2, 'SUV': 2.5, 'Auto': 1}

# Load trained model
logger.info('Loading model...')
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_cols = json.load(f)
    logger.info('[OK] Model and features loaded successfully')
except Exception as e:
    logger.error(f'Failed to load model: {e}')
    raise

# Load demand pricing engine if available
demand_pricing_engine = None
if DEMAND_PRICING_ENABLED:
    try:
        demand_pricing_engine = DemandPricingEngine()
        logger.info('[OK] Demand pricing engine initialized')
    except Exception as e:
        logger.warning(f'Demand pricing engine not available: {e}')
        DEMAND_PRICING_ENABLED = False



def init_db():
    """Initialize SQLite database for vehicles"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id TEXT PRIMARY KEY,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                vehicle_type TEXT NOT NULL,
                status TEXT DEFAULT 'available',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f'[OK] Database initialized: {DB_PATH}')
    except Exception as e:
        logger.error(f'Database initialization failed: {e}')
        raise


def validate_coordinates(lat, lon):
    """Validate latitude and longitude are within bounds"""
    return (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX)


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def log_request():
    """Decorator to log incoming requests"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            logger.info(f'{request.method} {request.path} - {request.remote_addr}')
            try:
                result = f(*args, **kwargs)
                logger.info(f'Response: {result[1] if isinstance(result, tuple) else 200}')
                return result
            except Exception as e:
                logger.error(f'Error: {str(e)}')
                raise
        return decorated_function
    return decorator


@app.errorhandler(400)
def bad_request(error):
    logger.warning(f'Bad request: {error}')
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Internal server error: {error}')
    return jsonify({'error': 'Internal server error'}), 500


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points (Haversine approximation)"""
    lat_diff = (lat2 - lat1) * 111
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    return max(distance, 0.5)
    """Calculate distance between two points"""
    lat_diff = (lat2 - lat1) * 111
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    return max(distance, 0.5)

def predict_eta(distance, hour, is_weekend, surge):
    """Predict ETA using trained model"""
    features = pd.DataFrame({
        'trip_distance': [distance],
        'distance_squared': [distance ** 2],
        'hour': [hour],
        'hour_sin': [np.sin(2 * np.pi * hour / 24)],
        'hour_cos': [np.cos(2 * np.pi * hour / 24)],
        'is_rush_hour': [int((7 <= hour <= 10) or (17 <= hour <= 20))],
        'is_weekend': [int(is_weekend)],
        'surge_multiplier': [surge]
    })
    
    eta = model.predict(features[feature_cols])[0]
    return max(eta, 2)  # Minimum 2 minutes

def calculate_fare(distance, duration, vehicle_type, surge):
    """Calculate trip fare"""
    if vehicle_type not in BASE_FARES:
        raise ValueError(f'Invalid vehicle type: {vehicle_type}')
    
    base = BASE_FARES[vehicle_type]
    fare = (base + (distance * PER_KM[vehicle_type]) + 
            (duration * PER_MIN[vehicle_type])) * surge
    return round(fare, 2)

def calculate_surge_multiplier(hour, day_of_week, available_vehicles_count=5, estimated_demand=8, grid_id='default'):
    """Calculate surge based on demand and supply"""
    # If demand pricing engine is available, use it for more accurate pricing
    if DEMAND_PRICING_ENABLED and demand_pricing_engine:
        try:
            result = demand_pricing_engine.calculate_dynamic_surge(
                current_hour=hour,
                day_of_week=day_of_week,
                available_vehicles=available_vehicles_count,
                total_demand_estimate=estimated_demand,
                grid_id=grid_id
            )
            surge = result.get('final_surge', 1.0)
            logger.debug(f"Demand-based surge: {surge}x (breakdown: {result.get('breakdown', '')})")
            return surge
        except Exception as e:
            logger.debug(f"Demand pricing calculation failed, falling back to time-based: {e}")
    
    # Fallback to time-based surge
    is_weekend = day_of_week >= 5
    
    if 7 <= hour <= 10 or 17 <= hour <= 20:  # Rush hours
        return random.uniform(1.3, 1.8)
    elif is_weekend and 20 <= hour <= 23:  # Weekend nights
        return random.uniform(1.2, 1.5)
    else:
        return random.uniform(1.0, 1.2)

@app.route('/')
@log_request()
def home():
    return jsonify({
        'message': 'AI Vehicle Matching API',
        'status': 'running',
        'endpoints': {
            'POST /vehicles/update': 'Update or add vehicle location',
            'GET /vehicles/list': 'List all available vehicles',
            'POST /ride/quote': 'Get ride quotes',
            'DELETE /vehicles/<vehicle_id>': 'Remove a vehicle'
        }
    })

@app.route('/vehicles/update', methods=['POST'])
@log_request()
def update_vehicle():
    """Update or add vehicle location"""
    try:
        data = request.json or {}
        
        required_fields = ['vehicle_id', 'lat', 'lon', 'vehicle_type']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
        
        # Validate vehicle type
        if data['vehicle_type'] not in VEHICLE_TYPES:
            return jsonify({'error': f'Invalid vehicle type. Must be one of: {VEHICLE_TYPES}'}), 400
        
        # Validate coordinates
        if not validate_coordinates(data['lat'], data['lon']):
            return jsonify({'error': f'Coordinates out of bounds. Lat: [{LAT_MIN}, {LAT_MAX}], Lon: [{LON_MIN}, {LON_MAX}]'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        status = data.get('status', 'available')
        
        cursor.execute('''
            INSERT OR REPLACE INTO vehicles 
            (vehicle_id, lat, lon, vehicle_type, status, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (data['vehicle_id'], data['lat'], data['lon'], data['vehicle_type'], status))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated vehicle: {data['vehicle_id']}")
        return jsonify({
            'message': 'Vehicle updated successfully',
            'vehicle': data
        }), 200
        
    except Exception as e:
        logger.error(f'Error updating vehicle: {e}')
        return jsonify({'error': str(e)}), 500



@app.route('/vehicles/list', methods=['GET'])
@log_request()
def list_vehicles():
    """List all vehicles"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM vehicles ORDER BY updated_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        vehicles_list = [dict(row) for row in rows]
        logger.info(f'Listed {len(vehicles_list)} vehicles')
        
        return jsonify({
            'count': len(vehicles_list),
            'vehicles': vehicles_list
        }), 200
        
    except Exception as e:
        logger.error(f'Error listing vehicles: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/vehicles/<vehicle_id>', methods=['DELETE'])
@log_request()
def delete_vehicle(vehicle_id):
    """Delete a vehicle"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM vehicles WHERE vehicle_id = ?', (vehicle_id,))
        conn.commit()
        conn.close()
        
        logger.info(f'Deleted vehicle: {vehicle_id}')
        return jsonify({'message': f'Vehicle {vehicle_id} deleted'}), 200
        
    except Exception as e:
        logger.error(f'Error deleting vehicle: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/ride/quote', methods=['POST'])
@log_request()
def get_quote():
    """Get ride quotes from available vehicles"""
    try:
        data = request.json or {}
        
        required_fields = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
        
        origin_lat = data['origin_lat']
        origin_lon = data['origin_lon']
        dest_lat = data['dest_lat']
        dest_lon = data['dest_lon']
        preference = data.get('preference', 'balanced')
        
        # Validate coordinates
        if not validate_coordinates(origin_lat, origin_lon):
            return jsonify({'error': 'Invalid origin coordinates'}), 400
        if not validate_coordinates(dest_lat, dest_lon):
            return jsonify({'error': 'Invalid destination coordinates'}), 400
        
        # Get current time info
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_weekend = day_of_week >= 5
        
        # Get vehicles from database first
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vehicles WHERE status = 'available'")
        rows = cursor.fetchall()
        conn.close()
        
        vehicles = [dict(row) for row in rows]
        
        # If no vehicles, create some dummy ones
        if len(vehicles) == 0:
            logger.info('No vehicles found, creating dummy vehicles...')
            for i in range(5):
                vehicle_id = f'DEMO_{i+1:03d}'
                lat = origin_lat + random.uniform(-0.02, 0.02)
                lon = origin_lon + random.uniform(-0.02, 0.02)
                
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO vehicles 
                    (vehicle_id, lat, lon, vehicle_type, status)
                    VALUES (?, ?, ?, ?, 'available')
                ''', (vehicle_id, lat, lon, random.choice(VEHICLE_TYPES)))
                conn.commit()
                conn.close()
                
                vehicles.append({
                    'vehicle_id': vehicle_id,
                    'lat': lat,
                    'lon': lon,
                    'vehicle_type': random.choice(VEHICLE_TYPES)
                })
        
        # Create grid ID from origin coordinates for demand prediction
        grid_id = f"{int(origin_lat*100)}_{int(origin_lon*100)}"
        
        # Calculate surge with vehicle count information
        estimated_demand = max(5, len(vehicles) * 1.5)  # Estimate demand as 1.5x available vehicles
        surge = calculate_surge_multiplier(hour, day_of_week, len(vehicles), estimated_demand, grid_id)
        
        # Calculate trip distance
        trip_distance = calculate_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        
        # Calculate metrics for each vehicle
        recommendations = []
        
        for vehicle in vehicles:
            try:
                # Distance to pickup
                pickup_distance = calculate_distance(
                    vehicle['lat'], vehicle['lon'], 
                    origin_lat, origin_lon
                )
                
                # Predict ETAs
                pickup_eta = predict_eta(pickup_distance, hour, is_weekend, 1.0)
                trip_duration = predict_eta(trip_distance, hour, is_weekend, surge)
                
                # Calculate fare
                fare = calculate_fare(trip_distance, trip_duration, 
                                    vehicle['vehicle_type'], surge)
                
                recommendations.append({
                    'vehicle_id': vehicle['vehicle_id'],
                    'vehicle_type': vehicle['vehicle_type'],
                    'pickup_eta_minutes': round(pickup_eta, 1),
                    'trip_duration_minutes': round(trip_duration, 1),
                    'trip_distance_km': round(trip_distance, 2),
                    'estimated_fare': fare,
                    'surge_multiplier': round(surge, 2),
                    'vehicle_lat': vehicle['lat'],
                    'vehicle_lon': vehicle['lon']
                })
            except Exception as e:
                logger.warning(f'Error calculating metrics for {vehicle["vehicle_id"]}: {e}')
                continue
        
        if not recommendations:
            return jsonify({'error': 'No recommendations available'}), 404
        
        # Rank vehicles based on preference
        if preference == 'fastest':
            recommendations.sort(key=lambda x: x['pickup_eta_minutes'])
        elif preference == 'cheapest':
            recommendations.sort(key=lambda x: x['estimated_fare'])
        else:  # balanced
            for rec in recommendations:
                eta_score = rec['pickup_eta_minutes'] / 30
                price_score = rec['estimated_fare'] / 500
                rec['score'] = (eta_score + price_score) / 2
            recommendations.sort(key=lambda x: x['score'])
        
        logger.info(f'Generated {len(recommendations)} quotes for {preference} preference')
        
        return jsonify({
            'preference': preference,
            'surge_active': surge > 1.2,
            'surge_multiplier': round(surge, 2),
            'recommendations': recommendations[:5]
        }), 200
        
    except Exception as e:
        logger.error(f'Error getting quotes: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/demand/forecast', methods=['POST'])
@log_request()
def demand_forecast():
    """Get demand forecast for a specific area and time"""
    if not DEMAND_PRICING_ENABLED or not demand_pricing_engine:
        return jsonify({'error': 'Demand forecasting not available'}), 503
    
    try:
        data = request.json or {}
        
        # Parse parameters
        lat = data.get('lat')
        lon = data.get('lon')
        hour = data.get('hour')  # Optional, defaults to current
        day_of_week = data.get('day_of_week')  # Optional, defaults to current
        
        # Validate coordinates if provided
        if lat is not None and lon is not None:
            if not validate_coordinates(lat, lon):
                return jsonify({'error': 'Coordinates out of bounds'}), 400
            grid_id = f"{int(lat*100)}_{int(lon*100)}"
        else:
            grid_id = data.get('grid_id', 'default')
        
        # Use current time if not specified
        now = datetime.now()
        if hour is None:
            hour = now.hour
        if day_of_week is None:
            day_of_week = now.weekday()
        
        # Validate hour and day_of_week
        if not (0 <= hour <= 23):
            return jsonify({'error': 'Hour must be between 0 and 23'}), 400
        if not (0 <= day_of_week <= 6):
            return jsonify({'error': 'Day of week must be between 0 (Monday) and 6 (Sunday)'}), 400
        
        # Get demand prediction
        demand_result = demand_pricing_engine.predict_demand(
            grid_id=grid_id,
            hour=hour,
            day_of_week=day_of_week,
            trip_distance=12.0,
            surge_multiplier=1.0
        )
        
        if 'error' in demand_result:
            return jsonify({'error': demand_result['error']}), 500
        
        logger.info(f'Forecast for {grid_id} @ {hour}:00 (day {day_of_week}): {demand_result["predicted_demand"]} trips/hr')
        
        return jsonify({
            'grid_id': grid_id,
            'hour': hour,
            'day_of_week': day_of_week,
            'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
            'predicted_demand': demand_result['predicted_demand'],
            'demand_ratio': demand_result['demand_ratio'],
            'recommended_surge': demand_result['recommended_surge'],
            'confidence': demand_result['confidence'],
            'forecast_time': demand_result['timestamp']
        }), 200
    
    except Exception as e:
        logger.error(f'Error getting demand forecast: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    init_db()
    
    logger.info('='*60)
    logger.info('Starting AI Vehicle Matching API')
    logger.info('='*60)
    logger.info(f'Database: {DB_PATH}')
    logger.info(f'Model: {MODEL_PATH}')
    logger.info(f'Debug mode: {DEBUG_MODE}')
    logger.info('Available endpoints:')
    logger.info('  GET  /              - API info')
    logger.info('  POST /vehicles/update - Update/add vehicle')
    logger.info('  GET  /vehicles/list  - List all vehicles')
    logger.info('  DELETE /vehicles/<id> - Remove vehicle')
    logger.info('  POST /ride/quote     - Get ride quotes')
    logger.info('  POST /demand/forecast - Get demand forecast')
    logger.info(f'Demand pricing: {"ENABLED" if DEMAND_PRICING_ENABLED else "DISABLED"}')
    logger.info('='*60)
    
    app.run(debug=DEBUG_MODE, port=5000)