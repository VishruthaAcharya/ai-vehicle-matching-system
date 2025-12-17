"""
Demand Forecasting Module
Predicts ride demand per grid cell and hour using time-series analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
GRID_SIZE = 50  # meters (approximately 1km x 1km grid cells)
MODEL_PATH = 'demand_model.joblib'
SCALER_PATH = 'demand_scaler.joblib'

print("Demand Forecasting Model Training")
print("="*60)

# Load trip data
logger.info('Loading trip data...')
df = pd.read_csv('trip_data.csv')

logger.info(f'Loaded {len(df)} trips')
logger.info('Creating grid-based demand features...')

# Create hourly demand per grid cell
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['date'] = df['timestamp'].dt.date

# Create grid cells (using existing grid_id from data)
# If grid_id doesn't exist, create it
if 'grid_id' not in df.columns:
    df['lat_bin'] = (df['origin_lat'] * 100).astype(int)
    df['lon_bin'] = (df['origin_lon'] * 100).astype(int)
    df['grid_id'] = df['lat_bin'].astype(str) + '_' + df['lon_bin'].astype(str)

# Aggregate demand by grid, hour, and day
demand_data = df.groupby(['grid_id', 'date', 'hour']).agg({
    'trip_id': 'count',  # demand (number of trips)
    'trip_distance': 'mean',
    'surge_multiplier': 'mean',
    'is_weekend': 'first'
}).reset_index()

demand_data.rename(columns={'trip_id': 'demand'}, inplace=True)

logger.info(f'Created {len(demand_data)} grid-hour-day combinations')
logger.info(f'Total unique grids: {demand_data["grid_id"].nunique()}')
logger.info(f'Date range: {demand_data["date"].min()} to {demand_data["date"].max()}')

# Feature engineering for demand prediction
logger.info('Engineering features for RandomForest model...')

demand_data['date'] = pd.to_datetime(demand_data['date'])
demand_data['day_of_week'] = demand_data['date'].dt.dayofweek
demand_data['is_weekend'] = (demand_data['day_of_week'] >= 5).astype(int)

# Simple lag features within each grid
demand_data = demand_data.sort_values(['grid_id', 'date', 'hour'])
demand_data['demand_lag_1h'] = demand_data.groupby('grid_id')['demand'].shift(1).fillna(demand_data['demand'].mean())
demand_data['demand_lag_24h'] = demand_data.groupby('grid_id')['demand'].shift(24).fillna(demand_data['demand'].mean())

# Hour-based features
demand_data['hour_sin'] = np.sin(2 * np.pi * demand_data['hour'] / 24)
demand_data['hour_cos'] = np.cos(2 * np.pi * demand_data['hour'] / 24)
demand_data['is_rush_hour'] = ((demand_data['hour'] >= 7) & (demand_data['hour'] <= 10) |
                               (demand_data['hour'] >= 17) & (demand_data['hour'] <= 20)).astype(int)

# Fill any remaining NaNs (numeric columns only)
numeric_cols = demand_data.select_dtypes(include=[np.number]).columns
demand_data[numeric_cols] = demand_data[numeric_cols].fillna(demand_data[numeric_cols].mean())

logger.info(f'Final dataset size: {len(demand_data)} records')

# Prepare training data
feature_cols = ['hour', 'hour_sin', 'hour_cos', 'is_weekend', 'is_rush_hour',
                'demand_lag_1h', 'demand_lag_24h', 'trip_distance', 'surge_multiplier']

X = demand_data[feature_cols]
y = demand_data['demand']

# Scale features
logger.info('Scaling features...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Train XGBoost model
logger.info('Training RandomForest demand forecasting model...')
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled, y)

# Evaluate model
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mape = np.mean(np.abs((y - y_pred) / (y + 1))) * 100  # +1 to avoid division by zero

logger.info('\n' + '='*60)
logger.info('DEMAND FORECAST MODEL PERFORMANCE')
logger.info('='*60)
logger.info(f'Mean Absolute Error (MAE): {mae:.2f} trips/hour')
logger.info(f'Root Mean Squared Error (RMSE): {rmse:.2f} trips/hour')
logger.info(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
logger.info('='*60)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

logger.info('\nFeature Importance:')
for _, row in feature_importance.iterrows():
    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

# Save model and scaler
logger.info(f'\nSaving model to {MODEL_PATH}...')
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

logger.info(f'Saving feature list...')
import json
with open('demand_features.json', 'w') as f:
    json.dump(feature_cols, f)

logger.info('[OK] Demand forecasting model saved successfully!')
logger.info('='*60)
