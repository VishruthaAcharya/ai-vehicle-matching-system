import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib
import json
import logging
import sys


def setup_logging(log_path='train_log.txt'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (append)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def main():
    logger = setup_logging()
    try:
        logger.info('Loading data: trip_data.csv')
        df = pd.read_csv('trip_data.csv')

        logger.info('Creating features')
        df['distance_squared'] = df['trip_distance'] ** 2
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 10)) |
                             ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)

        feature_cols = ['trip_distance', 'distance_squared', 'hour', 'hour_sin',
                        'hour_cos', 'is_rush_hour', 'is_weekend', 'surge_multiplier']

        X = df[feature_cols]
        y = df['trip_duration']

        logger.info('Splitting data (80/20)')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f'Training model on {len(X_train)} records; testing on {len(X_test)}')

        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            verbose=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # avoid divide-by-zero in MAPE
        y_test_safe = y_test.replace(0, 1e-6)
        mape = np.mean(np.abs((y_test_safe - y_pred) / y_test_safe)) * 100

        logger.info('MODEL PERFORMANCE')
        logger.info(f'Mean Absolute Error (MAE): {mae:.2f} minutes')
        logger.info(f'Root Mean Squared Error (RMSE): {rmse:.2f} minutes')
        logger.info(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info('\nFeature importance:')
        for _, row in importance.iterrows():
            logger.info(f"{row['feature']}: {row['importance']}")

        logger.info('Saving model to eta_model.joblib')
        joblib.dump(model, 'eta_model.joblib')

        logger.info('Saving feature list to model_features.json')
        with open('model_features.json', 'w') as f:
            json.dump(feature_cols, f)

        logger.info('Saved artifacts: eta_model.joblib, model_features.json')

        # Show a small sample of predictions
        sample = X_test.head(5).copy()
        sample['actual_duration'] = y_test.head(5).values
        sample['predicted_duration'] = model.predict(sample[feature_cols])
        logger.info('Sample predictions:')
        logger.info('\n' + sample.to_string(index=False))

    except Exception as e:
        logger.exception('Error during training')


if __name__ == '__main__':
    main()