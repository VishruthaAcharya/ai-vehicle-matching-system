# AI‑Driven Vehicle Matching & Dynamic Pricing System

A production‑ready, end‑to‑end **AI vehicle matching platform** that recommends the best available vehicle for a ride request using **ETA prediction, demand forecasting, dynamic pricing, and preference‑aware vehicle ranking**.

The system is built with **Python, Flask, LightGBM, and scikit‑learn**, and is designed to be reproducible, testable, and extensible.

---
## Implementation Summary

This project implements an end-to-end AI-driven vehicle matching system:

- Generated a synthetic dataset of 10,000 trips simulating urban ride-hailing behavior
- Trained a LightGBM model to predict pickup ETA and trip duration
- Built a RandomForest-based demand forecasting model at grid-hour resolution
- Designed a demand-driven dynamic pricing engine combining demand and supply signals
- Implemented preference-aware vehicle ranking (fastest, cheapest, balanced)
- Integrated all components into a Flask REST API with persistent vehicle storage
- Evaluated models using MAE, RMSE, and MAPE with visual analysis in Jupyter notebooks


##  Key Capabilities

### 1) ETA Prediction

* **Model:** LightGBM Regressor
* **Objective:** Predict pickup ETA and trip duration
* **Training Data:** 10,000 synthetic trips
* **Metrics:**

  * MAE: **4.55 minutes**
  * RMSE: **6.33 minutes**
  * MAPE: **10.54%**
* **Features:** distance, time of day, rush hour, weekend indicator, cyclical hour encoding, surge context
* **Artifact:** `eta_model.joblib`
* **Used in:** `/ride/quote`

---

### 2) Demand Forecasting

* **Model:** RandomForest Regressor
* **Objective:** Predict short‑term demand per geographic grid cell (hourly)
* **Spatial Resolution:** ~900 grid cells (lat/lon bucketing)
* **Metrics:**

  * MAE: **0.00 trips/hour**
  * RMSE: **0.02 trips/hour**
  * MAPE: **0.05%**
* **Features:** temporal signals, lagged demand, trip distance, surge context
* **Artifact:** `demand_model.joblib`
* **Used in:** `/demand/forecast` and pricing engine

> Note: Very low error reflects the structured nature of synthetic demand and is intended as a proof‑of‑concept.

---

### 3) Dynamic Pricing

* **Engine:** `DemandPricingEngine`
* **Approach:** Demand‑driven pricing with supply adjustment
* **Surge Range:** **0.8× – 1.8×**
* **Formula:**

  ```
  final_surge = demand_surge × (supply_surge ^ 0.3)
  ```
* **Signals Used:** predicted demand, available vehicle supply, time‑of‑day modifiers
* **Integrated:** live in `/ride/quote`

---

### 4) Intelligent Vehicle Ranking

* **Goal:** Recommend the best vehicles based on user preference
* **Modes:**

  * `fastest` → minimize ETA
  * `cheapest` → minimize fare
  * `balanced` → trade‑off between ETA and cost
* **Method:** Weighted scoring with normalization
* **Output:** Top‑5 ranked vehicles with ETA, fare, surge, and score

---

##  System Architecture (High Level)

```
Data Generation → Model Training → Saved Models
       ↓                ↓
  trip_data.csv     eta_model.joblib
                    demand_model.joblib
                         ↓
                demand_pricing.py
                         ↓
                    Flask API
                         ↓
                   SQLite (vehicles)
```

---

##  API Endpoints

### Health

* `GET /` — API status and available routes

### Vehicle Management

* `POST /vehicles/update` — Add or update a vehicle
* `GET /vehicles/list` — List all vehicles
* `DELETE /vehicles/<vehicle_id>` — Remove a vehicle

### Core Functionality

* `POST /ride/quote` — Get top‑5 vehicle recommendations (ETA + pricing + ranking)
* `POST /demand/forecast` — Predict demand and recommended surge (bonus feature)

---

##  Testing & Validation

The project includes multiple testing utilities:

* `test_accuracy.py` — Quick accuracy check
* `quick_test.py` — Fast validation of all 4 features
* `test_all_features.py` — Comprehensive system test

All models, endpoints, and pricing logic have been validated with automated and manual tests.

---

##  Project Structure

```
vehicle-matching/
├── generate_data.py            # Synthetic data generation (10K trips)
├── train_model.py              # ETA model training (LightGBM)
├── demand_forecast.py          # Demand model training (RandomForest)
├── demand_pricing.py           # Dynamic pricing engine
├── app.py                      # Flask REST API
├── vehicles.db                 # SQLite vehicle database
├── trip_data.csv               # Training dataset
├── eta_model.joblib            # Trained ETA model
├── demand_model.joblib         # Trained demand model
├── model_features.json         # ETA feature list
├── demand_features.json        # Demand feature list
├── analysis.ipynb              # EDA & evaluation notebook
├── test_api.py                 # API tests
└── README.md                   # Documentation
```

---

##  Setup & Run

### Requirements

* Python **3.12+**
* pip
* Windows / Linux / macOS

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API

```bash
python app.py
```

API will be available at `http://localhost:5000`

---

##  Example Usage

### Register a Vehicle

```bash
curl -X POST http://localhost:5000/vehicles/update \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id":"KA01AB0001","lat":13.0,"lon":77.6,"vehicle_type":"Mini"}'
```

### Get Ride Quote

```bash
curl -X POST http://localhost:5000/ride/quote \
  -H "Content-Type: application/json" \
  -d '{"origin_lat":13.0,"origin_lon":77.6,"dest_lat":13.05,"dest_lon":77.65,"preference":"balanced"}'
```

---

##  Limitations

* Uses synthetic data (no real road‑network routing)
* No real‑time traffic or GPS streams
* Demand patterns are simulated for demonstration

---

##  Future Enhancements

* Real‑time traffic integration (OSM / Maps API)
* Reinforcement learning for surge pricing
* Predictive vehicle repositioning
* User preference learning over time
* Containerized deployment (Docker)

---

##  Author

**Vishrutha Acharya**
Email: [vishruthaacharya30@gmail.com](mailto:vishruthaacharya30@gmail.com)


