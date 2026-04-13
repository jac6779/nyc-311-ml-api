# NYC 311 Complaint Resolution Prediction API

An end-to-end machine learning project that predicts whether a New York City 311 complaint will be resolved within one week using historical service request data.

This project follows a structured CRISP-DM workflow and is designed to show the full path from notebook-based modeling to production-style inference. The final model was packaged into a FastAPI application, containerized with Docker, and deployed on AWS.

* * *

## Live API

- [Formatted HTML page](https://d3oxki74u11f6.cloudfront.net)
- [Swagger Docs](https://dyypyhmjdv.us-east-1.awsapprunner.com/docs)
- [Health Check](https://dyypyhmjdv.us-east-1.awsapprunner.com/health)

* * *

## Project Overview

New York City's 311 system receives high volumes of service requests across categories like noise, sanitation, street conditions, and building issues. Resolution times can vary significantly depending on the agency, complaint type, time of day, and location.

This project focuses on predicting whether a complaint will be resolved within seven days, turning a real public-service process into a practical binary classification problem.

### Target Variable

Binary classification:

- `1` → Complaint resolved within 7 days
- `0` → Complaint not resolved within 7 days

* * *

## Data Source

- NYC Open Data – 311 Service Requests

The raw data was pulled from the NYC Open Data API and included complaint metadata, timestamps, agency information, location fields, ZIP code, borough, latitude, and longitude.

* * *

## Project Structure (CRISP-DM)

### 1️⃣ Preprocessing (`01_nyc_311_project_preprocessing.ipynb`)

- Pulled NYC 311 complaint data from the NYC Open Data API
- Parsed and cleaned `created_date` and `closed_date`
- Calculated `resolution_time_days`
- Created the binary target `resolution_in_wk`
- Dropped `status` to avoid target leakage
- Engineered time-based features:
  - `complaint_hr`
  - `complaint_day`
  - `complaint_month`
- Standardized borough and ZIP code values
- Converted latitude and longitude to numeric fields
- Filled missing categorical values with `"unknown"`

* * *

### 2️⃣ Exploratory Data Analysis (`02_nyc_311_project_exploratory_analysis.ipynb`)

- Reviewed dataset structure, missingness, and class balance
- Explored the target distribution for complaints resolved within one week
- Analyzed resolution time patterns in days
- Compared complaint behavior by borough
- Compared complaint categories and agency-level patterns
- Used visual analysis to understand which features were likely to matter most for prediction

* * *

### 3️⃣ Feature Engineering (`03_nyc_311_project_feature_engineering.ipynb`)

- Removed unused or leaky variables before modeling
- Defined `resolution_in_wk` as the modeling target
- Selected predictors describing:
  - complaint type
  - agency
  - borough
  - location type
  - time of complaint
  - latitude / longitude
- Used a preprocessing pipeline with:
  - `StandardScaler` for numeric fields
  - `OneHotEncoder(handle_unknown="ignore")` for categorical fields
- Split the data into stratified train/test sets
- Exported transformed train/test datasets for modeling
- Saved the fitted preprocessing pipeline for deployment

* * *

### 4️⃣ Modeling (`04_nyc_311_project_modeling.ipynb`)

Implemented and compared three classification models:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation focused on:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

### Model Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.957 | 0.529 | 0.291 | 0.888 | 0.438 |
| XGBoost | 0.952 | 0.470 | 0.219 | 0.949 | 0.356 |
| Logistic Regression | 0.943 | 0.414 | 0.209 | 0.910 | 0.340 |

Random Forest was selected as the final model because it produced the strongest overall balance of ROC-AUC, PR-AUC, precision, and F1-score on this imbalanced classification task.

* * *

### 5️⃣ Export Pipeline (`05_nyc_311_project_export_pipeline.ipynb`)

- Loaded the saved preprocessor and final Random Forest model
- Tested inference using deployment-ready artifacts
- Built a ZIP-to-coordinate lookup table using median latitude/longitude by ZIP
- Prepared supporting files for deployment and easier user input handling
- Structured outputs for downstream API use and Tableau-ready workflows

* * *

## Modeling Approach

This project intentionally compares:

- Interpretable baseline modeling → Logistic Regression
- Tree-based ensemble modeling → Random Forest
- Boosted tabular modeling → XGBoost

Because the target class was relatively infrequent, model evaluation emphasized not just ROC-AUC but also PR-AUC, precision, recall, and F1-score.

* * *

## Key Techniques

- Binary target engineering from real complaint resolution timestamps
- Leakage prevention by removing outcome-related variables
- Time-based feature engineering from complaint creation timestamps
- Geographic cleaning using borough and ZIP standardization
- Preprocessing with scaling + one-hot encoding
- Stratified train/test split for balanced evaluation
- Multi-model comparison framework for tabular classification
- Artifact export for deployment-ready inference

* * *

## Deployment Architecture

NYC Open Data API → Preprocessing → Feature Engineering → Model Training → Saved Artifacts (`.joblib`) → Dockerized FastAPI App → AWS Deployment → Public API

* * *

## Example API Request

```json
{
  "agency": "NYPD",
  "complaint_type": "Noise - Residential",
  "descriptor": "Loud Music/Party",
  "location_type": "Residential Building/House",
  "borough": "BROOKLYN",
  "incident_zip": "11201",
  "latitude": 40.6943,
  "longitude": -73.9928,
  "complaint_hr": 22,
  "complaint_day": 5,
  "complaint_month": 7
}
```

## Example Response

```json
{
  "prediction": 1.0,
  "probability": 0.82
}
```

* * *

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Docker
- AWS
- Tableau

* * *

## Docker Usage

```bash
docker build -t nyc-311-api .
docker run -p 8080:8080 nyc-311-api
```

* * *

## Why This Project Matters

This project demonstrates:

- End-to-end supervised machine learning workflow
- Real-world target engineering from operational data
- Feature engineering across categorical, temporal, and geographic fields
- Model comparison on an imbalanced classification problem
- Transition from notebook experimentation to deployment-ready inference
- Production-oriented ML packaging with FastAPI, Docker, and AWS

* * *

## Related Projects

- [Citi Bike Dock Availability Prediction](https://github.com/jac6779/citi-bike-prediction)  

- [Brooklyn Home Price Prediction API](https://github.com/jac6779/brooklyn-home-sales-llm)  

These projects complement this work by showing:

- API deployment
- cloud-based model serving
- end-to-end tabular ML pipelines
- production-style workflows

* * *

## Author

Justin Cox

GitHub: [https://github.com/jac6779](https://github.com/jac6779)  
LinkedIn: [https://www.linkedin.com/in/justincox1](https://www.linkedin.com/in/justincox1)
