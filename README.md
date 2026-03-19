# NYC 311 Service Request Prediction API

End-to-end machine learning project that predicts NYC 311 service request outcomes using historical complaint data. This project demonstrates a production-style ML workflow including data preprocessing, model training, containerized deployment, and real-time inference using AWS SageMaker.

The system is designed as a **portfolio-ready machine learning application**, showing how data science models can be operationalized and served through an API.

---

## Project Highlights

- Built an end-to-end ML pipeline for NYC 311 service request prediction  
- Performed data preprocessing and feature engineering using Python and pandas  
- Trained machine learning models using scikit-learn  
- Containerized inference using Docker  
- Deployed a live model endpoint with AWS SageMaker  
- Integrated AWS services including S3, SageMaker, and IAM  
- Designed the system for API-based access and automated data refreshes  

---

## Business Problem

New York City's 311 service request system receives millions of complaints each year covering issues such as:

- Noise complaints  
- Sanitation issues  
- Building violations  
- Street conditions  
- Public safety concerns  

Analyzing patterns in these requests can help:

- Anticipate service demand  
- Identify complaint trends across boroughs  
- Improve response prioritization  
- Allocate city resources more effectively  

---

## Dataset

**Source:** NYC Open Data – 311 Service Requests  

---

## Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- AWS (S3, SageMaker, IAM)  
- Docker  

---

## Deployment Architecture

NYC 311 Dataset → Feature Engineering → Model → Docker → SageMaker → API

---

## Example API Request

```json
{
  "borough": "BROOKLYN",
  "complaint_type": "Noise - Residential",
  "created_date": "2024-01-01 22:00:00"
}
```

---

## Example Response

```json
{
  "n_rows": 1,
  "predictions": [
    {
      "pred_gt_7d": 1,
      "pred_prob_gt_7d": 0.82,
      "pred_prob_within_7d": 0.18,
      "predicted_close_within_7_days": 0
    }
  ]
}
```

---

## Docker Usage

```bash
docker build -t nyc311-ml-api .
docker run -p 8080:8080 nyc311-ml-api
```

---

## Future Improvements

- Public API via API Gateway  
- Automated data refresh  
- Model retraining pipeline  
