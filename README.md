# NYC 311 Service Request Prediction API

End-to-end machine learning project that predicts NYC 311 service request outcomes using historical complaint data. This project demonstrates a production-style ML workflow including data preprocessing, model training, containerized deployment, and real-time inference using AWS.

The system is designed as a portfolio-ready machine learning application, showing how data science models can be operationalized and served through a live API.

---

## 🚀 Live API

🔗 **Swagger Docs:**  
https://dyypyhmjdv.us-east-1.awsapprunner.com/docs  

🔗 **Health Check:**  
https://dyypyhmjdv.us-east-1.awsapprunner.com/health  

---

## 📌 Project Highlights

- Built an end-to-end ML pipeline for NYC 311 service request prediction  
- Performed data preprocessing and feature engineering using Python and pandas  
- Trained machine learning models using scikit-learn  
- Combined preprocessing + model into a single deployable pipeline  
- Containerized inference using Docker  
- Deployed a live API using AWS App Runner  
- Stored container images in Amazon ECR  
- Designed interactive API with Swagger documentation  

---

## 🧠 Business Problem

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

## 📊 Dataset

Source: NYC Open Data – 311 Service Requests  

---

## 🛠 Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- FastAPI  
- Docker  
- AWS (ECR, App Runner, IAM)  

---

## 🏗 Deployment Architecture

NYC 311 Dataset → Feature Engineering → ML Pipeline → Docker → Amazon ECR → AWS App Runner → Public API

---

## 🔌 Example API Request

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

## 📈 Example Response

```json
{
  "prediction": 1.0,
  "probability": 0.82
}
```

---

## 🐳 Docker Usage

```bash
docker build -t nyc-311-api .
docker run -p 8080:8080 nyc-311-api
```

---

## 🔮 Future Improvements

- Simplify API inputs (derive time features automatically)  
- Integrate live NYC Open Data API for real-time predictions  
- Add automated retraining pipeline  
- Add monitoring and logging  
- Introduce API Gateway for advanced routing and security  

---

## 📌 Notes

This project demonstrates transitioning from a notebook-based data science workflow to a production-ready ML system, including API design, containerization, and cloud deployment.
