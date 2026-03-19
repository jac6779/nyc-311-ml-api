\# NYC 311 Service Request Prediction API



End-to-end machine learning project that predicts NYC 311 service request outcomes using historical complaint data. This project demonstrates a production-style ML workflow including data preprocessing, model training, containerized deployment, and real-time inference using AWS SageMaker.



The system is designed as a \*\*portfolio-ready machine learning application\*\*, showing how data science models can be operationalized and served through an API.



\---



\## Project Highlights



\- Built an \*\*end-to-end ML pipeline\*\* for NYC 311 service request prediction  

\- Performed \*\*data preprocessing and feature engineering\*\* using Python and pandas  

\- Trained machine learning models using scikit-learn  

\- \*\*Containerized inference using Docker\*\*  

\- Deployed a \*\*live model endpoint with AWS SageMaker\*\*  

\- Integrated AWS services including \*\*S3, SageMaker, and IAM\*\*  

\- Designed the system for \*\*API-based access and automated data refreshes\*\*  



\---



\## Business Problem



New York City's \*\*311 service request system\*\* receives millions of complaints each year covering issues such as:



\- noise complaints  

\- sanitation issues  

\- building violations  

\- street conditions  

\- public safety concerns  



Analyzing patterns in these requests can help:



\- anticipate service demand  

\- identify complaint trends across boroughs  

\- improve response prioritization  

\- allocate city resources more effectively  



This project demonstrates how machine learning can be used to \*\*model and predict patterns in NYC 311 service requests\*\* using historical data.



\---



\## Dataset



\*\*Source:\*\* NYC Open Data – 311 Service Requests  



Key fields used include:



\- Complaint Type  

\- Borough  

\- Created Date / Time  

\- Agency responsible  

\- Resolution status  



Data preprocessing included:



\- cleaning inconsistent complaint categories  

\- handling missing values  

\- feature engineering from timestamps  

\- encoding categorical variables for model inputs  



\---



\## Tech Stack



\### Programming

\- Python  

\- Pandas  

\- NumPy  



\### Machine Learning

\- Scikit-learn  

\- Feature engineering pipelines  



\### Infrastructure

\- AWS S3 — model artifacts and data storage  

\- AWS SageMaker — model deployment and inference  

\- AWS IAM — secure access management  



\### Deployment

\- Docker containerization  

\- SageMaker real-time inference endpoint  



\---



\## Machine Learning Workflow



1\. \*\*Data Collection\*\*  

&#x20;  Retrieve NYC 311 service request data from NYC Open Data.



2\. \*\*Data Cleaning \& Feature Engineering\*\*  

&#x20;  Transform raw request data into structured model features.



3\. \*\*Model Training\*\*  

&#x20;  Train machine learning models to identify patterns in service requests.



4\. \*\*Model Evaluation\*\*  

&#x20;  Evaluate model performance using validation metrics.



5\. \*\*Deployment\*\*  

&#x20;  Package inference code inside a Docker container and deploy to AWS SageMaker.



6\. \*\*Real-Time Inference\*\*  

&#x20;  Serve predictions through a live SageMaker endpoint.



\---



\## Deployment Architecture

NYC 311 Dataset

│

▼

Data Cleaning / Feature Engineering

│

▼

Model Training (Python / scikit-learn)

│

▼

Docker Container

│

▼

AWS SageMaker Endpoint

│

▼

API Requests for Predictions



\---



\## Model Performance



The model was evaluated on a validation dataset.



\*\*Sample metrics (update with your actual results):\*\*



\- Accuracy: 0.84  

\- Precision: 0.81  

\- Recall: 0.79  

\- F1 Score: 0.80  



The model performs well on high-frequency complaint categories such as noise and sanitation complaints.



\---



\## Example API Request



Example JSON input:



```json

{

&#x20; "borough": "BROOKLYN",

&#x20; "complaint\_type": "Noise - Residential",

&#x20; "created\_date": "2024-01-01 22:00:00"

}



