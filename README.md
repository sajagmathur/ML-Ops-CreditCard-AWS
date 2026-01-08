# ML-Ops-CreditCard-AWS
Building Credit Card Fraud Model Using Amazon Web Services
**Pipeline 1: **
1. Raw Data / Output Storage: AWS S3 Buckets
2. Code Creation: Amazon Sagemaker
3. Code Storage: GitHub
4. Experiment Tracking and Model Registry: ML Flow Registry, S3 Buckets
5. Training Pipeline: AWS StepFunctions + Sagemaker Jobs + Github Actions
6. MLFLow: EC2 Instance + S3 for Experiments + SQLite for 
7. Inferencing: Sagemaker Deployable Model, Sagemaker Batch Transform Job
8. Monitoring: Sagemaker Processing Job

Other Services Used:
1. AWS Cloud SHell: For running shell scripts
2. AWS CloudWatch: For logs

Run Modes:

Execution input	What runs
 { "run": "all" }	Full pipeline
{ "run": "train" }	Start from Training
{ "run": "register" }	Start From Register 
{ "run": "select_champion" }	Start from Champion selection 
{ "run": "batch_inference" }	Start From Batch inference
{"run": "monitor"} 	Start From Monitor 
<img width="584" height="286" alt="image" src="https://github.com/user-attachments/assets/8f82f3e3-bb1f-42d8-8e07-c7691b1c19c7" />


Code Push Trial V1
