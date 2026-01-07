# ML-Ops-CreditCard-AWS
Building Credit Card Fraud Model Using Amazon Web Services

1. Raw Data / Output Storage: AWS S3 Buckets
2. Code Creation: Amazon Sagemaker
3. Code Storage: GitHub
4. Experiment Tracking and Model Registry: ML Flow Registry, S3 Buckets
5. Training Pipeline: AWS StepFunctions + Sagemaker Jobs + Github Actions
6. MLFLow: EC2 Instance + S3 for Experiments + SQLite for 
7. Inferencing: Sagemaker Deployable Model, Sagemaker Batch Transform Job

Other Services Used:
1. AWS Cloud SHell: For running shell scripts
2. AWS CloudWatch: For logs

Run Modes:

| Execution input                | What runs               |
| ------------------------------ | ----------------------- |
| `{}` or `{ "run": "all" }`     | Full pipeline           |
| `{ "run": "train" }`           | Training only           |
| `{ "run": "register" }`        | Register model only     |
| `{ "run": "select_champion" }` | Champion selection only |
| `{ "run": "batch_inference" }` | Batch inference only    |


Code Push Trial
