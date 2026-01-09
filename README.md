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

**Pipeline 2**
1. Sagemaker Batch Transform Job
2. CI Implementation using Github Actions
3. Machine using Step Functions

**Pipeline 3**
1. Sagemaker Processing Job
2. CI Implementation using Github Actions
3. Machine using Step Functions
Other Services Used:
1. AWS Cloud SHell: For running shell scripts
2. AWS CloudWatch: For logs

**Pipeline 4**
1. Retraining trigger using Lambda
2. Machine Implemented in StepFunctions
Execution input	What runs
 { "run": "all" }	Full pipeline
{ "run": "train" }	Start from Training
{ "run": "register" }	Start From Register 
{ "run": "select_champion" }	Start from Champion selection 
{ "run": "batch_inference" }	Start From Batch inference
{"run": "monitor"} 	Start From Monitor 
<img width="584" height="286" alt="image" src="https://github.com/user-attachments/assets/8f82f3e3-bb1f-42d8-8e07-c7691b1c19c7" />


Code Push Trial V1

---
Setup of Step 2: Model Registry on AWS StepFunctions

1. Ensure EC2 has SSM Access: AmazonSSMManagedInstanceCore IAM role
2. Check if ssm agent is running
	1. Enter SSM Agent Setup in AWS CloudShell: aws ssm start-session --target i-08d78517733fe6290
	2. Check SSM Agent is running or not: sudo systemctl status amazon-ssm-agent

Errors Corrected:
1. IAM Permission Issue: 
Option A (recommended first): Attach an inline policy
This avoids affecting other services.
	1. On the role page, click Add permissions
	2. Choose Create inline policy
	3. Click the JSON tab
	4. Delete everything in the editor
	5. Paste this policy:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowSSMSendCommand",
      "Effect": "Allow",
      "Action": [
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:ListCommands",
        "ssm:ListCommandInvocations"
      ],
      "Resource": [
        "arn:aws:ec2:us-east-1:075960506214:instance/i-08d78517733fe6290",
        "arn:aws:ssm:us-east-1:075960506214:document/*"
        "arn:aws:ssm:us-east-1::document/AWS-RunShellScript"
      ]
    }
  ]
}
	1. Click Next
	2. Policy name:

StepFunctions-SSM-SendCommand
	3. Click Create policy
✅ Permissions are now attached

Ron only register model: 
{
  "runOnly": "RegisterModel"
}

Launch MLFlow using EC2:
aws ec2 start-instances --instance-ids i-08d78517733fe6290

Describe instance:
aws ec2 describe-instances --instance-ids i-08d78517733fe6290 --query "Reservations[0].Instances[0].State.Name" --output text


Launching MLFlow
ssh -i my-key.pem -L 5000:localhost:5000 ec2-user@10.85.114.162


Clean Up MLFLow:

2. Launch instance: aws ssm start-session --target i-08d78517733fe6290

Run this command:

aws s3 rm s3://mlops-creditcard/prod_outputs/mlflow --recursive 

set -e
echo "Removing MLflow DB"
rm -f /home/ssm-user/mlflow/mlflow.db
sudo rm -rf /home/ssm-user/mlops_run
rm -f /home/ssm-user/mlflow/mlflow.db-journal
rm -rf /home/ssm-user/mlflow/mlruns
echo "Listing MLflow directory"
ls -lh /home/ssm-user/mlflow || true
echo "MLflow DB cleanup complete"

ls /home/ssm-user/mlflow 

Create Model: 

Go Here: 
Models | Amazon SageMaker AI | us-east-1

Model Name: inference-aws-model
Container input options: Provide model artifacts and inference image location
Model Compression Type: CompressedModel --> Use a single model
Container: 
683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

Location of Artifacts: https://mlops-creditcard.s3.us-east-1.amazonaws.com/prod_codes/inference_aws.tar.gz


BATCH Transform IAM Fix:
Error: AccessDeniedException : 'arn:aws:iam::075960506214:role/service-role/StepFunctions-mlops-creditcard-pipeline-role-nlzj4jhtt' is not authorized to create managed-rule. first lets correct this

Fix: Add IAM Role

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "events:PutRule",
        "events:PutTargets",
        "events:DescribeRule",
        "events:DeleteRule",
        "events:RemoveTargets"
      ],
      "Resource": "*"
    }
  ]
}

Run Particular Job: 
Execution input	What runs
 { "run": "all" }	Full pipeline
{ "run": "train" }	Start from Training
{ "run": "register" }	Start From Register 
{ "run": "select_champion" }	Start from Champion selection 
{ "run": "batch_inference" }	Start From Batch inference
{"run": "monitor"} 	Start From Monitor 

From <https://github.com/sajagmathur/ML-Ops-CreditCard-AWS> 


---

Created separate jobs for monitoring and inferencing.

---

Create Retraining Trigger:

1. Created a stepfunctions machine
2. Create Lambda function

1️⃣ Create IAM Role for Lambda
Go to

IAM → Roles → Create role

Trusted entity

AWS service

Lambda

Permissions

Attach:

AWSLambdaBasicExecutionRole

Then add inline policy 👇

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::mlops-creditcard/monitoring_outputs/retraining_decision/*"
    }
  ]
}

Role name
lambda-retraining-decision-role


✅ Create role

2️⃣ Create Lambda Function
Go to

Lambda → Create function

Author from scratch

Function name:

check_retraining_decision


Runtime:

Python 3.10


Execution role:

Use existing role

Select:

lambda-retraining-decision-role


Click Create function

3️⃣ Paste Lambda Code

In Code → lambda_function.py, paste 👇

import boto3
import csv
import io

s3 = boto3.client("s3")

BUCKET = "mlops-creditcard"
KEY = "monitoring_outputs/retraining_decision/retraining_decision.csv"

def lambda_handler(event, context):
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    body = obj["Body"].read().decode("utf-8")

    reader = csv.reader(io.StringIO(body))
    rows = list(reader)

    decision = rows[1][0].strip().upper()

    return {
        "retraining_required": decision
    }


Click Deploy

4️⃣ Test Lambda (CRITICAL)
Create test event

Name: test

Body:

{}

Click Test
Expected output:
{
  "retraining_required": "YES"
}


If you see:

❌ AccessDenied → IAM policy wrong

❌ IndexError → CSV format wrong

Fix before moving on.

5️⃣ Create Retraining Trigger Step Function
Go to

Step Functions → Create state machine

Author with code

Type: Standard

Paste definition 👇
{
  "Comment": "Auto retraining trigger based on monitoring CSV output",
  "StartAt": "ReadRetrainingDecision",
  "States": {
    "ReadRetrainingDecision": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {
        "FunctionName": "check_retraining_decision"
      },
      "ResultSelector": {
        "decision.$": "$.Payload.retraining_required"
      },
      "ResultPath": "$.decision",
      "Next": "DecisionChoice"
    },

    "DecisionChoice": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.decision.decision",
          "StringEquals": "YES",
          "Next": "TriggerTrainingPipeline"
        }
      ],
      "Default": "NoRetrainingNeeded"
    },

    "TriggerTrainingPipeline": {
      "Type": "Task",
      "Resource": "arn:aws:states:::states:startExecution",
      "Parameters": {
        "StateMachineArn": "PASTE_YOUR_MAIN_PIPELINE_ARN_HERE",
        "Input": {
          "run": "all",
          "triggered_by": "monitoring",
          "reason": "auto_retraining"
        }
      },
      "End": true
    },

    "NoRetrainingNeeded": {
      "Type": "Succeed"
    }
  }
}

Replace:
PASTE_YOUR_MAIN_PIPELINE_ARN_HERE

IAM Role

Create new role automatically

Or reuse your existing Step Functions role

Name
creditcard-auto-retraining-trigger


Click Create state machine

6️⃣ Test Step Function Manually
Click Start execution

Input:

{}

Expected behavior
CSV value	Result
YES	Training pipeline starts
NO	Workflow ends successfully


For Demo:

Clear: 
Prod_codes
Monitoring_Inputs
Prod_Outputs
Data/Raw
 prod_inputs

Fill:
Data/raw
Prod_inputs
Monitoring_Inputs/Current_Data
Clean MLFlow EC2

Launch instance: aws ssm start-session --target i-08d78517733fe6290

Run this command:

aws s3 rm s3://mlops-creditcard/prod_outputs/mlflow --recursive 

set -e
echo "Removing MLflow DB"
rm -f /home/ssm-user/mlflow/mlflow.db
sudo rm -rf /home/ssm-user/mlops_run
rm -f /home/ssm-user/mlflow/mlflow.db-journal
rm -rf /home/ssm-user/mlflow/mlruns
echo "Listing MLflow directory"
ls -lh /home/ssm-user/mlflow || true
echo "MLflow DB cleanup complete"

<img width="732" height="10539" alt="image" src="https://github.com/user-attachments/assets/ae50ef00-9206-41b4-8daf-f11c06938122" />

