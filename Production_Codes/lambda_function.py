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
