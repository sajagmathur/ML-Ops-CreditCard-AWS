
import boto3
import csv
import io
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET = "mlops-creditcard"
KEY = "monitoring_outputs/retraining_decision/retrain_decision.csv"  # corrected

def lambda_handler(event, context):
    try:
        # Optional but helpful: confirm the object exists
        s3.head_object(Bucket=BUCKET, Key=KEY)

        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        body = obj["Body"].read().decode("utf-8")

        reader = csv.reader(io.StringIO(body))
        rows = list(reader)

        # Expect at least 2 rows (header + data), first column present
        if len(rows) < 2 or len(rows[1]) < 1:
            raise ValueError("CSV does not contain row 2 col 1 as expected")

        decision = rows[1][0].strip().upper()
        logger.info(f"Read decision from s3://{BUCKET}/{KEY}: {decision}")

        # If you want to return the raw "YES"/"NO" string:
        return {"retraining_required": decision}

        # Or, if you prefer boolean:
        # return {"retraining_required": decision == "YES"}

    except ClientError as e:
        code = e.response["Error"]["Code"]
        logger.error(f"AWS error {code} for s3://{BUCKET}/{KEY}")
        return {"statusCode": 404 if code in ("NoSuchKey", "404") else 500, "error": code}
    except Exception as ex:
        logger.exception("Unhandled error")
        return {"statusCode": 500, "error": str(ex)}
