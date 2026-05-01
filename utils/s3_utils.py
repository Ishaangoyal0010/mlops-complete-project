import boto3
import os

BUCKET_NAME = "carproject-ishaan"
MODEL_KEY = "model/best_model.pkl"


def download_model(local_path="artifacts/best_model.pkl"):
    os.makedirs("artifacts", exist_ok=True)

    s3 = boto3.client("s3")

    s3.download_file(
        BUCKET_NAME,
        MODEL_KEY,
        local_path
    )
    print("model downloaded")


def upload_file(local_path, s3_key):
    s3 = boto3.client("s3")

    s3.upload_file(
        local_path,
        BUCKET_NAME,
        s3_key
    )

    print("File uploaded to S3")