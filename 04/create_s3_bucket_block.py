from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Accessing variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Now you can use these credentials in your AWS SDK

def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key 
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="my-first-bucket-abc-jith", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()