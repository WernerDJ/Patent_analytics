import boto3
from botocore.exceptions import ClientError
import os

# Test S3 connection
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_S3_REGION_NAME', 'eu-north-1')
)

bucket_name = os.environ.get('AWS_STORAGE_BUCKET_NAME')

try:
    # Try to list objects in bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
    print(f"✅ Successfully connected to S3 bucket: {bucket_name}")
    print(f"Region: {os.environ.get('AWS_S3_REGION_NAME')}")
    
    # Try to upload a test file
    test_content = b"Test file from Docker"
    s3_client.put_object(
        Bucket=bucket_name,
        Key='test/docker_test.txt',
        Body=test_content,
        ContentType='text/plain'
    )
    
    url = f"https://{bucket_name}.s3.amazonaws.com/test/docker_test.txt"
    print(f"✅ Test file uploaded successfully!")
    print(f"URL: {url}")
    
except ClientError as e:
    print(f"❌ Error connecting to S3: {e}")
    print(f"Bucket: {bucket_name}")
    print(f"Region: {os.environ.get('AWS_S3_REGION_NAME')}")
