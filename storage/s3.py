import os
import boto3
from typing import Optional, Union, BinaryIO
from pathlib import Path

class S3Storage:
    def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None):
        """
        Initialize S3 Storage wrapper.
        
        Credentials are automatically loaded from environment variables:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_SESSION_TOKEN (optional)
        - AWS_REGION (optional)
        
        :param bucket_name: Name of the S3 bucket
        :param endpoint_url: Optional custom endpoint URL (e.g. for R2, MinIO). 
                             Defaults to S3_ENDPOINT_URL env var if not provided.
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url
        )

    def put_file(self, local_path: Union[str, Path], key: str) -> None:
        """
        Upload a local file to S3.
        
        :param local_path: Path to the local file
        :param key: S3 object key
        """
        self.s3_client.upload_file(str(local_path), self.bucket_name, key)

    def get_file(self, key: str, local_path: Union[str, Path]) -> None:
        """
        Download a file from S3 to local path.
        
        :param key: S3 object key
        :param local_path: Path to save the file locally
        """
        self.s3_client.download_file(self.bucket_name, key, str(local_path))

    def put_object(self, body: Union[bytes, str, BinaryIO], key: str) -> None:
        """
        Upload raw data (bytes, string, or file-like object) to S3.
        
        :param body: Data to upload
        :param key: S3 object key
        """
        if isinstance(body, str):
            body = body.encode("utf-8")
            
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=body
        )

    def get_object(self, key: str) -> bytes:
        """
        Get object content as bytes.
        
        :param key: S3 object key
        :return: Object content as bytes
        """
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return response["Body"].read()



