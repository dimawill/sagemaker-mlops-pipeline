"""Register a model package only after the held-out RMSE gate passes."""
import os

import boto3


def handler(event, _context):
    if event['passed'] is not True:
        raise ValueError('Refusing to register model without passing evaluation')
    result = boto3.client('sagemaker').create_model_package(
        ModelPackageGroupName=os.environ['GROUP_NAME'],
        ModelPackageDescription='Synthetic daily demand XGBoost; held-out RMSE gate',
        ModelApprovalStatus='Approved',
        InferenceSpecification={
            'Containers': [{'Image': os.environ['IMAGE_URI'], 'ModelDataUrl': event['model_uri']}],
            'SupportedContentTypes': ['text/csv'],
            'SupportedResponseMIMETypes': ['text/csv'],
        },
        ModelMetrics={'ModelQuality': {'Statistics': {
            'ContentType': 'application/json', 'S3Uri': event['report_uri']}}},
    )
    return {'model_package_arn': result['ModelPackageArn']}
