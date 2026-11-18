"""Lambda metric gate compares held-out labels to XGBoost batch predictions."""
import json
import math
import os

import boto3


def rmse(labels, predictions):
    if not labels or len(labels) != len(predictions):
        raise ValueError('Prediction count must match nonempty test set')
    if not all(math.isfinite(v) for v in labels + predictions):
        raise ValueError('Metrics must be finite')
    return math.sqrt(sum((expected - actual) ** 2 for expected, actual in zip(labels, predictions)) / len(labels))


def parse_numbers(text):
    return [float(line.strip()) for line in text.splitlines() if line.strip()]


def handler(event, _context):
    bucket = os.environ['ARTIFACT_BUCKET']
    s3 = boto3.client('s3')
    prefix = f"runs/{event['run']}/"
    def read(key):
        return s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
    score = rmse(parse_numbers(read(prefix + 'labels.csv')),
                 parse_numbers(read(prefix + 'evaluation/test.csv.out')))
    threshold = float(os.environ['RMSE_THRESHOLD'])
    report_key = prefix + 'evaluation/metrics.json'
    s3.put_object(Bucket=bucket, Key=report_key, ContentType='application/json',
                  Body=json.dumps({'rmse': score, 'threshold': threshold, 'passed': score <= threshold}).encode())
    return {'rmse': score, 'passed': score <= threshold,
            'report_uri': f's3://{bucket}/{report_key}'}
