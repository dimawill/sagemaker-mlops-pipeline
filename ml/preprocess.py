"""Lambda data preparation: validate bounded CSV and publish isolated run channels."""
import csv
import io
import math
import os
import re
from datetime import date

import boto3
from dataset import FIELDS
from train import channel_csv, features_csv


def split_rows(text):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError('Expected day,weekday,promo,price,demand CSV header')
    rows = []
    previous = None
    for item in reader:
        if len(rows) >= 10000:
            raise ValueError('Dataset exceeds 10000 rows')
        if None in item or any(value is None for value in item.values()):
            raise ValueError('Malformed CSV row')
        day = date.fromisoformat(item['day'])
        if previous is not None and day <= previous:
            raise ValueError('Dates must be strictly increasing')
        previous = day
        weekday = int(item['weekday'])
        promo = int(item['promo'])
        price = float(item['price'])
        demand = float(item['demand'])
        if weekday != day.weekday() or promo not in (0, 1) or not all(
                math.isfinite(v) for v in (price, demand)) or price <= 0 or demand < 0:
            raise ValueError('Invalid demand row')
        rows.append({'weekday': weekday, 'promo': promo, 'price': price, 'demand': demand})
    if len(rows) < 20:
        raise ValueError('At least 20 ordered rows required')
    first = int(len(rows) * .7)
    second = int(len(rows) * .85)
    return rows[:first], rows[first:second], rows[second:]


def handler(event, _context):
    run = event['run']
    if not re.fullmatch(r'[A-Za-z0-9-]{1,40}', run):
        raise ValueError('Run name must be 1-40 ASCII alphanumeric/hyphen characters')
    data_bucket = os.environ['DATA_BUCKET']
    artifact_bucket = os.environ['ARTIFACT_BUCKET']
    s3 = boto3.client('s3')
    source = s3.get_object(Bucket=data_bucket, Key='input/demand.csv')['Body'].read(2_000_001)
    if len(source) > 2_000_000:
        raise ValueError('Dataset exceeds 2 MB')
    train, validation, test = split_rows(source.decode('utf-8-sig'))
    prefix = f'runs/{run}/'
    objects = {
        'train.csv': channel_csv(train),
        'validation.csv': channel_csv(validation),
        'test.csv': features_csv(test),
        'labels.csv': '\n'.join(str(row['demand']) for row in test) + '\n',
        'batch.csv': features_csv(test),
    }
    for name, content in objects.items():
        s3.put_object(Bucket=artifact_bucket, Key=prefix + name,
                      Body=content.encode('utf-8'), ContentType='text/csv')
    uri = lambda name: f's3://{artifact_bucket}/{prefix}{name}'
    return { 'train': uri('train.csv'), 'validation': uri('validation.csv'),
             'test': uri('test.csv'), 'labels': uri('labels.csv'), 'batch': uri('batch.csv'),
             'eval_output': uri('evaluation/'), 'batch_output': uri('batch-output/'),
             'prediction_key': prefix + 'evaluation/test.csv.out',
             'report_key': prefix + 'evaluation/metrics.json', 'run': run }
