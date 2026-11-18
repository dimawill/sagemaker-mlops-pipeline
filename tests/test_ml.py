import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from dataset import dataset_csv, generate_rows
from preprocess import split_rows
from train import channel_csv, features_csv
from evaluate import parse_numbers, rmse


def test_deterministic_chronological_dataset_and_channels():
    rows = generate_rows()
    assert rows == generate_rows()
    train, validation, test = split_rows(dataset_csv(rows))
    assert len(train) == 168 and len(validation) == 36 and len(test) == 36
    assert len(channel_csv(train).splitlines()) == 168
    assert len(features_csv(test).splitlines()[0].split(',')) == 3
    assert len(channel_csv(train).splitlines()[0].split(',')) == 4
    assert rows != generate_rows(seed=18)


def test_data_contract_rejects_bad_rows():
    original = dataset_csv(generate_rows())
    with pytest.raises(ValueError, match='header'):
        split_rows(original.replace('weekday', 'weekday_changed', 1))
    with pytest.raises(ValueError, match='strictly increasing'):
        split_rows(original.replace('2025-01-02', '2025-01-01', 1))
    with pytest.raises(ValueError, match='Invalid'):
        split_rows(original.replace(',2,1,8.0,', ',2,3,8.0,', 1))
    with pytest.raises(ValueError, match='At least 20'):
        split_rows('\n'.join(original.splitlines()[:10]) + '\n')
    with pytest.raises(ValueError, match='10000'):
        split_rows(dataset_csv(generate_rows(count=10001)))


def test_metric_gate_and_invalid_predictions():
    assert rmse([5.0, 7.0], [4.0, 8.0]) == 1.0
    assert parse_numbers('1.0\n 2.0\n') == [1.0, 2.0]
    assert rmse([5.0], [5.0]) <= 15
    assert rmse([5.0], [21.0]) > 15
    with pytest.raises(ValueError):
        rmse([1.0], [])
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        rmse([math.nan], [1.0])


def test_preprocess_and_evaluate_handlers_without_aws(monkeypatch):
    import preprocess
    import evaluate

    class S3:
        def __init__(self):
            self.objects = {('input', 'input/demand.csv'): dataset_csv(generate_rows()).encode()}

        def get_object(self, Bucket, Key):
            import io
            return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}

        def put_object(self, Bucket, Key, Body, **_kwargs):
            self.objects[(Bucket, Key)] = Body

    store = S3()
    monkeypatch.setattr(preprocess.boto3, 'client', lambda _service: store)
    monkeypatch.setattr(evaluate.boto3, 'client', lambda _service: store)
    monkeypatch.setenv('DATA_BUCKET', 'input')
    monkeypatch.setenv('ARTIFACT_BUCKET', 'artifacts')
    monkeypatch.setenv('RMSE_THRESHOLD', '15')
    result = preprocess.handler({'run': 'run-1'}, None)
    assert result['train'] == 's3://artifacts/runs/run-1/train.csv'
    assert len(store.objects[('artifacts', 'runs/run-1/test.csv')].splitlines()) == 36
    with pytest.raises(ValueError, match='Run name'):
        preprocess.handler({'run': '../bad'}, None)
    store.objects[('artifacts', result['prediction_key'])] = store.objects[('artifacts', 'runs/run-1/labels.csv')]
    passed = evaluate.handler({'run': 'run-1'}, None)
    assert passed['passed'] and passed['rmse'] == 0
    assert b'"passed": true' in store.objects[('artifacts', result['report_key'])]
    store.objects[('artifacts', result['prediction_key'])] = b'10000\n' * 36
    failed = evaluate.handler({'run': 'run-1'}, None)
    assert not failed['passed'] and failed['rmse'] > 15


def test_registry_rejects_failed_gate_without_aws_call(monkeypatch):
    import registry
    monkeypatch.setattr(registry.boto3, 'client', lambda _service: pytest.fail('unexpected AWS request'))
    with pytest.raises(ValueError, match='Refusing'):
        registry.handler({'passed': False}, None)
