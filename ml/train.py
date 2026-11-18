"""Prepare SageMaker built-in XGBoost training channels (label-first, no header)."""
import csv
import io


def channel_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerows([[row['demand'], row['weekday'], row['promo'], row['price']]
                      for row in rows])
    return out.getvalue()


def features_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerows([[row['weekday'], row['promo'], row['price']] for row in rows])
    return out.getvalue()
