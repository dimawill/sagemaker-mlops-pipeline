"""Small, deterministic illustrative daily demand dataset; no external download."""
import csv
import io
import math
import random
from datetime import date, timedelta

FIELDS = ('day', 'weekday', 'promo', 'price', 'demand')


def generate_rows(count=240, seed=17):
    if count < 20:
        raise ValueError('At least 20 days required')
    rng = random.Random(seed)
    start = date(2025, 1, 1)
    rows = []
    for i in range(count):
        day = start + timedelta(days=i)
        weekday = day.weekday()
        promo = int(i % 11 < 3)
        price = round(8 + (i % 7) * 0.5, 2)
        demand = round(95 + 19 * (weekday >= 5) + 24 * promo - 3 * price
                       + 3 * math.sin(i / 12) + rng.uniform(-3, 3), 3)
        rows.append({'day': day.isoformat(), 'weekday': weekday, 'promo': promo,
                     'price': price, 'demand': demand})
    return rows


def dataset_csv(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()
