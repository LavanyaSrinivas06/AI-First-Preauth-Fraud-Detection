import os
import pandas as pd
import numpy as np
import random
from faker import Faker

# Set up Faker and seed
fake = Faker()
np.random.seed(42)

# Create output directory
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

# Config
num_rows = 1000
fraud_ratio = 0.1
num_frauds = int(num_rows * fraud_ratio)

# Helper functions
def random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def random_bool(prob_true=0.5):
    return np.random.rand() < prob_true

# Generate data
data = []
for i in range(num_rows):
    is_fraud = i < num_frauds

    account_created = fake.unix_time(end_datetime='-30d') if is_fraud else fake.unix_time(start_datetime='-2y')
    avs_full_result = random.choice(['N', 'Y']) if is_fraud else 'Y'
    cvv_result = random.choice(['N', 'Y', None]) if is_fraud else 'Y'
    card_country = random.choice(['GB', 'US', 'CN', 'RU']) if is_fraud else 'GB'
    shipping_country = random.choice(['GB', 'US', 'DE']) if is_fraud else 'GB'

    row = {
        "order_id": fake.uuid4(),
        "order_type": "WEB",
        "customer_ip": random_ip(),
        "user_agent": fake.user_agent(),
        "merchant_device_id": fake.uuid4(),
        "forter_token_cookie": fake.md5(),
        "account_id": str(fake.random_number(digits=10)),
        "account_created": account_created,
        "checkout_time": fake.unix_time(),
        "amount": round(random.uniform(10.0, 500.0), 2),
        "currency": "GBP",
        "delivery_type": "PHYSICAL",
        "payment_method": "CREDIT_CARD",
        "card_brand": random.choice(["VISA", "MASTERCARD", "AMEX"]),
        "card_country": card_country,
        "cvv_result": cvv_result,
        "avs_full_result": avs_full_result,
        "3ds_triggered": random_bool(0.7 if is_fraud else 0.2),
        "customer_email": fake.email(),
        "customer_country": "GB",
        "billing_country": "GB",
        "shipping_country": shipping_country,
        "used_saved_data": random_bool(),
        "merchant_id": str(fake.random_number(digits=3)),
        "merchant_name": fake.company(),
        "segment": random.choice(["1-UK", "4-EU", "5-US"]),
        "is_fraud": int(is_fraud)
    }
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV and JSON
csv_file = os.path.join(output_dir, "fake_fraud_data.csv")
json_file = os.path.join(output_dir, "fake_fraud_data.json")

df.to_csv(csv_file, index=False)
df.to_json(json_file, orient="records", lines=True)

print(f"✅ Data generated and saved to:\n- {csv_file}\n- {json_file}")
