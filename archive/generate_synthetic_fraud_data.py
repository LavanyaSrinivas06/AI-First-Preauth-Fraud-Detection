import os
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Set up Faker and seed
fake = Faker()
np.random.seed(42)

# Create output directory
output_dir = os.path.join("data", "raw")
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

def get_currency(region):
    return {
        "uk": "GBP",
        "eu": "EUR",
        "us": "USD",
        "asia": random.choice(["INR", "SGD", "CNY"])
    }[region]

def get_checkout_time():
    now = datetime.now()
    random_time = now - timedelta(minutes=random.randint(0, 60*24*30))
    return int(random_time.timestamp()), random_time.hour

# Generate data
data = []
device_tokens_seen = set()

for i in range(num_rows):
    is_fraud = i < num_frauds
    segment = random.choice(["uk", "eu", "us", "asia"])
    currency = get_currency(segment)
    account_created = fake.date_time_between(start_date="-2y", end_date="-1d")
    account_created_ts = int(account_created.timestamp())
    checkout_ts, checkout_hour = get_checkout_time()
    amount = round(random.uniform(10.0, 500.0), 2)
    avs_full_result = random.choice(['N', 'Y']) if is_fraud else 'Y'
    cvv_result = random.choice(['N', 'Y', None]) if is_fraud else 'Y'
    card_country = random.choice(['GB', 'US', 'CN', 'RU']) if is_fraud else 'GB'
    shipping_country = random.choice(['GB', 'US', 'DE']) if is_fraud else 'GB'
    device_token = fake.md5()
    device_is_new = device_token not in device_tokens_seen
    device_tokens_seen.add(device_token)
    attempts_today = random.randint(1, 6 if is_fraud else 3)

    # Fraud flags
    flag_new_high_value_acc = (datetime.now() - account_created).days < 30 and amount > 300
    flag_address_mismatch = shipping_country != "GB"
    flag_avs_failed = avs_full_result == 'N'
    flag_cvv_failed = cvv_result == 'N'
    flag_odd_hour = checkout_hour < 5
    flag_rapid_retries = attempts_today > 3
    flag_guest_new_device = random_bool(0.4) and device_is_new
    flag_express_high_value = amount > 250 and random.choice(["Standard", "Express", "Next-Day"]) in ["Express", "Next-Day"]
    flag_geo_mismatch = card_country != "GB"

    row = {
        "order_id": fake.uuid4(),
        "order_type": "WEB",
        "customer_ip": random_ip(),
        "user_agent": fake.user_agent(),
        "merchant_device_id": fake.uuid4(),
        "device_session_token": device_token,
        "account_id": str(fake.random_number(digits=10)),
        "account_created": account_created_ts,
        "checkout_time": checkout_ts,
        "amount": amount,
        "currency": currency,
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
        "merchant_name": "ShopVerse",
        "region_segment": segment,
        "cart_total_items": random.randint(1, 5),
        "browser_language": random.choice(["en-GB", "en-US", "fr-FR", "de-DE"]),
        "coupon_used": random_bool(0.3),
        "guest_checkout": random_bool(0.4),
        "shipping_speed": random.choice(["Standard", "Express", "Next-Day"]),
        "attempts_today": attempts_today,
        "device_type": random.choice(["Desktop", "Mobile", "Tablet"]),
        "browser_name": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
        "flag_new_high_value_acc": int(flag_new_high_value_acc),
        "flag_address_mismatch": int(flag_address_mismatch),
        "flag_avs_failed": int(flag_avs_failed),
        "flag_cvv_failed": int(flag_cvv_failed),
        "flag_odd_hour": int(flag_odd_hour),
        "flag_rapid_retries": int(flag_rapid_retries),
        "flag_guest_new_device": int(flag_guest_new_device),
        "flag_express_high_value": int(flag_express_high_value),
        "flag_geo_mismatch": int(flag_geo_mismatch),
        "is_fraud": int(is_fraud)
    }

    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV and JSON
csv_file = os.path.join(output_dir, "ecom_synthetic_fraud_dataset_v1.csv")
json_file = os.path.join(output_dir, "ecom_synthetic_fraud_dataset_v1.json")

df.to_csv(csv_file, index=False)
df.to_json(json_file, orient="records", lines=True)

print(f"✅ Data generated and saved to:\n- {csv_file}\n- {json_file}")
