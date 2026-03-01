from locust import HttpUser, task, between
import random
import uuid
import time

class PreAuthCheckoutUser(HttpUser):
    host = "http://127.0.0.1:8010"
    wait_time = between(0.05, 0.3)  # realistic checkout spacing

    def _base_payload(self):
        return {
            "transaction_id": str(uuid.uuid4()),
            "amount": round(random.uniform(5, 800), 2),
            "currency": "EUR",

            # device / browser
            "device_os": random.choice(["Android", "iOS", "Windows", "MacOS"]),
            "browser": random.choice(["Chrome", "Safari", "Firefox"]),
            "is_new_device": random.random() < 0.1,

            # geo / network
            "ip_country": random.choice(["DE", "FR", "NL", "IT"]),
            "billing_country": "DE",
            "shipping_country": "DE",
            "is_proxy_vpn": random.random() < 0.05,
            "ip_reputation": round(random.uniform(0.0, 1.0), 3),

            # behavioral / velocity
            "txn_count_5m": random.randint(0, 5),
            "txn_count_30m": random.randint(0, 10),
            "txn_count_60m": random.randint(0, 20),

            # account
            "account_age_days": random.randint(0, 2000),
            "token_age_days": random.randint(0, 1000),

            # derived flags
            "night_txn": random.random() < 0.2,
            "weekend_txn": random.random() < 0.3,
        }

    @task(7)
    def legit_checkout(self):
        payload = self._base_payload()
        self.client.post("/preauth/decision", json=payload)

    @task(2)
    def gray_zone_checkout(self):
        payload = self._base_payload()
        payload["amount"] = random.uniform(300, 900)
        payload["txn_count_5m"] = random.randint(5, 10)
        payload["is_new_device"] = True
        self.client.post("/preauth/decision", json=payload)

    @task(1)
    def fraud_like_checkout(self):
        payload = self._base_payload()
        payload.update({
            "is_proxy_vpn": True,
            "ip_reputation": random.uniform(0.7, 1.0),
            "txn_count_5m": random.randint(15, 30),
            "account_age_days": random.randint(0, 10),
            "amount": random.uniform(500, 1200),
        })
        self.client.post("/preauth/decision", json=payload)
