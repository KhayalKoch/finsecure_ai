import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_transactions(n_rows=100000, fraud_ratio=0.015): # Set base fraud ratio to 1.5%
    np.random.seed(42)
    random.seed(42)

    # Ensure the directory exists
    os.makedirs("data/raw", exist_ok=True)

    transactions = []

    for i in range(n_rows):
        transaction_id = i + 1
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 365),
            seconds=random.randint(0, 86400)
        )

        # Generate realistic transaction amounts using lognormal distribution
        amount = round(np.random.lognormal(mean=4.5, sigma=0.8), 2)

        sender_id = f"user_{random.randint(1, 5000)}"
        receiver_id = f"user_{random.randint(5001, 10000)}"
        location = random.choice(["DE", "AT", "CH", "UK", "FR", "US", "TR"])
        transaction_type = random.choice(["transfer", "withdrawal", "payment"])
        hour = timestamp.hour
        is_international = 1 if location != "DE" else 0

        # --- REALISTIC FRAUD LOGIC ---
        is_fraud = 0
        
        # 1. Base probability for random fraud cases
        if random.random() < fraud_ratio:
            is_fraud = 1
            # Some fraud cases involve very high amounts
            if random.random() < 0.3:
                amount = round(random.uniform(5000, 12000), 2)

        # 2. Heuristic 1: High amount during late-night hours (Suspicious)
        # If transaction is between 00:00-05:00 and amount > 4000
        if hour < 6 and amount > 4000:
            if random.random() < 0.2: # 20% chance to be flagged as fraud
                is_fraud = 1

        # 3. Heuristic 2: Large international transfers
        if is_international == 1 and amount > 7000:
            if random.random() < 0.15: # 15% probability
                is_fraud = 1

        transactions.append([
            transaction_id, timestamp, amount, sender_id, receiver_id, 
            location, transaction_type, hour, is_international, is_fraud
        ])

        # 4. Multi-step fraud chain (Simulating money laundering)
        if is_fraud == 1 and random.random() < 0.2: # 20% of frauds form chains
            chain_length = random.randint(1, 3)
            prev_receiver = receiver_id
            for j in range(chain_length):
                new_receiver = f"user_{random.randint(5001, 10000)}"
                new_amount = round(amount * random.uniform(0.7, 0.95), 2)
                new_timestamp = timestamp + timedelta(minutes=random.randint(5, 60))
                
                transactions.append([
                    n_rows + len(transactions) + 1, new_timestamp, new_amount, 
                    prev_receiver, new_receiver, location, transaction_type, 
                    new_timestamp.hour, is_international, 1
                ])
                prev_receiver = new_receiver

    # Create the final DataFrame
    df = pd.DataFrame(transactions, columns=[
        "transaction_id", "timestamp", "amount", "sender_id", "receiver_id",
        "location", "transaction_type", "hour", "is_international", "fraud_label"
    ])

    return df

if __name__ == "__main__":
    df = generate_transactions(n_rows=100000)
    df.to_csv("data/raw/transactions.csv", index=False)
    print(f"Dataset generated successfully! Total rows: {len(df)}")
    print(f"Final Fraud Ratio: {df['fraud_label'].mean():.2%}")