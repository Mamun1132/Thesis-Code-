"""Train the screening model on the cleaned Kurigram data and save it to models/."""
import json

from screening.pipeline import train_and_save

if __name__ == "__main__":
    meta = train_and_save()
    print(json.dumps({k: meta[k] for k in ("training_records", "high_risk_records", "threshold", "scikit_learn")}, indent=2))
