import pandas as pd
import numpy as np
from pathlib import Path

def run_etl(input_path='data/insurance_claims.csv', output_path='data/clean_insurance_claims.csv'):
    df = pd.read_csv(input_path)
    # Basic cleaning
    df = df.drop_duplicates(subset=['policy_id'])
    df['gender'] = df['gender'].fillna('M')
    df['region'] = df['region'].fillna('North')
    # Feature engineering
    df['claim_amount_log'] = df['claim_amount'].apply(lambda x: np.log1p(x))
    # Save cleaned file
    Path(output_path).parents[0].mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Cleaned data saved to {output_path}')

if __name__ == '__main__':
    run_etl()
