import pandas as pd
from datetime import datetime

#Load raw data
df = pd.read_csv('data/synthetic_pharmacy_dataset.csv')
print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop unrecoverable rows (P9999, P1048 placeholders)
df = df.dropna(subset=['product_name', 'category'])
print(f"After dropping missing rows: {len(df)} rows remaining")

# Convert date columns
for col in ['sale_date', 'expiration_date', 'restock_date']:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

df = df.dropna(subset=['sale_date', 'expiration_date', 'restock_date'])

# Fix store_postal_code (float → string) 
df['store_postal_code'] = df['store_postal_code'].fillna(0).astype(int).astype(str)
df['store_postal_code'] = df['store_postal_code'].replace('0', 'Unknown')

# Feature engineering
today = pd.Timestamp(datetime.today().date())

df['sale_month']         = df['sale_date'].dt.month
df['sale_year']          = df['sale_date'].dt.year
df['days_to_expiry']     = (df['expiration_date'] - today).dt.days
df['days_since_restock'] = (today - df['restock_date']).dt.days

#Save clean file
df.to_csv('data/preprocessed_pharmacy.csv', index=False)
print(f"Saved: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nSample:")
print(df[['product_name', 'category', 'units_sold', 'days_to_expiry', 'sale_month']].head(5))
