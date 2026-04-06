import pandas as pd

df = pd.read_csv('../CSV2/med_house_price.csv', encoding='latin-1')  # or 'cp1252' if this doesn't work
indiana_rows = df[df.astype(str).apply(lambda x: x.str.contains('Indiana', case=False, na=False).any(), axis=1)]

indiana_rows.to_csv('../CSV2/IHP_base_indiana_house_prices.csv', index=False)