customer_data = df_no_duplicate.dropna(subset=['CustomerID']).groupby('CustomerID')['InvoiceDate'].max().reset_index()
customer_data.rename(columns={'InvoiceDate': 'LastPurchaseDate'}, inplace=True)

reference_date = df_no_duplicate['InvoiceDate'].max()

customer_data['DaysSinceLastPurchase'] = (reference_date - customer_data['LastPurchaseDate']).dt.days
customer_data['Churn'] = (customer_data['DaysSinceLastPurchase'] > 90).astype(int)

# Drop columns if they already exist to avoid duplicate _x, _y columns when rerunning
cols_to_drop = [col for col in ['DaysSinceLastPurchase', 'Churn'] if col in df_no_duplicate.columns]
if cols_to_drop:
    df_no_duplicate = df_no_duplicate.drop(columns=cols_to_drop)

df_no_duplicate = pd.merge(df_no_duplicate, customer_data[['CustomerID', 'DaysSinceLastPurchase', 'Churn']], on='CustomerID', how='left')

# Fill NaN values for customers without ID with 0 or a placeholder, then convert to integer
df_no_duplicate['Churn'] = df_no_duplicate['Churn'].fillna(0).astype(int)
df_no_duplicate['DaysSinceLastPurchase'] = df_no_duplicate['DaysSinceLastPurchase'].fillna(-1).astype(int)

df_no_duplicate
