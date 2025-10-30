import sqlite3
import pandas as pd

# Check database structure
conn = sqlite3.connect('archive (1)/database.sqlite')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in database:", tables)

# If there are tables, show sample data
for table in tables:
    table_name = table[0]
    print(f"\n--- Table: {table_name} ---")
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
    print(df)
    print(f"Columns: {df.columns.tolist()}")

conn.close()
