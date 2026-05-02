import sqlite3
import pandas as pd
try:
    with sqlite3.connect("data/paper_trades.db") as conn:
        df = pd.read_sql("SELECT * FROM paper_trades", conn)
        print(df.head())
except Exception as e:
    print(e)
