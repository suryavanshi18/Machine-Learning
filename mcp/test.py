import random
from fastmcp import FastMCP
import os
import sqlite3

DB_PATH=os.path.join(os.path.dirname(__file__),"expenses.db")
# print(DB_PATH)
# print(os.path.dirname(__file__))
# print(os.path.basename(__file__))
mcp=FastMCP("ExpenseTracker")

def inti_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
                CREATE TABLE IF NOT EXISTS expenses(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT NOT NULL,
                  amount REAL NOT NULL,
                  category TEXT NOT NULL,
                  subcategory TEXT DEFAULT '',
                  note TEXT DEFAULT ''
                  )
        """)
    inti_db()

@mcp.tool()
def add_expense(date,amount,category,subcategory="",note=""):
    '''Add a new expense entry to the database.'''
    with sqlite3.connect(DB_PATH) as c:
        cur=c.execute(
            """INSERT INTO expenses(date,amount,category,subcategory,note) VALUES (?,?,?,?,?)""",
            (date,amount,category,subcategory,note)
        )
        return {"status":"ok","id":cur.lastrowid}

@mcp.tool()
def list_expenses(start_date,end_date):
    '''List expenses from start date to end date'''
    with sqlite3.connect(DB_PATH) as c:
        cur=c.execute(
            "SELECT * FROM expenses where date BETWEEN ? AND ? ORDER BY id ASC",
            (start_date,end_date)
        )
        cols=[d[0] for d in cur.description]
        return [dict(zip(cols,r)) for r in cur.fetchall()]

@mcp.tool()
def summarize(start_date,end_date,category=None):
    '''Summarize expenses by category within an inclusive date range'''
    with sqlite3.connect(DB_PATH) as c:
        query=(
            """
            SELECT category,SUM(amount) AS total_amount
            FROM expenses
            WHERE date BETWEEN ? AND ?
            """
        )
        params=[start_date,end_date]
        if category:
            query+=" AND category =?"
            params.append(category)
        query+=" GROUP BY category ORDER BY category ASC"
        cur=c.execute(query,params)
        cols=[d[0] for d in cur.description]
        return [dict(zip(cols,r)) for r in cur.fetchall()]
    