import sqlite3
import os

db_path = "../LemGendaryDatasets/LemGendizedNimaAuthenticityLarge/manifold_registry.db"

if not os.path.exists(db_path):
    print(f"File not found: {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("--- SOURCE SUMMARY ---")
    cursor.execute("SELECT source, COUNT(*) FROM registry GROUP BY source")
    rows = cursor.fetchall()
    for row in rows:
        print(f"Source: {row[0]} | Count: {row[1]}")
    
    print("\n--- DUPLICATE HASH CHECK ---")
    cursor.execute("SELECT hash, COUNT(*) as c FROM registry WHERE hash IS NOT NULL GROUP BY hash HAVING c > 1 LIMIT 10")
    dupes = cursor.fetchall()
    if dupes:
        print(f"Found {len(dupes)} duplicate hashes (first 10 shown):")
        for d in dupes:
            print(f"Hash: {d[0]} | Count: {d[1]}")
    else:
        print("No duplicate hashes found.")

    conn.close()
