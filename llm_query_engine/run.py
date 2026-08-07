import sys
import os

# Dynamically find the folder containing query_engine.py and add it to path
for root, dirs, files in os.walk("."):
    if "query_engine.py" in files:
        sys.path.insert(0, os.path.abspath(root))
        break

# Import and test QueryEngine
from query_engine import QueryEngine

if __name__ == "__main__":
    engine = QueryEngine()
    result = engine.answer("Show me temperature in the North Atlantic")
    print("SUCCESS! QueryEngine executed successfully:")
    print(result.model_dump_json(indent=2))