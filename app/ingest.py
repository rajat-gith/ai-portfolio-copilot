"""CLI entry point. Kept at this path so `python -m app.ingest` keeps working;
the actual pipeline implementation lives in app/ingestion/pipeline.py.
"""
from .ingestion.pipeline import main

if __name__ == "__main__":
    main()
