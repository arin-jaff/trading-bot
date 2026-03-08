"""Export data for Colab training.

Run this locally to prepare the training dataset, then upload to Google Drive.

Usage:
    python export_for_colab.py

Output files (in data/exports/):
    - trump_speeches.jsonl      -> Upload to Drive for fine-tuning notebook
    - term_context.json         -> Upload to Drive for Monte Carlo notebook
    - event_speech_pairs.json   -> Upload to Drive for RAG context
"""

from src.database.db import init_db
from src.ml.data_exporter import DataExporter


def main():
    init_db()
    exporter = DataExporter()

    print("=" * 60)
    print("Exporting data for Colab training")
    print("=" * 60)

    # 1. Training corpus
    print("\n[1/3] Exporting speech corpus for fine-tuning...")
    stats = exporter.export_training_corpus(format='jsonl')
    print(f"  -> {stats}")

    # 2. Term context for RAG
    print("\n[2/3] Exporting term context for RAG pipeline...")
    stats = exporter.export_term_context()
    print(f"  -> {stats}")

    # 3. Event-speech pairs
    print("\n[3/3] Exporting event-speech pairs...")
    stats = exporter.export_event_history()
    print(f"  -> {stats}")

    print("\n" + "=" * 60)
    print("Done! Upload these files to Google Drive:")
    print("  data/exports/trump_speeches.jsonl")
    print("  data/exports/term_context.json")
    print("  data/exports/event_speech_pairs.json")
    print("")
    print("Place them in: Google Drive > trump_trading_bot > data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
