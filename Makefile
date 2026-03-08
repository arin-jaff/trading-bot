.PHONY: install api gui all clean init export-colab

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm || true

init:
	python -c "from src.database.db import init_db; init_db()"

api:
	python run_api.py

gui:
	python run_gui.py

all:
	@echo "Starting API server and GUI..."
	python run_api.py &
	sleep 2
	python run_gui.py

export-colab:
	python export_for_colab.py

clean:
	rm -rf data/trading_bot.db
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
