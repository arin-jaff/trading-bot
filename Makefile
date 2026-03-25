.PHONY: install install-pi install-finetune api clean init deploy-pi import-twitter

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm || true

install-pi:
	pip install -r requirements-pi.txt
	pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch

init:
	python -c "from src.database.db import init_db; init_db()"

api:
	python run_api.py

deploy-pi:
	bash deploy/setup-pi.sh

install-finetune:
	pip install torch transformers peft datasets accelerate

import-twitter:
	python -c "from src.scraper.social_media_importer import SocialMediaImporter; i=SocialMediaImporter(); print(i.import_twitter_archive())"

clean:
	rm -rf data/trading_bot.db
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
