"""Standalone fine-tuning script for Mac.

Runs the full Pythia-410M LoRA fine-tuning pipeline locally on your Mac,
then pushes predictions to the Pi via API.

Usage:
    # 1. Copy the latest DB from Pi (get the latest corpus)
    scp arin@<pi-ip>:~/trading-bot/data/trading_bot.db data/trading_bot.db

    # 2. Run fine-tuning (auto-pushes to Pi when done)
    python scripts/fine_tune_mac.py --pi-url http://<pi-ip>:8000

    # Or without auto-push (manual scp later):
    python scripts/fine_tune_mac.py
"""

import json
import os
import re
import sys
import time
import random
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db import init_db, get_session
from src.database.models import Speech, Term, ModelVersion

# ── Configuration (edit these if needed) ──

MODEL_NAME = os.getenv('FINE_TUNE_MODEL', 'EleutherAI/pythia-410m')
LORA_RANK = int(os.getenv('FINE_TUNE_LORA_RANK', '16'))
EPOCHS = int(os.getenv('FINE_TUNE_EPOCHS', '3'))
MAX_LENGTH = int(os.getenv('FINE_TUNE_MAX_LENGTH', '512'))
BATCH_SIZE = int(os.getenv('FINE_TUNE_BATCH_SIZE', '1'))
GRAD_ACCUM = int(os.getenv('FINE_TUNE_GRAD_ACCUM', '8'))
LEARNING_RATE = float(os.getenv('FINE_TUNE_LR', '5e-4'))
MC_SIMS = int(os.getenv('FINE_TUNE_MC_SIMS', '200'))

ADAPTERS_DIR = os.path.join('data', 'models', 'gpt2_lora')
PREDICTIONS_DIR = os.path.join('data', 'predictions')

SCENARIO_PROMPTS = {
    'rally': 'Trump rally speech:',
    'press_conference': 'Press conference:',
    'chopper_talk': 'Trump remarks to reporters:',
    'fox_interview': 'Fox News interview:',
    'social_media': 'Trump post:',
}

SCENARIO_WORD_COUNTS = {
    'rally': 5000,
    'press_conference': 2000,
    'chopper_talk': 800,
    'fox_interview': 1500,
    'social_media': 300,
}

DEFAULT_SCENARIO_WEIGHTS = {
    'rally': 0.40,
    'press_conference': 0.25,
    'chopper_talk': 0.10,
    'fox_interview': 0.15,
    'social_media': 0.10,
}


def detect_lora_targets(model) -> list[str]:
    """Auto-detect LoRA target module names."""
    module_names = {name for name, _ in model.named_modules()}
    for candidate in [
        ['query_key_value'],
        ['c_attn'],
        ['q_proj', 'v_proj'],
    ]:
        if all(any(c in name for name in module_names) for c in candidate):
            return candidate
    raise ValueError("Could not detect LoRA target modules")


def load_corpus() -> list[str]:
    """Load training texts from DB."""
    with get_session() as session:
        speeches = session.query(Speech).filter(
            Speech.transcript.isnot(None),
            Speech.is_processed == True,
            Speech.word_count >= 50,
        ).all()
        return [s.transcript for s in speeches if s.transcript]


def load_terms() -> list[str]:
    """Load tracked terms from DB."""
    with get_session() as session:
        terms = session.query(Term).all()
        return [t.normalized_term for t in terms]


def train(corpus_texts: list[str]) -> dict:
    """Fine-tune Pythia-410M with LoRA."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    os.makedirs(ADAPTERS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Fine-Tuning {MODEL_NAME}")
    print(f"  Corpus: {len(corpus_texts)} texts")
    print(f"  LoRA rank: {LORA_RANK}, Epochs: {EPOCHS}")
    print(f"{'='*60}\n")

    # Tokenize
    print("[1/4] Tokenizing corpus...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_input_ids = []
    for text in corpus_texts:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        for i in range(0, len(encoded), MAX_LENGTH):
            chunk = encoded[i:i + MAX_LENGTH]
            if len(chunk) >= 64:
                chunk = chunk + [tokenizer.eos_token_id] * (MAX_LENGTH - len(chunk))
                all_input_ids.append(chunk)

    print(f"  {len(all_input_ids)} training sequences of length {MAX_LENGTH}")
    random.shuffle(all_input_ids)

    # Load model + LoRA
    print(f"[2/4] Loading {MODEL_NAME} + LoRA...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    targets = detect_lora_targets(model)
    print(f"  LoRA targets: {targets}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {trainable:,} trainable / {total_params:,} total ({100*trainable/total_params:.2f}%)")

    # Training loop
    print(f"[3/4] Training ({EPOCHS} epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = (len(all_input_ids) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
    global_step = 0
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(EPOCHS):
        random.shuffle(all_input_ids)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(all_input_ids), BATCH_SIZE):
            batch = all_input_ids[i:i + BATCH_SIZE]
            input_ids = torch.tensor(batch, dtype=torch.long)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            epoch_loss += outputs.loss.item()
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = epoch_loss / n_batches
                    elapsed = time.time() - start_time
                    pct = global_step / max(1, total_steps) * 100
                    eta = (elapsed / max(1, global_step)) * (total_steps - global_step)
                    print(f"  Step {global_step}/{total_steps} ({pct:.0f}%) | "
                          f"loss={avg:.4f} | "
                          f"elapsed={elapsed/60:.1f}m | "
                          f"ETA={eta/60:.1f}m")

        avg_loss = epoch_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"  Epoch {epoch+1}/{EPOCHS} complete — avg_loss={avg_loss:.4f}")

    # Save adapter
    print("[4/4] Saving adapter...")
    adapter_path = os.path.join(ADAPTERS_DIR, 'adapter_latest')
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    duration = time.time() - start_time
    print(f"\n  Saved to {adapter_path}")
    print(f"  Training time: {duration/60:.1f} minutes")
    print(f"  Final loss: {avg_loss:.4f}, Best loss: {best_loss:.4f}")

    return {
        'adapter_path': adapter_path,
        'corpus_size': len(corpus_texts),
        'training_seconds': round(duration, 2),
        'final_loss': round(avg_loss, 4),
        'best_loss': round(best_loss, 4),
        'total_steps': global_step,
    }


def run_monte_carlo(terms: list[str], adapter_path: str) -> dict:
    """Run Monte Carlo simulations with the fine-tuned model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"\n{'='*60}")
    print(f"  Pythia Monte Carlo — {MC_SIMS} simulations")
    print(f"  Terms: {len(terms)}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading fine-tuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Term patterns
    term_patterns = {}
    for term in terms:
        normalized = term.lower().strip()
        pattern = re.compile(r'\b' + re.escape(normalized) + r'\b', re.IGNORECASE)
        term_patterns[normalized] = pattern

    term_stats = {t: {'speeches_containing': 0, 'total_mentions': 0}
                  for t in term_patterns}

    # Simulations
    scenario_sims = {s: max(1, int(MC_SIMS * w))
                     for s, w in DEFAULT_SCENARIO_WEIGHTS.items()}
    total_sims = sum(scenario_sims.values())
    sim_count = 0
    total_words = 0
    start_time = time.time()

    for scenario, n_sims in scenario_sims.items():
        word_count = min(500, SCENARIO_WORD_COUNTS.get(scenario, 500))
        prompt = SCENARIO_PROMPTS.get(scenario, 'Trump speech:')

        for i in range(n_sims):
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=word_count * 2,
                    temperature=1.0,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            total_words += len(text.split())

            for term_key, pattern in term_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    term_stats[term_key]['speeches_containing'] += 1
                    term_stats[term_key]['total_mentions'] += len(matches)

            sim_count += 1
            if sim_count % 10 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / sim_count) * (total_sims - sim_count)
                print(f"  Sim {sim_count}/{total_sims} | "
                      f"elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m")

    # Build predictions
    avg_words = total_words / total_sims if total_sims > 0 else 500
    predictions = []
    for term_key, stats in term_stats.items():
        prob = stats['speeches_containing'] / total_sims if total_sims > 0 else 0
        avg_mentions = stats['total_mentions'] / total_sims if total_sims > 0 else 0
        predictions.append({
            'term': term_key,
            'probability': round(prob, 4),
            'speeches_containing': stats['speeches_containing'],
            'total_mentions': stats['total_mentions'],
            'avg_mentions_per_speech': round(avg_mentions, 4),
            'model_name': 'pythia_monte_carlo',
            'confidence': min(1.0, total_sims / 200),
        })

    duration = time.time() - start_time
    print(f"\n  {total_sims} simulations in {duration/60:.1f} minutes")

    return {
        'term_predictions': predictions,
        'simulation_params': {
            'num_simulations': total_sims,
            'scenario_weights': DEFAULT_SCENARIO_WEIGHTS,
            'avg_words_per_speech': round(avg_words, 0),
            'model_type': 'pythia_lora',
            'model_name': MODEL_NAME,
        },
        'generated_at': datetime.now().isoformat(),
    }


def push_to_pi(pred_path: str, pi_url: str) -> bool:
    """Push predictions to the Pi via its API."""
    import requests as req

    upload_url = f'{pi_url.rstrip("/")}/api/fine-tune/upload-predictions'
    print(f"\nPushing predictions to Pi at {upload_url}...")

    try:
        with open(pred_path) as f:
            data = json.load(f)
        resp = req.post(upload_url, json=data, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            print(f"  Pushed {result.get('predictions_saved', '?')} predictions to Pi")
            return True
        else:
            print(f"  Push failed: {resp.status_code} — {resp.text}")
            return False
    except Exception as e:
        print(f"  Push failed: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune Pythia-410M on Mac')
    parser.add_argument('--pi-url', type=str, default=None,
                        help='Pi API URL (e.g. http://192.168.0.100:8000). Auto-pushes predictions when set.')
    args = parser.parse_args()

    init_db()

    # Load corpus
    print("Loading corpus from DB...")
    corpus = load_corpus()
    if not corpus:
        print("ERROR: No training texts found. Copy the DB from Pi first:")
        print("  scp arin@<pi-ip>:~/trading-bot/data/trading_bot.db data/trading_bot.db")
        sys.exit(1)
    print(f"  {len(corpus)} texts loaded")

    # Load terms
    terms = load_terms()
    print(f"  {len(terms)} terms loaded")

    # Fine-tune
    result = train(corpus)

    # Monte Carlo
    pred_path = None
    if terms:
        predictions = run_monte_carlo(terms, result['adapter_path'])

        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        pred_path = os.path.join(PREDICTIONS_DIR, 'predictions_pythia.json')
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to {pred_path}")
    else:
        print("\nNo terms found — skipping Monte Carlo")

    # Push to Pi if URL provided
    if pred_path and args.pi_url:
        push_to_pi(pred_path, args.pi_url)
    elif pred_path:
        print(f"\nTo push to Pi, either:")
        print(f"  python scripts/fine_tune_mac.py --pi-url http://<pi-ip>:8000")
        print(f"  OR: scp {pred_path} arin@<pi-ip>:~/trading-bot/data/predictions/")

    print(f"\n{'='*60}")
    print("  DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
