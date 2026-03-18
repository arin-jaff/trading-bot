"""LLM fine-tuning with LoRA — RUNS ON MAC ONLY, NOT ON PI.

Used by: scripts/fine_tune_mac.py
NOT imported by any Pi code path. PyTorch does not run on Raspberry Pi ARM.

Default model: Llama-3.2-1B (Meta). All torch/transformers imports are lazy.
"""

import os
import json
import time
import math
import re
import random
import threading
from datetime import datetime
from typing import Optional
from loguru import logger

from ..database.db import get_session
from ..database.models import Speech, ModelVersion
from ..config import config


# Where to save LoRA adapters and checkpoints
ADAPTERS_DIR = os.path.join('data', 'models', 'gpt2_lora')
CHECKPOINTS_DIR = os.path.join('data', 'models', 'gpt2_checkpoints')

# Scenario prompts for generation
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


def _detect_lora_targets(model) -> list[str]:
    """Auto-detect the right LoRA target module names for any causal LM.

    Different architectures use different names for their attention projections:
    - GPT-2: 'c_attn' (fused QKV)
    - Pythia / GPT-NeoX: 'query_key_value' (fused QKV)
    - Llama / Qwen / Phi: 'q_proj', 'v_proj' (separate)
    - OPT: 'q_proj', 'v_proj'
    """
    module_names = {name for name, _ in model.named_modules()}

    # Check candidates in order of specificity
    for candidate in [
        ['query_key_value'],          # Pythia, GPT-NeoX, BLOOM
        ['c_attn'],                   # GPT-2
        ['q_proj', 'v_proj'],         # Llama, Qwen, OPT, Phi
        ['qkv_proj'],                 # Some merged QKV variants
    ]:
        if all(any(c in name for name in module_names) for c in candidate):
            return candidate

    # Fallback: find any linear layers with 'attn' in the name
    attn_linears = [
        name.split('.')[-1] for name, mod in model.named_modules()
        if 'attn' in name.lower() and hasattr(mod, 'weight')
        and mod.weight.dim() == 2
    ]
    if attn_linears:
        return list(set(attn_linears))[:2]

    raise ValueError("Could not detect LoRA target modules for this model architecture")


class GPT2FineTuner:
    """Fine-tune GPT-2 Small with LoRA on the Pi 4 CPU."""

    def __init__(self):
        os.makedirs(ADAPTERS_DIR, exist_ok=True)
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

        self._status = {
            'state': 'idle',  # idle, training, complete, error, stopped
            'stage': '',
            'progress': 0.0,
            'current_epoch': 0,
            'total_epochs': config.fine_tune_epochs,
            'current_step': 0,
            'total_steps': 0,
            'loss': None,
            'best_loss': None,
            'tokens_per_second': None,
            'eta_seconds': None,
            'memory_mb': None,
            'error': None,
        }
        self._loss_history = []  # [(step, loss), ...]
        self._stop_requested = False
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def get_status(self) -> dict:
        return self._status.copy()

    def get_loss_history(self) -> list[dict]:
        return [{'step': s, 'loss': l} for s, l in self._loss_history]

    def train(self) -> Optional[dict]:
        """Run full fine-tuning loop.

        Loads corpus from DB, tokenizes, trains GPT-2 + LoRA with:
        - os.nice(19) for lowest CPU priority
        - batch_size=1, gradient_accumulation=configurable
        - Progress updates every 10 steps
        - Checkpoints every 500 steps
        - Graceful stop via _stop_requested

        Returns result dict or None on failure.
        """
        if not self._lock.acquire(blocking=False):
            return {'status': 'already_running'}

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError as e:
            self._status.update(state='error', error=f'Missing dependency: {e}. Run: make install-finetune')
            self._lock.release()
            return None

        try:
            # Set lowest CPU priority
            try:
                os.nice(19)
            except (OSError, AttributeError):
                pass

            self._stop_requested = False
            self._loss_history = []
            start_time = time.time()

            self._status.update(
                state='training', stage='Loading corpus',
                progress=0.0, error=None,
                total_epochs=config.fine_tune_epochs,
            )

            # Phase 1: Load corpus
            corpus_texts = self._load_corpus()
            if not corpus_texts:
                raise RuntimeError("No training texts found in DB")

            logger.info(f"Fine-tuner: loaded {len(corpus_texts)} texts")

            # Phase 2: Tokenize
            self._status.update(stage='Tokenizing', progress=0.05)
            model_name = config.fine_tune_model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            max_length = config.fine_tune_max_length
            all_input_ids = []
            for text in corpus_texts:
                encoded = tokenizer.encode(text, add_special_tokens=True)
                # Chunk into max_length sequences
                for i in range(0, len(encoded), max_length):
                    chunk = encoded[i:i + max_length]
                    if len(chunk) >= 64:  # skip very short chunks
                        # Pad to max_length
                        chunk = chunk + [tokenizer.eos_token_id] * (max_length - len(chunk))
                        all_input_ids.append(chunk)

            if not all_input_ids:
                raise RuntimeError("Tokenization produced no training sequences")

            logger.info(f"Fine-tuner: {len(all_input_ids)} training sequences of length {max_length}")
            random.shuffle(all_input_ids)

            # Phase 3: Load model + LoRA
            self._status.update(stage=f'Loading {model_name} + LoRA', progress=0.1)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # Auto-detect LoRA target modules based on model architecture
            target_modules = _detect_lora_targets(model)
            logger.info(f"Fine-tuner: detected LoRA targets: {target_modules}")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.fine_tune_lora_rank,
                lora_alpha=config.fine_tune_lora_rank * 2,
                lora_dropout=0.05,
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.train()

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Fine-tuner: {trainable:,} trainable params / {total_params:,} total ({100*trainable/total_params:.2f}%)")

            # Phase 4: Training loop
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.fine_tune_learning_rate,
            )

            batch_size = config.fine_tune_batch_size
            grad_accum = config.fine_tune_grad_accum
            epochs = config.fine_tune_epochs
            total_steps = (len(all_input_ids) * epochs) // (batch_size * grad_accum)
            self._status.update(total_steps=total_steps)

            global_step = 0
            best_loss = float('inf')
            checkpoint_interval = 500

            # Try to resume from checkpoint
            resume_info = self._load_checkpoint()
            start_epoch = 0
            if resume_info:
                start_epoch = resume_info.get('epoch', 0)
                global_step = resume_info.get('step', 0)
                best_loss = resume_info.get('best_loss', float('inf'))
                self._loss_history = resume_info.get('loss_history', [])
                logger.info(f"Resuming from checkpoint: epoch {start_epoch}, step {global_step}")

            for epoch in range(start_epoch, epochs):
                self._status.update(
                    current_epoch=epoch + 1,
                    stage=f'Epoch {epoch + 1}/{epochs}',
                )

                random.shuffle(all_input_ids)
                epoch_loss = 0.0
                epoch_tokens = 0
                step_start = time.time()

                optimizer.zero_grad()
                for i in range(0, len(all_input_ids), batch_size):
                    if self._stop_requested:
                        self._save_checkpoint(epoch, global_step, best_loss)
                        self._status.update(state='stopped', stage='Training stopped by user')
                        logger.info(f"Fine-tuning stopped at epoch {epoch+1}, step {global_step}")
                        return {'status': 'stopped', 'step': global_step, 'best_loss': best_loss}

                    batch = all_input_ids[i:i + batch_size]
                    input_ids = torch.tensor(batch, dtype=torch.long)
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss / grad_accum
                    loss.backward()

                    epoch_loss += outputs.loss.item()
                    epoch_tokens += input_ids.numel()

                    if (i // batch_size + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        # Progress update every 10 steps
                        if global_step % 10 == 0:
                            elapsed = time.time() - step_start
                            tok_per_sec = epoch_tokens / max(0.1, elapsed)
                            current_loss = epoch_loss / max(1, (i // batch_size + 1))

                            if current_loss < best_loss:
                                best_loss = current_loss

                            self._loss_history.append((global_step, round(current_loss, 4)))

                            # Memory
                            mem_mb = None
                            try:
                                import psutil
                                proc = psutil.Process()
                                mem_mb = round(proc.memory_info().rss / (1024 * 1024))
                            except Exception:
                                pass

                            progress = global_step / max(1, total_steps)
                            eta = (time.time() - start_time) / max(0.001, progress) * (1 - progress) if progress > 0.01 else None

                            self._status.update(
                                current_step=global_step,
                                loss=round(current_loss, 4),
                                best_loss=round(best_loss, 4),
                                tokens_per_second=round(tok_per_sec, 1),
                                memory_mb=mem_mb,
                                progress=progress,
                                eta_seconds=round(eta) if eta else None,
                            )

                        # Checkpoint every N steps
                        if global_step % checkpoint_interval == 0:
                            self._save_checkpoint(epoch, global_step, best_loss)
                            logger.info(f"Checkpoint at step {global_step}, loss={epoch_loss / max(1, (i // batch_size + 1)):.4f}")

                avg_epoch_loss = epoch_loss / max(1, len(all_input_ids) // batch_size)
                logger.info(f"Epoch {epoch+1}/{epochs} complete: avg_loss={avg_epoch_loss:.4f}")

            # Phase 5: Save final adapter
            self._status.update(stage='Saving adapter', progress=0.95)
            adapter_path = os.path.join(ADAPTERS_DIR, 'adapter_latest')
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            training_duration = time.time() - start_time

            # Create ModelVersion record
            version_str = self._next_version()
            with get_session() as session:
                mv = ModelVersion(
                    version=version_str,
                    model_type='gpt2_lora',
                    corpus_size=len(corpus_texts),
                    corpus_word_count=sum(len(t.split()) for t in corpus_texts),
                    training_duration_seconds=round(training_duration, 2),
                    artifact_path=adapter_path,
                    is_active=False,  # Don't deactivate Markov model
                    metrics={
                        'final_loss': round(avg_epoch_loss, 4),
                        'best_loss': round(best_loss, 4),
                        'total_steps': global_step,
                        'trainable_params': trainable,
                        'lora_rank': config.fine_tune_lora_rank,
                        'training_sequences': len(all_input_ids),
                    },
                    notes=f'GPT-2 LoRA fine-tune, {global_step} steps, loss={avg_epoch_loss:.4f}',
                )
                session.add(mv)

            self._status.update(
                state='complete', stage=f'Training complete — v{version_str}',
                progress=1.0,
            )

            result = {
                'status': 'complete',
                'version': version_str,
                'corpus_size': len(corpus_texts),
                'training_seconds': round(training_duration, 2),
                'final_loss': round(avg_epoch_loss, 4),
                'total_steps': global_step,
            }
            logger.info(f"GPT-2 fine-tuning complete: {result}")
            return result

        except Exception as e:
            self._status.update(state='error', error=str(e))
            logger.error(f"GPT-2 fine-tuning failed: {e}")
            return None

        finally:
            self._lock.release()
            # Free memory
            self._model = None
            self._tokenizer = None
            try:
                import gc
                gc.collect()
            except Exception:
                pass

    def stop_training(self):
        """Request graceful stop of training. Saves checkpoint."""
        self._stop_requested = True
        logger.info("Fine-tuning stop requested")

    def generate_speech(self, scenario_type: str = 'rally',
                        word_count: Optional[int] = None,
                        temperature: float = 1.0,
                        topic_bias: Optional[str] = None) -> str:
        """Generate a simulated Trump speech using the fine-tuned GPT-2 model."""
        self._load_model()
        if not self._model or not self._tokenizer:
            return ""

        target_words = word_count or SCENARIO_WORD_COUNTS.get(scenario_type, 3000)
        prompt = SCENARIO_PROMPTS.get(scenario_type, 'Trump speech:')
        if topic_bias:
            prompt = f"{prompt} {topic_bias}"

        return self._generate_text(prompt, target_words, temperature)

    def generate_from_prompt(self, prompt: str,
                             word_count: int = 500,
                             temperature: float = 1.0,
                             qa_mode: bool = False) -> str:
        """Generate text from a user prompt, matching MarkovChainTrainer API."""
        self._load_model()
        if not self._model or not self._tokenizer:
            return ""

        if qa_mode:
            prompt = f"Question: {prompt}\nTrump's answer:"

        return self._generate_text(prompt, word_count, temperature)

    def run_monte_carlo(self, terms: list[str],
                        num_simulations: Optional[int] = None,
                        scenario_weights: Optional[dict] = None) -> dict:
        """Run Monte Carlo simulations using GPT-2 for term prediction.

        Returns predictions in the same format as MarkovChainTrainer.run_monte_carlo().
        Uses fewer simulations by default (GPT-2 on CPU is slow).
        """
        self._load_model()
        if not self._model or not self._tokenizer:
            return {}

        num_sims = num_simulations or config.fine_tune_mc_sims
        weights = scenario_weights or DEFAULT_SCENARIO_WEIGHTS

        # Normalize term patterns
        term_patterns = {}
        for term in terms:
            normalized = term.lower().strip()
            pattern = re.compile(r'\b' + re.escape(normalized) + r'\b', re.IGNORECASE)
            term_patterns[normalized] = pattern

        term_stats = {t: {'speeches_containing': 0, 'total_mentions': 0}
                      for t in term_patterns}

        # Per-scenario simulation counts
        scenario_sims = {}
        for scenario, weight in weights.items():
            scenario_sims[scenario] = max(1, int(num_sims * weight))

        start_time = time.time()
        sim_count = 0
        total_sims = sum(scenario_sims.values())
        total_words_generated = 0

        for scenario, n_sims in scenario_sims.items():
            # Use shorter sims for GPT-2 on CPU + Poisson correction later
            word_count = min(500, SCENARIO_WORD_COUNTS.get(scenario, 500))

            for i in range(n_sims):
                speech = self.generate_speech(scenario, word_count, temperature=1.0)
                total_words_generated += len(speech.split())

                for term_key, pattern in term_patterns.items():
                    matches = pattern.findall(speech)
                    if matches:
                        term_stats[term_key]['speeches_containing'] += 1
                        term_stats[term_key]['total_mentions'] += len(matches)

                sim_count += 1
                if sim_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = sim_count / elapsed if elapsed > 0 else 1
                    eta = (total_sims - sim_count) / rate if rate > 0 else 0
                    self._status.update(
                        stage=f'GPT-2 Monte Carlo: {sim_count}/{total_sims}',
                        progress=sim_count / total_sims,
                        eta_seconds=round(eta, 1),
                    )

        # Compute probabilities
        avg_words = total_words_generated / total_sims if total_sims > 0 else 500
        predictions = []
        for term_key, stats in term_stats.items():
            probability = stats['speeches_containing'] / total_sims if total_sims > 0 else 0
            avg_mentions = stats['total_mentions'] / total_sims if total_sims > 0 else 0

            predictions.append({
                'term': term_key,
                'probability': round(probability, 4),
                'speeches_containing': stats['speeches_containing'],
                'total_mentions': stats['total_mentions'],
                'avg_mentions_per_speech': round(avg_mentions, 4),
                'model_name': 'gpt2_monte_carlo',
                'confidence': min(1.0, total_sims / 200),
            })

        duration = time.time() - start_time
        logger.info(f"GPT-2 Monte Carlo: {total_sims} sims in {duration:.1f}s, {len(predictions)} terms")

        return {
            'term_predictions': predictions,
            'simulation_params': {
                'num_simulations': total_sims,
                'scenario_weights': weights,
                'avg_words_per_speech': round(avg_words, 0),
                'model_type': 'gpt2_lora',
            },
            'generated_at': datetime.utcnow().isoformat(),
            'discovered_phrases': [],
        }

    # ── Private helpers ──

    def _load_corpus(self) -> list[str]:
        """Load training corpus from database.

        Uses word_count >= 50 to include social_media_daily digests.
        """
        with get_session() as session:
            speeches = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.is_processed == True,
                Speech.word_count >= 50,
            ).all()
            return [s.transcript for s in speeches if s.transcript]

    def _load_model(self):
        """Load GPT-2 base + latest LoRA adapter for inference."""
        if self._model is not None:
            return

        adapter_path = os.path.join(ADAPTERS_DIR, 'adapter_latest')
        if not os.path.exists(adapter_path):
            logger.warning("No fine-tuned GPT-2 adapter found")
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            self._status.update(stage='Loading model')
            model_name = config.fine_tune_model
            base_model = AutoModelForCausalLM.from_pretrained(model_name)
            self._model = PeftModel.from_pretrained(base_model, adapter_path)
            self._model.eval()
            self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            logger.info(f"Loaded fine-tuned {model_name} + LoRA adapter")

        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            self._model = None
            self._tokenizer = None

    def _generate_text(self, prompt: str, word_count: int,
                       temperature: float) -> str:
        """Generate text using the loaded model."""
        import torch

        input_ids = self._tokenizer.encode(prompt, return_tensors='pt')
        max_new_tokens = min(word_count * 2, 2048)  # rough word→token ratio

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=max(0.3, min(2.0, temperature)),
                do_sample=True,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        text = self._tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the prompt prefix if present
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        # Trim to target word count
        words = text.split()
        if len(words) > word_count:
            # Find a sentence boundary near the target
            text = ' '.join(words[:word_count])
            last_period = text.rfind('.')
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]

        return text

    def _save_checkpoint(self, epoch: int, step: int, best_loss: float):
        """Save training checkpoint for resumption."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'best_loss': best_loss,
            'loss_history': self._loss_history[-1000:],  # keep last 1000
            'saved_at': datetime.utcnow().isoformat(),
        }
        path = os.path.join(CHECKPOINTS_DIR, 'checkpoint_latest.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f)

    def _load_checkpoint(self) -> Optional[dict]:
        """Load latest checkpoint for training resumption."""
        path = os.path.join(CHECKPOINTS_DIR, 'checkpoint_latest.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _next_version(self) -> str:
        """Get next version string for GPT-2 model."""
        with get_session() as session:
            latest = session.query(ModelVersion).filter_by(
                model_type='gpt2_lora'
            ).order_by(ModelVersion.created_at.desc()).first()

        if not latest:
            return '2.0.0'  # Start GPT-2 versions at 2.x

        parts = latest.version.split('.')
        if len(parts) == 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f'{major}.{minor}.{patch + 1}'
        return '2.0.0'

    def has_trained_model(self) -> bool:
        """Check if a fine-tuned adapter exists."""
        adapter_path = os.path.join(ADAPTERS_DIR, 'adapter_latest')
        return os.path.exists(adapter_path)
