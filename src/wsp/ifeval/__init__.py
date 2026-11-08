"""IFEval verifier (Zhou et al., 2023), vendored from google-research/instruction_following_eval (Apache-2.0).

Only changes: package-relative imports and the punkt sentence tokenizer loader (punkt_tab, see instructions_util).
Data: data/ifeval/input_data.jsonl (541 prompts). Needs `nltk.download("punkt_tab", download_dir="data/nltk_data")` once.
"""
import nltk

from wsp.paths import DATA

nltk.data.path.insert(0, str(DATA / "nltk_data"))
