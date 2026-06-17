"""
Amharic → Oromo Inference Script
=================================
Updated to:
  - Load max_in / max_out from config.json (no hardcoding)
  - Match BOS/EOS handling from the fixed training script
"""

import argparse
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "amh_omo_transformer.keras")
AM_MODEL_PATH = os.path.join(BASE_DIR, "am.model")
OM_MODEL_PATH = os.path.join(BASE_DIR, "om.model")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# =====================================================
# LOAD CONFIG — no more hardcoded MAX_IN / MAX_OUT
# =====================================================

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    MAX_IN = config["max_in"]
    MAX_OUT = config["max_out"]
    print(f"Loaded config: max_in={MAX_IN}, max_out={MAX_OUT}")
else:
    # Fallback (will be wrong if these don't match training)
    print("WARNING: config.json not found — using fallback values!")
    MAX_IN = 153
    MAX_OUT = 72

# Decoder input length = MAX_OUT - 1 (we drop the last token)
DEC_SEQ_LEN = MAX_OUT - 1


# =====================================================
# LOAD MODEL (with backward-compatibility patching)
# =====================================================


def _patch_keras_config(obj):
    """Remove config keys added by newer Keras that older versions reject."""
    modified = False
    unsupported_keys = {"quantization_config", "use_gate"}

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in unsupported_keys:
                obj.pop(key)
                modified = True
            elif key == "dtype" and isinstance(obj[key], dict):
                dtype_obj = obj[key]
                if dtype_obj.get("class_name") == "DTypePolicy":
                    obj[key] = dtype_obj.get("config", {}).get("name", "float32")
                    modified = True
                else:
                    if _patch_keras_config(obj[key]):
                        modified = True
            else:
                if _patch_keras_config(obj[key]):
                    modified = True
    elif isinstance(obj, list):
        for item in obj:
            if _patch_keras_config(item):
                modified = True

    return modified


def load_model_compatible(path):
    """Load the saved .keras model and remove unsupported layer config fields."""
    with zipfile.ZipFile(path, "r") as reader:
        config = json.loads(reader.read("config.json").decode("utf-8"))

    if not _patch_keras_config(config):
        return tf.keras.models.load_model(path, compile=False)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(path, "r") as reader:
            with zipfile.ZipFile(tmp_path, "w") as writer:
                for entry in reader.namelist():
                    if entry == "config.json":
                        writer.writestr(entry, json.dumps(config))
                    else:
                        writer.writestr(entry, reader.read(entry))
        return tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


print("Loading model...")
model = load_model_compatible(MODEL_PATH)

am_sp = spm.SentencePieceProcessor()
am_sp.load(AM_MODEL_PATH)

om_sp = spm.SentencePieceProcessor()
om_sp.load(OM_MODEL_PATH)

BOS = om_sp.bos_id()  # 2
EOS = om_sp.eos_id()  # 3


# =====================================================
# TRANSLATE
# =====================================================


def translate(sentence):
    """Translate a single Amharic sentence to Oromo."""
    encoded = am_sp.encode(str(sentence), out_type=int)
    encoded = pad_sequences([encoded], maxlen=MAX_IN, padding="post")

    # Start with BOS token — matches training decoder input
    output = [BOS]

    for _ in range(DEC_SEQ_LEN):
        dec_input = pad_sequences([output], maxlen=DEC_SEQ_LEN, padding="post")
        preds = model.predict([encoded, dec_input], verbose=0)
        next_token = int(np.argmax(preds[0, len(output) - 1]))

        if next_token == EOS:
            break
        output.append(next_token)

    # Strip special tokens before decoding
    tokens = [t for t in output if t not in (BOS, EOS, 0)]
    return om_sp.decode(tokens)


# =====================================================
# BLEU
# =====================================================


def compute_bleu(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    smooth = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)


def load_test_df():
    df = pd.read_excel(os.path.join(BASE_DIR, "cleaned_amh_omo.xlsx"))
    _, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return test_df


def compute_bleu_on_df(df, n=10):
    sample_size = min(n, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    scores = []
    smooth = SmoothingFunction().method4
    for idx, row in sample_df.iterrows():
        try:
            ref = str(row["Oromo"]).split()
            pred = translate(str(row["Amharic"])).split()
            if ref and pred:
                scores.append(sentence_bleu([ref], pred, smoothing_function=smooth))
        except Exception as e:
            print(f"Skipping row {idx}: {e}")
    return float(np.mean(scores)) if scores else 0.0


def evaluate_test_bleu(n=10):
    test_df = load_test_df()
    bleu_score = compute_bleu_on_df(test_df, n=n)
    print(f"BLEU score ({n} samples): {bleu_score:.4f}")
    return bleu_score


# =====================================================
# CLI
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Amharic-to-Oromo inference")
    parser.add_argument("--translate", type=str, help="Translate a single Amharic sentence")
    parser.add_argument("--reference", type=str, help="Reference Oromo sentence for BLEU scoring")
    parser.add_argument("--eval-bleu", action="store_true", help="Compute BLEU on the test split")
    parser.add_argument("--n", type=int, default=10, help="Number of test samples for BLEU evaluation")
    args = parser.parse_args()

    if args.eval_bleu:
        evaluate_test_bleu(n=args.n)
        return

    if args.translate:
        translation = translate(args.translate)
        print("Amharic:", args.translate)
        print("Oromo:  ", translation)
        if args.reference:
            bleu_score = compute_bleu(args.reference, translation)
            print("BLEU:   ", round(bleu_score, 4))
        return

    # Interactive mode
    while True:
        source = input("\nEnter Amharic text (or 'quit'): ")
        if source.strip().lower() == "quit":
            break
        translation = translate(source)
        print("Oromo:", translation)


if __name__ == "__main__":
    main()