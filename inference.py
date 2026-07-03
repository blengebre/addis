"""
Amharic → Oromo Inference Script
=================================
Fixed and optimized to be 100% compatible with the updated training configuration.
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
import sacrebleu

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "amh_omo_transformer.keras")
SP_MODEL_PATH = os.path.join(BASE_DIR, "am_om.model")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# =====================================================
# LOAD CONFIG — fallback parameters if needed
# =====================================================

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    MAX_IN = config["max_in"]
    MAX_OUT = config["max_out"]
    VOCAB_SIZE = config["vocab_size"]
    print(f"Loaded config: max_in={MAX_IN}, max_out={MAX_OUT}")
else:
    print("WARNING: config.json not found — using fallback values!")
    MAX_IN = 153
    MAX_OUT = 75
    VOCAB_SIZE = 8000


# =====================================================
# CUSTOM METRICS FOR COMPATIBLE LOADING
# =====================================================

def masked_loss(y_true, y_pred):
    return 0.0  

def masked_accuracy(y_true, y_pred):
    return 0.0

# =====================================================
# LOAD MODEL WITH PATCHING
# =====================================================

def _patch_keras_config(obj):
    """Remove config keys added by newer Keras versions that older versions reject."""
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
    """Load the saved .keras model with registered custom objects and patched config."""
    custom_objects = {
        "masked_loss": masked_loss,
        "masked_accuracy": masked_accuracy
    }
    
    with zipfile.ZipFile(path, "r") as reader:
        config_data = json.loads(reader.read("config.json").decode("utf-8"))

    if not _patch_keras_config(config_data):
        return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(path, "r") as reader:
            with zipfile.ZipFile(tmp_path, "w") as writer:
                for entry in reader.namelist():
                    if entry == "config.json":
                        writer.writestr(entry, json.dumps(config_data))
                    else:
                        writer.writestr(entry, reader.read(entry))
        return tf.keras.models.load_model(tmp_path, custom_objects=custom_objects, compile=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


print("Loading model...")
model = load_model_compatible(MODEL_PATH)

# =====================================================
# DYNAMIC SHAPE EXTRACTION
# =====================================================
# Instantly extracts exact layer requirements directly from the model architecture graph
try:
    # Look up the shape of the decoder input layer (usually index 1)
    decoder_shape = model.inputs[1].shape
    DEC_SEQ_LEN = decoder_shape[1]
    
    # Also double check the encoder shape requirements while we are here
    encoder_shape = model.inputs[0].shape
    MAX_IN = encoder_shape[1] if encoder_shape[1] is not None else MAX_IN
    print(f"Model-extracted boundaries: MAX_IN={MAX_IN}, DEC_SEQ_LEN={DEC_SEQ_LEN}")
except Exception as e:
    print(f"Fallback boundary selection tracking triggered: {e}")
    DEC_SEQ_LEN = MAX_OUT - 1

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

BOS = sp.bos_id()
EOS = sp.eos_id()
PAD = sp.pad_id()

print("Shared vocabulary size:", sp.get_piece_size())


# =====================================================
# INFERENCE TRANSLATION ENGINE
# =====================================================

def translate(sentence):
    """Translate a single Amharic sentence to Oromo with matched causal padding."""
    encoded = sp.encode(str(sentence), out_type=int)
    encoded = pad_sequences([encoded], maxlen=MAX_IN, padding="post")

    output = [BOS]
    for _ in range(DEC_SEQ_LEN):
        # Build right-padded array ([BOS, tok1, tok2, 0, 0...]) to match the look-ahead structure
        decoder_input = pad_sequences([output], maxlen=DEC_SEQ_LEN, padding="post")
        
        # Predict the token probabilities matrix
        preds = model.predict([encoded, decoder_input], verbose=0)
        
        # Extract the predictions column relative to the true generated sequence location
        next_token = int(np.argmax(preds[0, len(output) - 1]))

        if next_token == EOS:
            break
        output.append(next_token)

    tokens = [t for t in output if t not in (BOS, EOS, 0)]
    return sp.decode(tokens)


def compute_bleu(reference, candidate):
    return sacrebleu.sentence_bleu(candidate, [reference]).score


def load_test_df():
    df = pd.read_excel(os.path.join(BASE_DIR, "cleaned_amh_omo.xlsx"))

    df["am_length"] = df["Amharic"].astype(str).apply(lambda x: len(x.split()))
    df["om_length"] = df["Oromo"].astype(str).apply(lambda x: len(x.split()))
    df["avg_length"] = (df["am_length"] + df["om_length"]) / 2

    df["length_group"] = pd.qcut(df["avg_length"], q=3, labels=["Short", "Medium", "Long"])

    test_parts = []
    for _, group in df.groupby("length_group"):
        _, temp = train_test_split(group, test_size=0.20, random_state=42, shuffle=True)
        _, test = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)
        test_parts.append(test)

    return pd.concat(test_parts).reset_index(drop=True)


def compute_bleu_on_df(df):
    references = []
    hypotheses = []

    for _, row in df.iterrows():
        hypotheses.append(translate(str(row["Amharic"])))
        references.append(str(row["Oromo"]))

    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def evaluate_bleu_by_length():
    test_df = load_test_df()
    print("\nBLEU by sentence length")

    for group in ["Short", "Medium", "Long"]:
        subset = test_df[test_df["length_group"] == group]
        bleu = compute_bleu_on_df(subset)
        print(f"{group:<8}: {bleu:.2f}")


def evaluate_test_bleu():
    test_df = load_test_df()
    bleu = compute_bleu_on_df(test_df)
    print(f"\nOverall BLEU: {bleu:.2f}")
    evaluate_bleu_by_length()
    return bleu

# =====================================================
# CLI CONTROL
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Amharic-to-Oromo inference engine")
    parser.add_argument("--translate", type=str, help="Translate a single Amharic sentence")
    parser.add_argument("--reference", type=str, help="Reference Oromo sentence for BLEU validation")
    parser.add_argument("--eval-bleu", action="store_true", help="Compute corpus BLEU metrics on the test dataset split")
   
    args = parser.parse_args()

    if args.eval_bleu:
        evaluate_test_bleu()
        return

    if args.translate:
        translation = translate(args.translate)
        print("Amharic:", args.translate)
        print("Oromo:  ", translation)
        if args.reference:
            bleu_score = compute_bleu(args.reference, translation)
            print("BLEU:   ", round(bleu_score, 4))
        return

    # Interactive CLI Mode
    while True:
        source = input("\nEnter Amharic text (or 'quit'): ")
        if source.strip().lower() == "quit":
            break
        translation = translate(source)
        print("Oromo:", translation)


if __name__ == "__main__":
    main()