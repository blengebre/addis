import argparse
import json
import os
import tempfile
import zipfile

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "amh_omo_transformer.keras")
AM_MODEL_PATH = os.path.join(BASE_DIR, "am.model")
OM_MODEL_PATH = os.path.join(BASE_DIR, "om.model")

MAX_IN = 153
MAX_OUT = 72


def load_model_compatible(path):
    """Load the saved .keras model and remove unsupported layer config fields."""
    with zipfile.ZipFile(path, "r") as reader:
        config = json.loads(reader.read("config.json").decode("utf-8"))
        modified = False

        for layer in config["config"]["layers"]:
            if layer["class_name"] == "MultiHeadAttention" and "use_gate" in layer["config"]:
                layer["config"].pop("use_gate")
                modified = True

        if not modified:
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


model = load_model_compatible(MODEL_PATH)

am_sp = spm.SentencePieceProcessor()
am_sp.load(AM_MODEL_PATH)

om_sp = spm.SentencePieceProcessor()
om_sp.load(OM_MODEL_PATH)


def translate(sentence):
    encoded = am_sp.encode(sentence, out_type=int)
    encoded = pad_sequences([encoded], maxlen=MAX_IN, padding="post")

    output = [om_sp.bos_id()]
    for _ in range(MAX_OUT - 1):
        dec_input = pad_sequences([output], maxlen=MAX_OUT - 1, padding="post")
        preds = model.predict([encoded, dec_input], verbose=0)
        next_token = int(np.argmax(preds[0, len(output) - 1]))

        if next_token == om_sp.eos_id():
            break

        output.append(next_token)

    tokens = [t for t in output if t not in (om_sp.bos_id(), om_sp.eos_id(), 0)]
    return om_sp.decode(tokens)


def compute_bleu(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    smooth = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)


def main():
    parser = argparse.ArgumentParser(description="Amharic-to-Oromo inference")
    parser.add_argument("--translate", type=str, help="Translate a single Amharic sentence")
    parser.add_argument("--reference", type=str, help="Reference Oromo sentence for BLEU scoring")
    args = parser.parse_args()

    if args.translate:
        translation = translate(args.translate)
        print("Amharic:", args.translate)
        print("Oromo:", translation)
        if args.reference:
            bleu_score = compute_bleu(args.reference, translation)
            print("BLEU:", round(bleu_score, 4))
        return

    while True:
        source = input("\nEnter Amharic text (or 'quit'): ")
        if source.strip().lower() == "quit":
            break
        translation = translate(source)
        print("Oromo:", translation)


if __name__ == "__main__":
    main()
