"""
Amharic → Oromo Transformer Training Script
=============================================
Fixed version with:
  1. max_in / max_out saved to config.json (no hardcoding)
  2. Masked loss & accuracy (ignoring padding tokens)
  3. BOS / EOS tokens prepended/appended to decoder targets
  4. 2-layer decoder (matching the 2-layer encoder)
"""

import datetime
import json
import os

import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# LOAD DATA
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(BASE_DIR, "cleaned_amh_omo.xlsx"))

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# =====================================================
# SAVE TEXT FILES FOR SENTENCEPIECE
# =====================================================

am_txt = os.path.join(BASE_DIR, "amharic.txt")
om_txt = os.path.join(BASE_DIR, "oromo.txt")

with open(am_txt, "w", encoding="utf-8") as f:
    for s in train_df["Amharic"]:
        f.write(str(s) + "\n")

with open(om_txt, "w", encoding="utf-8") as f:
    for s in train_df["Oromo"]:
        f.write(str(s) + "\n")

# =====================================================
# TRAIN SENTENCEPIECE TOKENIZERS
# =====================================================

spm.SentencePieceTrainer.train(
    input=am_txt,
    model_prefix=os.path.join(BASE_DIR, "am"),
    vocab_size=4000,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

spm.SentencePieceTrainer.train(
    input=om_txt,
    model_prefix=os.path.join(BASE_DIR, "om"),
    vocab_size=4000,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

# =====================================================
# LOAD TOKENIZERS
# =====================================================

am_sp = spm.SentencePieceProcessor()
am_sp.load(os.path.join(BASE_DIR, "am.model"))

om_sp = spm.SentencePieceProcessor()
om_sp.load(os.path.join(BASE_DIR, "om.model"))

am_vocab_size = am_sp.get_piece_size()
om_vocab_size = om_sp.get_piece_size()

print(f"Amharic vocab size: {am_vocab_size}")
print(f"Oromo vocab size:   {om_vocab_size}")

# =====================================================
# TOKENIZE
# =====================================================

X_train_raw = [am_sp.encode(str(s), out_type=int) for s in train_df["Amharic"]]
X_val_raw = [am_sp.encode(str(s), out_type=int) for s in val_df["Amharic"]]

# --- FIX 3: Prepend BOS and append EOS to target sequences ---
# This ensures the decoder sees <s> at the start during training,
# matching what inference.py does.
BOS = om_sp.bos_id()  # 2
EOS = om_sp.eos_id()  # 3

y_train_raw = [
    [BOS] + om_sp.encode(str(s), out_type=int) + [EOS]
    for s in train_df["Oromo"]
]
y_val_raw = [
    [BOS] + om_sp.encode(str(s), out_type=int) + [EOS]
    for s in val_df["Oromo"]
]

# =====================================================
# PADDING — FIX 1: save max_in / max_out to config.json
# =====================================================

max_in = max(len(x) for x in X_train_raw)
max_out = max(len(y) for y in y_train_raw)

print(f"\nmax_in  (encoder seq length): {max_in}")
print(f"max_out (decoder seq length): {max_out}")

# Save to config so inference.py can load them
config = {
    "max_in": int(max_in),
    "max_out": int(max_out),
    "am_vocab_size": int(am_vocab_size),
    "om_vocab_size": int(om_vocab_size),
    "d_model": 128,
    "num_heads": 4,
    "ff_dim": 512,
    "num_enc_layers": 2,
    "num_dec_layers": 2,
}
config_path = os.path.join(BASE_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to {config_path}")

X_train = pad_sequences(X_train_raw, maxlen=max_in, padding="post")
X_val = pad_sequences(X_val_raw, maxlen=max_in, padding="post")

y_train = pad_sequences(y_train_raw, maxlen=max_out, padding="post")
y_val = pad_sequences(y_val_raw, maxlen=max_out, padding="post")

# Decoder input  = [BOS, tok1, tok2, ..., tokN]  (drop last)
# Decoder target  = [tok1, tok2, ..., tokN, EOS]  (drop first)
decoder_input_train = y_train[:, :-1]
decoder_target_train = y_train[:, 1:]

decoder_input_val = y_val[:, :-1]
decoder_target_val = y_val[:, 1:]

print(f"Encoder input shape:  {X_train.shape}")
print(f"Decoder input shape:  {decoder_input_train.shape}")
print(f"Decoder target shape: {decoder_target_train.shape}")

# =====================================================
# MODEL SETTINGS
# =====================================================

d_model = config["d_model"]
num_heads = config["num_heads"]
ff_dim = config["ff_dim"]
num_enc_layers = config["num_enc_layers"]
num_dec_layers = config["num_dec_layers"]


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        d_model = tf.cast(self.d_model, tf.float32)
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


# =====================================================
# POSITIONAL ENCODING
# =====================================================


def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)


# =====================================================
# TRANSFORMER ENCODER BLOCK
# =====================================================


def encoder_block(x, num_heads, d_model, ff_dim, dropout=0.2):
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn))

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(ffn))
    return x


def build_encoder(x, num_layers):
    for _ in range(num_layers):
        x = encoder_block(x, num_heads, d_model, ff_dim)
    return x


# =====================================================
# TRANSFORMER DECODER BLOCK — FIX 4: stacked layers
# =====================================================


def decoder_block(x, enc_output, dropout=0.2):
    # Masked self-attention
    attn1 = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x, use_causal_mask=True)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn1))

    # Cross-attention
    attn2 = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, enc_output)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn2))

    # Feed-forward
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(ffn))
    return x


def build_decoder(x, enc_output, num_layers):
    for _ in range(num_layers):
        x = decoder_block(x, enc_output)
    return x


# =====================================================
# BUILD MODEL
# =====================================================

# Encoder
encoder_inputs = Input(shape=(max_in,), name="encoder_input")
enc_embed = Embedding(am_vocab_size, d_model)(encoder_inputs)
enc_embed = enc_embed + positional_encoding(max_in, d_model)
enc_output = build_encoder(enc_embed, num_layers=num_enc_layers)

# Decoder — input shape is max_out - 1 (we dropped last token)
dec_seq_len = max_out - 1
decoder_inputs = Input(shape=(dec_seq_len,), name="decoder_input")
dec_embed = Embedding(om_vocab_size, d_model)(decoder_inputs)
dec_embed = dec_embed + positional_encoding(dec_seq_len, d_model)
dec_output = build_decoder(dec_embed, enc_output, num_layers=num_dec_layers)

outputs = Dense(om_vocab_size, activation="softmax")(dec_output)

model = Model([encoder_inputs, decoder_inputs], outputs)

# =====================================================
# FIX 2: MASKED LOSS & ACCURACY (ignore padding)
# =====================================================

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")


def masked_loss(y_true, y_pred):
    """Loss that ignores padding (token 0) positions."""
    mask = tf.cast(y_true != 0, tf.float32)
    loss = loss_object(y_true, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    """Accuracy that ignores padding (token 0) positions."""
    pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
    match = tf.cast(pred_ids == y_true, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(match * mask) / tf.reduce_sum(mask)


learning_rate_schedule = CustomSchedule(d_model=d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate_schedule,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9,
)

model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

model.summary()

# =====================================================
# TRAIN
# =====================================================

log_dir = os.path.join(
    BASE_DIR, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

history = model.fit(
    [X_train, decoder_input_train],
    decoder_target_train,
    validation_data=([X_val, decoder_input_val], decoder_target_val),
    batch_size=16,
    epochs=20,
    callbacks=[tensorboard_cb, early_stopping],
)

print(f"\nTensorBoard log directory: {log_dir}")

# =====================================================
# SAVE MODEL
# =====================================================

model_path = os.path.join(BASE_DIR, "amh_omo_transformer.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

# =====================================================
# QUICK INFERENCE TEST
# =====================================================


def translate(sentence):
    """Translate a single Amharic sentence to Oromo."""
    encoded = am_sp.encode(str(sentence), out_type=int)
    encoded = pad_sequences([encoded], maxlen=max_in, padding="post")

    output = [BOS]
    for _ in range(dec_seq_len):
        dec_input = pad_sequences([output], maxlen=dec_seq_len, padding="post")
        preds = model.predict([encoded, dec_input], verbose=0)
        next_token = int(np.argmax(preds[0, len(output) - 1]))

        if next_token == EOS:
            break
        output.append(next_token)

    tokens = [t for t in output if t not in (BOS, EOS, 0)]
    return om_sp.decode(tokens)


# Test on a few training examples
print("\n" + "=" * 60)
print("QUICK TRANSLATION TEST (on training samples)")
print("=" * 60)
sample = train_df.head(5)
for _, row in sample.iterrows():
    src = str(row["Amharic"])
    ref = str(row["Oromo"])
    pred = translate(src)
    print(f"\n  AM: {src}")
    print(f"  REF: {ref}")
    print(f"  PRED: {pred}")

# =====================================================
# BLEU EVALUATION ON TEST SET
# =====================================================

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

smooth = SmoothingFunction().method4

print("\n\nComputing BLEU on test set (50 samples)...")
scores = []
test_sample = test_df.sample(n=min(50, len(test_df)), random_state=42)
for idx, row in test_sample.iterrows():
    try:
        ref = str(row["Oromo"]).split()
        pred = translate(str(row["Amharic"])).split()
        if ref and pred:
            scores.append(sentence_bleu([ref], pred, smoothing_function=smooth))
    except Exception as e:
        print(f"  Skipping row {idx}: {e}")

bleu = float(np.mean(scores)) if scores else 0.0
print(f"BLEU score: {bleu:.4f}")
