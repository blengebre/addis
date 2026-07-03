"""
Amharic → Oromo Transformer Training Script
=============================================
Fixed version: Fully synced padding structures and sequence matrix alignment.
"""

import datetime
import json
import os

import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    Callback,
)

import sacrebleu
import wandb

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# WANDB INITIALIZATION
# =====================================================
wandb.init(
    entity="blen-gebre-ug-addis-abeba-university",
    project="amharic-oromo-transformer",
    config={
        "architecture": "Transformer",
        "dataset": "cleaned_amh_omo.xlsx",
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": "Noam schedule",
        "optimizer": "Adam",
        "d_model": 128,
        "num_heads": 4,
        "ff_dim": 512,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "vocab_size": 8000,
        "tokenizer": "SentencePiece BPE"
    }
)
df = pd.read_excel(os.path.join(BASE_DIR, "cleaned_amh_omo.xlsx"))

# =====================================================
# LENGTH-BASED DATASET SPLIT
# =====================================================
df["am_length"] = df["Amharic"].astype(str).apply(lambda x: len(x.split()))
df["om_length"] = df["Oromo"].astype(str).apply(lambda x: len(x.split()))
df["avg_length"] = (df["am_length"] + df["om_length"]) / 2

df["length_group"] = pd.qcut(df["avg_length"], q=3, labels=["Short", "Medium", "Long"])

train_parts, val_parts, test_parts = [], [], []

for group_name, group in df.groupby("length_group"):
    train_split, temp_split = train_test_split(group, test_size=0.20, random_state=42, shuffle=True)
    val_split, test_split = train_test_split(temp_split, test_size=0.50, random_state=42, shuffle=True)
    
    train_parts.append(train_split)
    val_parts.append(val_split)
    test_parts.append(test_split)

train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

# =====================================================
# CORPUS & TOKENIZER TRAINING
# =====================================================
multi_txt = os.path.join(BASE_DIR, "am_om_corpus.txt")
with open(multi_txt, "w", encoding="utf-8") as f:
    for sentence in train_df["Amharic"]:
        f.write(str(sentence).strip() + "\n")
    for sentence in train_df["Oromo"]:
        f.write(str(sentence).strip() + "\n")

spm.SentencePieceTrainer.train(
    input=multi_txt,
    model_prefix=os.path.join(BASE_DIR, "am_om"),
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    shuffle_input_sentence=True,
)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(BASE_DIR, "am_om.model"))
vocab_size = sp.get_piece_size()

# =====================================================
# TOKENIZE & ADD CONTROL TOKENS
# =====================================================
X_train_raw = [sp.encode(str(s), out_type=int) for s in train_df["Amharic"]]
X_val_raw = [sp.encode(str(s), out_type=int) for s in val_df["Amharic"]]

BOS, EOS, PAD = sp.bos_id(), sp.eos_id(), sp.pad_id()

y_train_raw = [[BOS] + sp.encode(str(s), out_type=int) + [EOS] for s in train_df["Oromo"]]
y_val_raw = [[BOS] + sp.encode(str(s), out_type=int) + [EOS] for s in val_df["Oromo"]]

max_in = max(len(x) for x in X_train_raw)
max_out = max(len(y) for y in y_train_raw)

# FIX: Explicitly enforce DEC_SEQ_LEN to match full target padding space
max_in = max(len(x) for x in X_train_raw)
max_out = max(len(y) for y in y_train_raw)

# FIX: Explicitly enforce dec_seq_len to match the true trimmed slice space (73)
dec_seq_len = max_out - 1

config = {
  "max_in": max_in,
  "max_out": max_out, 
  "vocab_size": vocab_size,
  "d_model": 128,
  "num_heads": 4,
  "ff_dim": 512,
  "num_enc_layers": 2,
  "num_dec_layers": 2
}
with open(os.path.join(BASE_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

X_train = pad_sequences(X_train_raw, maxlen=max_in, padding="post")
X_val = pad_sequences(X_val_raw, maxlen=max_in, padding="post")

# FIX: Directly use list comprehensions sliced into dec_seq_len width
decoder_input_train = pad_sequences([seq[:-1] for seq in y_train_raw], maxlen=dec_seq_len, padding="post")
decoder_target_train = pad_sequences([seq[1:] for seq in y_train_raw], maxlen=dec_seq_len, padding="post")

decoder_input_val = pad_sequences([seq[:-1] for seq in y_val_raw], maxlen=dec_seq_len, padding="post")
decoder_target_val = pad_sequences([seq[1:] for seq in y_val_raw], maxlen=dec_seq_len, padding="post")
# =====================================================
# MODEL ARCHITECTURE SETUP
# =====================================================
d_model, num_heads, ff_dim = config["d_model"], config["num_heads"], config["ff_dim"]
num_enc_layers, num_dec_layers = config["num_enc_layers"], config["num_dec_layers"]

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        d_model = tf.cast(self.d_model, tf.float32)
        step = tf.cast(step, tf.float32)
        return tf.math.rsqrt(d_model) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5))
    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rads = pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

def encoder_block(x, num_heads, d_model, ff_dim, dropout=0.2):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn))
    ffn = Dense(d_model)(Dense(ff_dim, activation="relu")(x))
    return LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(ffn))

def decoder_block(x, enc_output, dropout=0.2):
    attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x, use_causal_mask=True)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn1))
    attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query=x, key=enc_output, value=enc_output)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn2))
    ffn = Dense(d_model)(Dense(ff_dim, activation="relu")(x))
    return LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(ffn))

# Build Pipeline
encoder_inputs = Input(shape=(max_in,), name="encoder_input")
enc_embed = Embedding(vocab_size, d_model)(encoder_inputs) + positional_encoding(max_in, d_model)
enc_output = enc_embed
for _ in range(num_enc_layers):
    enc_output = encoder_block(enc_output, num_heads, d_model, ff_dim)

decoder_inputs = Input(shape=(dec_seq_len,), name="decoder_input")
dec_embed = Embedding(vocab_size, d_model)(decoder_inputs) + positional_encoding(dec_seq_len, d_model)
dec_output = dec_embed
for _ in range(num_dec_layers):
    dec_output = decoder_block(dec_output, enc_output)

outputs = Dense(vocab_size, activation="softmax")(dec_output)
model = Model([encoder_inputs, decoder_inputs], outputs)

# =====================================================
# MASKED METRICS & COMPILING
# =====================================================
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

def masked_loss(y_true, y_pred):
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(loss_object(y_true, y_pred) * mask) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast(y_true != 0, tf.float32)
    match = tf.cast(tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype) == y_true, tf.float32)
    return tf.reduce_sum(match * mask) / tf.reduce_sum(mask)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=masked_loss, metrics=[masked_accuracy])

# FIX: Completely rewritten translate framework to use post-padded indices correctly
def translate(sentence):
    encoded = sp.encode(str(sentence), out_type=int)
    encoded = pad_sequences([encoded], maxlen=max_in, padding="post")
    output = [BOS]
    
    for _ in range(dec_seq_len):
        # Match causal post-padding natively
        decoder_input = pad_sequences([output], maxlen=dec_seq_len, padding="post")
                
        preds = model.predict([encoded, decoder_input], verbose=0)
        next_token = int(np.argmax(preds[0, len(output) - 1]))
        if next_token == EOS:
            break
        output.append(next_token)
        
    return sp.decode([t for t in output if t not in (BOS, EOS, 0)])

# =====================================================
# CALLBACKS & FIT ENGINE
# =====================================================
class BLEUCallback(Callback):
    def __init__(self, validation_df):
        super().__init__()
        self.validation_df = validation_df
    def on_epoch_end(self, epoch, logs=None):
        hypotheses, references = [], []
        sample = self.validation_df.sample(200, random_state=epoch)
        for _, row in sample.iterrows():
            hypotheses.append(translate(str(row["Amharic"])))
            references.append([str(row["Oromo"])])
        bleu = sacrebleu.corpus_bleu(hypotheses, references).score
        print(f"\n--- Validation BLEU at Epoch {epoch+1}: {bleu:.2f} ---")
        if logs is not None: logs["val_bleu"] = bleu
        wandb.log({"val_bleu": bleu, "epoch": epoch + 1})

history = model.fit(
    [X_train, decoder_input_train], decoder_target_train,
    validation_data=([X_val, decoder_input_val], decoder_target_val),
    batch_size=16, epochs=20,
    callbacks=[
        TensorBoard(os.path.join(BASE_DIR, "logs")),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        BLEUCallback(val_df),
        WandbMetricsLogger(),
        WandbModelCheckpoint("wandb_best_model.keras", monitor="val_loss", save_best_only=True)
    ]
)

# Save Model Output cleanly
model.save(os.path.join(BASE_DIR, "amh_omo_transformer.keras"))
wandb.finish()