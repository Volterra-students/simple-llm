import json
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

parser = ArgumentParser()
parser.add_argument("file", help="path del file di testo da cui iniziare")
parser.add_argument("-o", "--output", help="path di output del modello keras",
                    default="model/model.keras")
parser.add_argument("-oi", "--output_indices", help="path di output dell'indici delle parole",
                    default="model/indices.json")
parser.add_argument("-l", "--len", help="lunghezza sequenza", default=5)
parser.add_argument("-e", "--epochs",
                    help="epoche per il training (da considerare l'early stopping)", default=300)
parser.add_argument("-p", "--patience", help="numero di epoch per l'early stopping", default=4)
args = parser.parse_args()

# per assicurarsi che le directory esistino
output = pathlib.Path(args.output)
if not output.parent.exists():
    os.makedirs(output.parent, exist_ok=True)
output_indices = pathlib.Path(args.output_indices)
if not output_indices.parent.exists():
    os.makedirs(output_indices.parent, exist_ok=True)

# =========================================================
# PREPARAZIONE TESTO
# =========================================================

with open(args.file, "rt") as f:
    testo = f.read()

# =========================================================
# TOKENIZZAZIONE
# =========================================================

tokenizer = Tokenizer()
tokenizer.fit_on_texts([testo])

sequenza_tokenizzata = tokenizer.texts_to_sequences([testo])[0]
vocabolario = len(tokenizer.word_index) + 1

print("Vocabolario:")
print(tokenizer.word_index)
print()

print("Sequenza tokenizzata:")
print(sequenza_tokenizzata)
print()

# =========================================================
# CREAZIONE DATASET
# =========================================================

lunghezza_sequenza = args.len

X = []
y = []

for i in range(len(sequenza_tokenizzata) - lunghezza_sequenza):
    seq_input = sequenza_tokenizzata[i:i + lunghezza_sequenza]
    parola_output = sequenza_tokenizzata[i + lunghezza_sequenza]

    X.append(seq_input)
    y.append(parola_output)

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# one-hot encoding di y
y = to_categorical(y, num_classes=vocabolario)

print("Shape y dopo one-hot:", y.shape)
print()

# =========================================================
# MODELLO
# =========================================================

modello = Sequential([
    Embedding(input_dim=vocabolario, output_dim=32),
    LSTM(64),
    Dense(vocabolario, activation="softmax")
])

modello.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

modello.summary()

# =========================================================
# TRAINING
# =========================================================

# callback che blocca l'effetto del overfitting, quando il loss rimane uguale per 4 epoche
early_stopping = EarlyStopping(monitor="loss",
                               patience=int(args.patience),
                               restore_best_weights=True)

modello.fit(X, y, epochs=int(args.epochs), verbose=1, callbacks=[early_stopping])
modello.summary()

modello.save(str(output))

# =========================================================
# MAPPA INVERSA indice -> parola
# =========================================================

indice_parola = {indice: parola for parola, indice in tokenizer.word_index.items()}

with open(output_indices, "wt") as f:
    json.dump(indice_parola, f)