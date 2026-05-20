import streamlit as st
import tensorflow as tf
import numpy as np
import re
import json
from argparse import ArgumentParser

from tensorflow.keras.models import load_model


parser = ArgumentParser()
parser.add_argument("-m", "--model", help="file di output del modello keras",
                    default="model/model.keras")
parser.add_argument("-i", "--indices", help="nome del file di testo da cui iniziare",
                    default="model/indices.json")
parser.add_argument("-l", "--len", help="lunghezza sequenza", default=5)
args = parser.parse_args()

# =========================
# CARICAMENTO MODELLO
# =========================

modello = load_model(args.model)

with open(args.indices, "rt") as f:
    indici = json.load(f)
indice_parola = {int(indice): parola for indice, parola in indici.items()}
parola_indice = {parola: indice for indice, parola in indice_parola.items()}

def text_to_sequence(frase):
    sequenza = []
    for parola in frase.split(" "):
        sequenza.append(parola_indice.get(parola))
    return sequenza

# =========================
# FUNZIONI
# =========================

def pulisci_frase(frase):
    frase = frase.lower()
    frase = re.sub(r"[^\w\sàèéìòù]", "", frase)
    frase = re.sub(r"\s+", " ", frase).strip()
    return frase

def predici_prossima_parola(frase):
    frase = pulisci_frase(frase)

    sequenza = text_to_sequence(frase)

    if len(sequenza) < args.len:
        return None

    sequenza = sequenza[-args.len:]
    sequenza = np.array([sequenza])

    predizione = modello.predict(sequenza, verbose=0)
    indice_predetto = np.argmax(predizione)

    return indice_parola.get(indice_predetto, None)

def genera_testo(seed_text, n_parole):
    testo_generato = seed_text

    for _ in range(n_parole):
        prossima = predici_prossima_parola(testo_generato)

        if prossima is None:
            break

        testo_generato += " " + prossima

    return testo_generato

# =========================
# INTERFACCIA WEB
# =========================

col1, col2 = st.columns([3, 1])

with col1:
    st.title("Un inferno di LLM")

with col2:
    st.image("images/dante.jpeg", width=120)



st.write("Inserisci una frase iniziale e il modello proverà a completarla.")

frase = st.text_input("Frase iniziale", "nel mezzo del cammin di")

numero_parole = st.slider(
    "Numero di parole da generare",
    min_value=1,
    max_value=20,
    value=10
)

if st.button("Genera testo"):
    risultato = genera_testo(frase, numero_parole)

    st.subheader("Risultato")
    st.write(risultato)