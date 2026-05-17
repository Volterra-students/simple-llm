import streamlit as st
import tensorflow as tf
import numpy as np
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# =========================
# PREPARAZIONE MODELLO
# =========================

testo = """
Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura,
ché la diritta via era smarrita.
Ahi quanto a dir qual era è cosa dura
esta selva selvaggia e aspra e forte
che nel pensier rinova la paura!
Tant' è amara che poco è più morte;
ma per trattar del ben ch'i' vi trovai,
dirò de l'altre cose ch'i' v'ho scorte.
Io non so ben ridir com' i' v'intrai,
tant' era pien di sonno a quel punto
che la verace via abbandonai.
Ma poi ch'i' fui al piè d'un colle giunto,
là dove terminava quella valle
che m'avea di paura il cor compunto,
guardai in alto e vidi le sue spalle
vestite già de' raggi del pianeta
che mena dritto altrui per ogne calle.
Allor fu la paura un poco queta,
che nel lago del cor m'era durata
la notte ch'i' passai con tanta pieta.
E come quei che con lena affannata,
uscito fuor del pelago a la riva,
si volge a l'acqua perigliosa e guata,
così l'animo mio, ch'ancor fuggiva,
si volse a retro a rimirar lo passo
che non lasciò già mai persona viva.
Poi ch'èi posato un poco il corpo lasso,
ripresi via per la piaggia diserta,
sì che 'l piè fermo sempre era 'l più basso.
Ed ecco, quasi al cominciar de l'erta,
una lonza leggiera e presta molto,
che di pel macolato era coverta;
e non mi si partia dinanzi al volto,
anzi 'mpediva tanto il mio cammino,
ch'i' fui per ritornar più volte vòlto.
Temp' era dal principio del mattino,
e 'l sol montava 'n sù con quelle stelle
ch'eran con lui quando l'amor divino
mosse di prima quelle cose belle;
sì ch'a bene sperar m'era cagione
di quella fiera a la gaetta pelle
l'ora del tempo e la dolce stagione;
ma non sì che paura non mi desse
la vista che m'apparve d'un leone.
"""

testo = testo.lower()
testo = re.sub(r"[^\w\sàèéìòù]", "", testo)
testo = re.sub(r"\s+", " ", testo).strip()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([testo])

sequenza_tokenizzata = tokenizer.texts_to_sequences([testo])[0]
vocabolario = len(tokenizer.word_index) + 1
indice_parola = {indice: parola for parola, indice in tokenizer.word_index.items()}

lunghezza_sequenza = 5

X = []
y = []

for i in range(len(sequenza_tokenizzata) - lunghezza_sequenza):
    X.append(sequenza_tokenizzata[i:i + lunghezza_sequenza])
    y.append(sequenza_tokenizzata[i + lunghezza_sequenza])

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=vocabolario)

@st.cache_resource
def crea_e_addestra_modello():
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

    modello.fit(X, y, epochs=300, verbose=0)
    return modello

modello = crea_e_addestra_modello()

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

    sequenza = tokenizer.texts_to_sequences([frase])[0]

    if len(sequenza) < lunghezza_sequenza:
        return None

    sequenza = sequenza[-lunghezza_sequenza:]
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