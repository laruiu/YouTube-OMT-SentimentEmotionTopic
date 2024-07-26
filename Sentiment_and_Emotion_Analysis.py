import os
import re
import pandas as pd
from transformers import pipeline
from feel_it import EmotionClassifier


class SentimentAndEmotionAnalysis:
    """Classe per l'analisi del sentiment e delle emozioni degli audio testo delle interviste di One More Time.
    Verrà assegnato un punteggio percentuale positivo e negativo per il sentiment, e verrà predetta un'emozione per
    ciascuna frase del corpus basata su regole lessicali. In mancanza di punteggiatura, data dal modello che ha
    convertito l'audio in testo, viene considerata una lunghezza massima di 300 caratteri per frase, quindi l'analisi
    verrà effettuata più precisamente su segmenti di testo. I risultati saranno esportati in un file CSV."""

    BASE_PATH = os.path.abspath(os.path.dirname(__file__))  # directory dello script

    def __init__(self, path_file_txt):
        """ Metodo che inizializza la classe SentimentAndEmotionAnalysis, chiamato ogni volta che viene creato
         un nuovo oggetto. Inizializza l'attributo path_file_txt che indica il percorso del file testo da analizzare"""

        self.path_file_txt = path_file_txt

    def preprocess_corpus(self):
        """Metodo che legge il file di testo del path specificato e che preprocessa il corpus del testo.
        Rimuove i newline e aggiunge un punto esclamativo alla fine dell'ultima riga per assicurarsi che le frasi
        possano essere divise secondo le formali regole  del lessico. Vista la mancata punteggiatura in molte parti
        del testo, se una frase supera i 300 caratteri, viene suddivisa in segmenti con limite 300 per mantenere
        l'integrità sintattica"""

        with open(self.path_file_txt, 'r', encoding='utf-8') as file_txt:
            rows = file_txt.readlines()
            rows[-1] = rows[-1].rstrip() + '!'  # Aggiunge '!' alla fine dell'ultima riga della lista rows

        corpus = " ".join([row.replace("\n", "") for row in rows])
        pattern = r'[^.?!]*(?:\.{3}|[.?!])'
        sentences = re.findall(pattern, corpus)

        preprocessed_segments = []
        for sentence in sentences:
            if len(sentence) <= 300:
                preprocessed_segments.append(sentence.strip())
            else:
                while len(sentence) > 300:
                    last_space_idx = sentence.rfind(' ', 0, 300)
                    if last_space_idx == -1:
                        last_space_idx = 300
                    preprocessed_segments.append(sentence[:last_space_idx].strip())
                    sentence = sentence[last_space_idx:].strip()
                preprocessed_segments.append(sentence.strip())

        return preprocessed_segments

    def sentiment_and_emotion_analysis(self):
        """Metodo che permette l'analisi del sentiment e delle emozioni di ciascun segmento di testo. Sono stati usati
        modelli pre-addestrati che fanno predizione del sentiment (positiva e negativa, indicata in percentuale) e delle
        emozioni (indicate come: joy, sadness, fear, anger).
        I dati per ciascun segmento di testo verrà salvato in un file csv all'interno di una cartella specifica"""

        # Caricamento del modello per l'analisi del sentiment del testo in lingua italiana
        sentiment_classifier = pipeline("text-classification", model='MilaNLProc/feel-it-italian-sentiment', top_k=2)

        # Caricamento della funzione per l'analisi delle emozioni del testo in lingua italiana
        emotion_classifier = EmotionClassifier()

        preprocessed_segments = self.preprocess_corpus()

        results = []
        for segment in preprocessed_segments:
            try:
                sentiment_predictions = sentiment_classifier(segment)[0]
                pos_score = next((item['score'] for item in sentiment_predictions if item['label'] == 'positive'), 0)
                neg_score = next((item['score'] for item in sentiment_predictions if item['label'] == 'negative'), 0)
                predicted_emotion = emotion_classifier.predict([segment])

                results.append({
                    'Frase': segment,
                    'Sentiment Positivo (%)': pos_score,
                    'Sentiment Negativo (%)': neg_score,
                    'Emozione Predetta': predicted_emotion
                })
            except Exception as e:
                print(f"Errore durante l'analisi della frase: {e}")

        df = pd.DataFrame(results)

        output_folder = os.path.join(self.BASE_PATH, 'file_csv')
        os.makedirs(output_folder, exist_ok=True)
        output_csv_path = os.path.join(output_folder, os.path.splitext(os.path.basename(self.path_file_txt))[0] + ".csv")
        df.to_csv(output_csv_path, index=False)

        return f"File CSV con i risultati dell'analisi del sentiment e delle emozioni creato con successo"


# Creazione istanze per la classe SentimentAndEmotionAnalysis che richiama l'attributo path_file_txt del costruttore
StevenBasalari = SentimentAndEmotionAnalysis(path_file_txt=r"C:\Users\39392\Desktop\progettoWebAnalytics\file_txt\BiancaBalti_model_large.txt")
BiancaBalti = SentimentAndEmotionAnalysis(path_file_txt=r"C:\Users\39392\Desktop\progettoWebAnalytics\file_txt\StevenBasalari_model_large.txt")

if __name__ == "__main__":
    # Chiamata classe con il metodo sentiment_and_emotion_analysis()
    print("Analisi per l'intervista di Steven Basalari: ", end="")
    print(StevenBasalari.sentiment_and_emotion_analysis())
    print("\n")
    print("Analisi per l'intervista di Bianca Balti: ", end="")
    print(BiancaBalti.sentiment_and_emotion_analysis())

