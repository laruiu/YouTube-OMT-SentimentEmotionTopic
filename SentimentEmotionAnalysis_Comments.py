import re
import os
import pandas as pd
from transformers import pipeline
from feel_it import EmotionClassifier
import csv


def clean_comment(comment):
    """Funzione che controlla i commenti dei file csv, rimuove sequenze e tag HTML (restituisce il carattere originale
    in alcuni casi), e rimuove e le emoji presenti"""

    EMOJI_PATTERN = re.compile("["
                               "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "\U0001F600-\U0001F64F"  # emoticons
                               "\U0001F680-\U0001F6FF"  # transport & map symbols
                               "\U0001F700-\U0001F77F"  # alchemical symbols
                               "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               "\U0001FA00-\U0001FA6F"  # Chess Symbols
                               "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               "\U00002702-\U000027B0"  # Dingbats
                               "\U000024C2-\U0001F251"
                               "]")

    if isinstance(comment, str):
        comment = comment.replace("&quot;", "").replace("&#39;", "'")
        comment = re.sub(r'<[^>]*>', '', comment)
        comment = EMOJI_PATTERN.sub(r' ', comment)  # per evitare che le parole attaccate alle emoji si attacchino
        comment = comment.replace("  ", "").strip()

        if len(comment) < 2:
            comment = None

    return comment


def process_files(path):
    """Funzione che elabora i file CSV del path definito, dove si trovano i file con i metadati scaricati dei video
    delle interviste. Vengono eliminati le righe che hanno i commenti vuoti e le modifiche vengono salvate nei file CSV
    di origine"""

    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)

            df = pd.read_csv(file_path)
            df['Commento'] = df['Commento'].apply(clean_comment)
            df = df.dropna(subset=['Commento'])  # elimina i commenti vuoti

            df.to_csv(file_path, index=False)

            print(f"I commenti del file {file} sono stati modificati")


def sentiment_and_emotion_analysis(path):
    """La funzione permette l'analisi del sentiment e delle emozioni dei commenti di video su YouTube. Sono stati usati
    modelli pre-addestrati che fanno predizione del sentiment (positiva e negativa) e delle emozioni (indicate come: joy,
    sadness, fear, anger). I dati per ciascun commento verranno salvati nel file csv di origine"""

    # Caricamento del modello per l'analisi del sentiment del testo in lingua italiana
    sentiment_classifier = pipeline("text-classification", model='MilaNLProc/feel-it-italian-sentiment', top_k=2)

    # Caricamento della funzione per l'analisi delle emozioni del testo in lingua italiana
    emotion_classifier = EmotionClassifier()

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                content = list(reader)

            header = content[0]
            header.extend(["Sentiment Positivo", "Sentiment Negativo", "Emozione Predetta"])
            new_content = [header]

            for row in content[1:]:
                segment = row[1]
                try:
                    sentiment_predictions = sentiment_classifier(segment)[0]
                    pos_score = next((item['score'] for item in sentiment_predictions if item['label'] == 'positive'),0)
                    neg_score = next((item['score'] for item in sentiment_predictions if item['label'] == 'negative'),0)
                    predicted_emotion = emotion_classifier.predict([segment])
                    row.extend([pos_score, neg_score, predicted_emotion])
                    new_content.append(row)
                except Exception as e:
                    print(f"Errore durante l'analisi della frase: {e}")

            # aggiornamento del file
            new_file_path = os.path.join(path, file_name)
            with open(new_file_path, "w", encoding="utf-8", newline='') as new_file:
                writer = csv.writer(new_file)
                writer.writerows(new_content)

            print(f"Il contenuto del file {file_name} è stato modificato e salvato come {new_file_path}")


if __name__ == "__main__":

    # Percorso della cartella che contiene i file CSV con tutti i commenti dei video delle interviste su Youtube
    path = r'C:\Users\39392\Desktop\progettoWebAnalytics\file_commenti_csv'

    # chiamata della funzione che si occupa dell'elaborazione dei file, che ha parametro il path della cartella con
    # i file CSV che hanno i dati estratti
    process_files(path)

    # chiamata della funzione che permette l'analisi del sentiment e delle emozioni dei commenti
    sentiment_and_emotion_analysis(path)
