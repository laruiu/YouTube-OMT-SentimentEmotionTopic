import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists(r"C:\Users\39392\Desktop\progettoWebAnalytics\Grafici_Immagini"):
    os.makedirs('Grafici_Immagini')


class GraficiSentimentEmotion:
    """Classe creata per gestire la creazione di diversi grafici basati sul dataset di frasi che compongono il testo
    audio delle interviste, in cui viene espressa la percentuale positiva e negativa del sentiment, e l'emozione
    predetta per ciascuna frase."""

    def __init__(self, path, nome):
        """Il costruttore della classe viene inizializzato con il percorso del file CSV che contiene il dataset
        con le informazioni sul sentiment e le emozioni di ciascuna frase che compone il testo dell'intervista, e con
        il nome dell'intervistato. Viene creato anche l'attributo df che è utile per richiamare la lettura dei file CSV
        tramite il percorso specificato."""

        self.path = path
        self.nome = nome
        try:
            self.df = pd.read_csv(self.path)
        except FileNotFoundError:
            print(f"Il file csv non è trovato")
            self.df = None

    def graficoBarreSentiment(self):
        """Metodo che analizza il sentiment medio delle frasi nell'intervista. Attraverso un grafico a barre viene
         mostrata la media percentuale del sentiment negativo e positivo delle frasi. Viene espresso anche il totale
         delle frasi per intervista."""

        frasi = self.df['Frase']
        sentiment_pos = self.df['Sentiment Positivo (%)']
        sentiment_neg = self.df['Sentiment Negativo (%)']

        print("Totale frasi:", len(frasi))
        tot_sentiment_pos = round(sentiment_pos.mean() * 100, 2)
        tot_sentiment_neg = round(sentiment_neg.mean() * 100, 2)

        print("Percentuale approssimativa Sentiment Positivo su tutte le frasi:", tot_sentiment_pos, "%")
        print("Percentuale approssimativa Sentiment Negativo su tutte le frasi:", tot_sentiment_neg, "%")

        labels = [f'Sentiment Positivo\n{tot_sentiment_pos}%', f'Sentiment Negativo\n{tot_sentiment_neg}%']
        values = [tot_sentiment_pos, tot_sentiment_neg]
        plt.figure(figsize=(6, 10))
        plt.bar(labels, values, color=['purple', 'grey'])
        plt.xlabel('Sentiment')
        plt.ylabel('Percentuale del Sentiment')
        plt.title(f"Intervista a {self.nome} - Grafico a barre su percentuale media \n"
                  f"dei sentiment positivo e negativo su {len(frasi)} frasi")
        plt.ylim(0, 100)
        plt.show()
        plt.savefig(os.path.join('Grafici_Immagini', f'{self.nome}_grafico_barre_sentiment.png'), bbox_inches='tight')

    def graficoSentimentSerieTemporale(self):
        """Metodo che esamina le variazioni del sentiment positivo e negativo nel tempo, calcolando la media dei due
        tipi di sentiment ogni 30 frasi, a parte l'ultimo gruppo che è costituito da meno frasi. Viene rappresentato
        graficamente attraverso barre accostate, mostrando la percentuale di negatività e positività presente durante
        lo svolgimento dell'intervista."""

        frasi = self.df['Frase']
        sentiment_pos = self.df['Sentiment Positivo (%)']
        sentiment_neg = self.df['Sentiment Negativo (%)']

        dimensione_gruppo = 30

        num_gruppi = len(frasi) // dimensione_gruppo + 1

        print("Numero gruppi per rappresentazione grafico a barre accostate:", num_gruppi)

        media_sentiment_pos = []
        media_sentiment_neg = []

        for i in range(num_gruppi):
            idx_inizio = i * dimensione_gruppo
            idx_fine = min((i + 1) * dimensione_gruppo,
                           len(frasi))
            media_pos = sentiment_pos[idx_inizio:idx_fine].mean()
            media_neg = sentiment_neg[idx_inizio:idx_fine].mean()
            media_sentiment_pos.append(media_pos)
            media_sentiment_neg.append(media_neg)

        plt.figure(figsize=(12, 8))

        x = range(num_gruppi)

        larghezza = 0.4

        plt.bar(x, media_sentiment_pos, width=larghezza, label='Sentiment Positivo', color='purple', align='center')
        plt.bar([i + larghezza for i in x], media_sentiment_neg, width=larghezza, label='Sentiment Negativo',
                color='grey',
                align='center')

        plt.xlabel('Gruppi di Frasi')
        plt.ylabel('Valore Sentiment per frase da 0 a 100')
        plt.xticks([i + larghezza / 2 for i in x], [f'{i + 1}' for i in x])
        plt.title(
            f"Intervista a {self.nome}: media sentiment positivo e negativo"
            f" per gruppi di frasi (ogni {dimensione_gruppo} frasi)")
        plt.legend()

        plt.show()
        plt.savefig(os.path.join('Grafici_Immagini', f'{self.nome}_grafico_serie_temporale_sentiment.png'),
                    bbox_inches='tight')

    def graficoTortaEmozioni(self):
        """Metodo che analizza e visualizza, tramite un grafico a torta, la presenza media di un certa emozione nelle
        frasi dell'intervista. Il valore medio è espresso in percentuale"""

        df = self.df

        conteggio_emozioni = {}
        totale_emozioni = 0

        for emozioni_str in self.df['Emozione Predetta']:
            emozioni = eval(emozioni_str)
            totale_emozioni += len(emozioni)
            for emozione in emozioni:
                if emozione in conteggio_emozioni:
                    conteggio_emozioni[emozione] += 1
                else:
                    conteggio_emozioni[emozione] = 1

        percentuali_medie = {emozione: (conteggio / totale_emozioni * 100) for emozione, conteggio in
                             conteggio_emozioni.items()}

        emozioni = list(percentuali_medie.keys())
        percentuali = list(percentuali_medie.values())

        colori_emozioni = {
            'joy': 'purple',
            'sadness': 'grey',
            'fear': 'green',
            'anger': 'salmon'
        }

        colors = [colori_emozioni[emozione] for emozione in emozioni]

        plt.figure(figsize=(8, 8))
        plt.pie(percentuali, labels=emozioni, autopct='%1.1f%%', startangle=180, colors=colors)
        plt.title(f"Intervista a {self.nome}: percentuale media delle emozioni")
        plt.axis('equal')
        plt.show()
        plt.savefig(os.path.join('Grafici_Immagini', f'{self.nome}_grafico_torta_emozioni.png'), bbox_inches='tight')

    def graficoSerieTemporaleEmozioni(self):
        """Metodo che analizza dal punto vista temporale la presenta ci un certo quantitativo di emozioni ogni 30 frasi,
         a parte l'ultimo gruppo composto da un numero minore. Attraverso dei marcatori (uno diverso per ogni emozione)
         è possibile vedere l'andamento di ciascuna emozione durante l'intervista."""

        df = self.df
        frasi = self.df['Frase']

        dimensione_gruppo = 30

        num_gruppi = len(frasi) // dimensione_gruppo + 1

        print("Numero gruppi per rappresentazione grafica per serie temporale sulle emozioni:", num_gruppi)

        occorrenze_emozioni = {emozione: [0] * num_gruppi for emozione in ['joy', 'sadness', 'fear', 'anger']}

        for i in range(num_gruppi):
            idx_inizio = i * dimensione_gruppo
            idx_fine = min((i + 1) * dimensione_gruppo, len(df))
            gruppo_frasi = df['Emozione Predetta'][idx_inizio:idx_fine]

            for emozioni_str in gruppo_frasi:
                emozioni = eval(emozioni_str)
                for emozione in emozioni:
                    occorrenze_emozioni[emozione][i] += 1

        for emozione in occorrenze_emozioni:
            for i in range(num_gruppi):
                occorrenze_emozioni[emozione][i] = (occorrenze_emozioni[emozione][i] / dimensione_gruppo) * 100

        colori = {'joy': 'purple', 'sadness': 'grey', 'fear': 'green', 'anger': 'salmon'}
        markers = {'joy': 'o', 'sadness': 's', 'fear': '^', 'anger': 'd'}

        plt.figure(figsize=(12, 8))

        for emozione, occorrenze in occorrenze_emozioni.items():
            x = range(1, num_gruppi + 1)
            plt.plot(x, occorrenze, label=emozione, color=colori[emozione], marker=markers[emozione], markersize=8)

        plt.xlabel('Gruppi di Frasi')
        plt.ylabel('Percentuale di Frasi con Emozione')
        plt.xticks([i for i in range(1, num_gruppi + 1)], [f'{i}' for i in range(1, num_gruppi + 1)])
        plt.title(
            f"Intervista a {self.nome}: Percentuale di tipo di emozione per gruppi di frasi (ogni {dimensione_gruppo} frasi)")
        plt.legend()

        plt.show()
        plt.savefig(os.path.join('Grafici_Immagini', f'{self.nome}_grafico_serie_temporale_emozioni.png'),
                    bbox_inches='tight')


# Si istanziano oggetti per la classe GraficiSentimentEmotion che hanno come attributo il percorso del dataset di
# ciascuna intervista e il nome della persona intervistata
StevenBasalari = GraficiSentimentEmotion(
    path=r"C:\Users\39392\Desktop\progettoWebAnalytics\file_csv\StevenBasalari_model_large.csv",
    nome="Steven Basalari")

BiancaBalti = GraficiSentimentEmotion(
    path=r"C:\Users\39392\Desktop\progettoWebAnalytics\file_csv\BiancaBalti_model_large.csv",
    nome="Bianca Balti"
)

if __name__ == "__main__":
    # Richiamo oggetti della classe e successivamente i vari metodi

    print("Steven Basalari:")
    StevenBasalari.graficoBarreSentiment()
    StevenBasalari.graficoSentimentSerieTemporale()
    StevenBasalari.graficoTortaEmozioni()
    StevenBasalari.graficoSerieTemporaleEmozioni()

    print("Bianca Balti:")
    BiancaBalti.graficoBarreSentiment()
    BiancaBalti.graficoSentimentSerieTemporale()
    BiancaBalti.graficoTortaEmozioni()
    BiancaBalti.graficoSerieTemporaleEmozioni()
