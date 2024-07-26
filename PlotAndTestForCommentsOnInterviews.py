import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency

os.makedirs('grafici_immagini_commentiYT', exist_ok=True)


def plot_sentiment(directory_path):
    """Funzione che crea un grafico a barre accostate che indica per ciascuna intervista il sentiment medio negativo e
    positivo (in percentuale) dei commenti per ciascuna intervista"""

    file_names, positive_sentiments, negative_sentiments = [], [], []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)
            file_names.append(os.path.splitext(filename)[0])
            positive_sentiments.append(df['Sentiment Positivo'].mean() * 100)
            negative_sentiments.append(df['Sentiment Negativo'].mean() * 100)

    results_df = pd.DataFrame({'File': file_names, 'Sentiment Positivo': positive_sentiments, 'Sentiment Negativo': negative_sentiments})

    x = range(len(file_names))
    width = 0.35
    fig, ax = plt.subplots()
    bars1 = ax.bar(x, results_df['Sentiment Positivo'], width, label='Sentiment Positivo', color='purple')
    bars2 = ax.bar([i + width for i in x], results_df['Sentiment Negativo'], width, label='Sentiment Negativo', color='grey')
    ax.set_ylabel('Valore Medio (%)')
    ax.set_title('Media del sentiment positivo e negativo\n dei commenti per intervista (%)')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(file_names, rotation=90)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    fig.savefig(os.path.join('grafici_immagini_commentiYT', 'sentiment_per_intervista.png'), dpi=300, bbox_inches='tight')


def plot_emotions(directory_path):
    """Funzione che crea un grafico a barre accostate per indicare l'occorrenza (in percentuale) di una certa emozione
    all'interno dei commenti per ciascuna intervista"""

    emotions = ['joy', 'sadness', 'anger', 'fear']
    file_names = []
    emotion_percentages = {emotion: [] for emotion in emotions}

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)
            file_names.append(os.path.splitext(filename)[0])

            emotion_count = df['Emozione Predetta'].str.strip("[]").str.replace("'", "").str.split(
                ",").explode().value_counts()
            total_comments = len(df)
            for emotion in emotions:
                percentage = (emotion_count.get(emotion, 0) / total_comments) * 100
                emotion_percentages[emotion].append(percentage)

    results_df = pd.DataFrame(emotion_percentages, index=file_names)

    fig, ax = plt.subplots()
    x = range(len(file_names))
    width = 0.2
    colors = {'joy': 'purple', 'sadness': 'grey', 'anger': 'blue', 'fear': 'red'}
    for i, emotion in enumerate(emotions):
        ax.bar([p + width * i for p in x], results_df[emotion], width, label=emotion, color=colors[emotion])
    ax.set_ylabel('Percentuale (%)')
    ax.set_title('Distribuzione delle emozioni\n dei commenti per intervista (%)')
    ax.set_xticks([p + width * 1.5 for p in x])
    ax.set_xticklabels(file_names, rotation=90)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    fig.savefig(os.path.join('grafici_immagini_commentiYT', 'emozioni_per_intervista.png'), dpi=300, bbox_inches='tight')


def test_negative_sentiment(directory_path):
    """Funzione che analizza i sentiment negativi delle interviste, li combina in un unico DataFrame, esegue alcune
     operazioni di pre-elaborazione sui dati ed effettua un'analisi della varianza (ANOVA) per determinare se ci
     sono differenze significative nei sentiment negativi tra i file considerati"""

    dfs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)
            df['File'] = os.path.splitext(filename)[0]
            df['Sentiment_Negativo'] = df['Sentiment Negativo'].astype(str).str.replace('[^0-9.-]', '',
                                                                                        regex=True).astype(float)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    model = ols('Sentiment_Negativo ~ C(File)', data=combined_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print("\nInterpretazione del risultato dell'ANOVA sui sentiment negativi delle interviste:")
    if anova_table['PR(>F)'][0] < 0.05:
        print("Esiste una differenza significativa nei sentiment negativi tra i commenti delle interviste prese in "
              "considerazione.")
    else:
        print("Non c'è evidenza sufficiente per concludere che ci siano differenze significative nei sentiment negativi "
              "tra le interviste considerate.")


def test_emotions(directory_path):
    """Funzione che analizza l'associazione tra emozioni (gioia, tristezza, rabbia, paura) e sentiment (positivo/negativo)
    nei file CSV in una directory. Conta le occorrenze di ogni emozione per commenti positivi e negativi, esegue un test
    del Chi-quadrato e riporta se c'è un'associazione significativa tra tipo di emozione e tipo di commento."""

    dfs = []
    emotions = ['joy', 'sadness', 'anger', 'fear']
    observed_frequencies = {'joy': {'positive': 0, 'negative': 0}, 'sadness': {'positive': 0, 'negative': 0},
                            'anger': {'positive': 0, 'negative': 0}, 'fear': {'positive': 0, 'negative': 0}}

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)
            dfs.append(df)
            for emotion in emotions:
                positive_count = df[df['Emozione Predetta'].str.contains(emotion) &
                                    (df['Sentiment Positivo'] > df['Sentiment Negativo'])].shape[0]
                negative_count = df[df['Emozione Predetta'].str.contains(emotion) &
                                    (df['Sentiment Negativo'] > df['Sentiment Positivo'])].shape[0]
                observed_frequencies[emotion]['positive'] += positive_count
                observed_frequencies[emotion]['negative'] += negative_count

    observed_values = [[observed_frequencies[emotion]['positive'] for emotion in emotions],
                       [observed_frequencies[emotion]['negative'] for emotion in emotions]]
    chi2, p, _, _ = chi2_contingency(observed_values)

    print("Test del Chi-quadro:")
    print(f"Statistiche del test: {chi2}")
    print(f"Valore p: {p}")

    if p < 0.05:
        print("Ci sono evidenze di un'associazione significativa tra il tipo di emozione e il tipo di commento"
              " (positivo o negativo).")
    else:
        print("Non ci sono evidenze di un'associazione significativa tra il tipo di emozione e il tipo di commento"
              " (positivo o negativo).")


if __name__ == "__main__":

    # percorso della cartella che contiene i file con i metadati dei commenti scaricati, il livello di sentiment
    # positivo e negativo, e l'emozione associata al commento
    directory_path = r'C:\Users\39392\Desktop\progettoWebAnalytics\file_commenti_csv'

    # chiamate delle funzioni per la creazione dei grafici e i vari test
    plot_sentiment(directory_path)
    plot_emotions(directory_path)
    test_negative_sentiment(directory_path)
    test_emotions(directory_path)


