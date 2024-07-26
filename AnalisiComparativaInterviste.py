import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from collections import Counter


def t_test_sentiment_negativo(BiancaBalti_csv, StevenBasalari_csv):
    """Questa funzione esegue un t-test per confrontare i sentiment negativi tra due interviste. Passa come parametri,
    tramite percorso, i file CSV contenenti i dati delle due interviste analizzate"""

    df_BiancaBalti = pd.read_csv(BiancaBalti_csv)
    df_StevenBasalari = pd.read_csv(StevenBasalari_csv)

    # esecuzione del t-test per confrontare i sentiment negativi tra le due interviste
    t_stat, p_value = ttest_ind(df_BiancaBalti['Sentiment Negativo (%)'], df_StevenBasalari['Sentiment Negativo (%)'],
                                equal_var=False)

    # interpretazione del risultato
    alpha = 0.05  # Livello di significatività
    if p_value < alpha:
        risultato_sentiment = "Esiste una differenza significativa nei sentiment negativi tra le due interviste."
    else:
        risultato_sentiment = "Non c'è una differenza significativa nei sentiment negativi tra le due interviste."

    return t_stat, p_value, risultato_sentiment


def confronto_emozioni_predette(BiancaBalti_csv, StevenBasalari_csv):
    """Questa funzione confronta le distribuzioni delle emozioni predette tra due interviste utilizzando il
    test del chi-quadro. Passa come parametri, tramite percorso, i file CSV contenenti i dati delle due interviste
    analizzate"""

    df_BiancaBalti = pd.read_csv(BiancaBalti_csv)
    df_StevenBasalari = pd.read_csv(StevenBasalari_csv)

    # conteggio delle occorrenze delle emozioni predette per ciascuna intervista
    conteggio_emozioni_BiancaBalti = Counter(df_BiancaBalti['Emozione Predetta'].apply(eval).sum())
    conteggio_emozioni_StevenBasalari = Counter(df_StevenBasalari['Emozione Predetta'].apply(eval).sum())

    # creazione dei dataframe per le distribuzioni di frequenza delle emozioni predette
    df_conteggio_BiancaBalti = pd.DataFrame.from_dict(conteggio_emozioni_BiancaBalti, orient='index',
                                                      columns=['BiancaBalti'])
    df_conteggio_StevenBasalari = pd.DataFrame.from_dict(conteggio_emozioni_StevenBasalari, orient='index',
                                                         columns=['StevenBasalari'])

    # unione dei dataframe delle distribuzioni delle emozioni predette
    df_comparativo_emozioni = df_conteggio_BiancaBalti.join(df_conteggio_StevenBasalari, how='outer').fillna(0)

    # esecuzione del test del chi-quadro per confrontare le distribuzioni delle emozioni predette
    chi2, p_value, _, _ = chi2_contingency(df_comparativo_emozioni)

    # Interpretazione del risultato
    alpha = 0.05  # Livello di significatività
    if p_value < alpha:
        risultato_emozioni = "Esiste una differenza significativa nelle distribuzioni delle emozioni predette tra le due interviste."
    else:
        risultato_emozioni = "Non c'è una differenza significativa nelle distribuzioni delle emozioni predette tra le due interviste."

    return chi2, p_value, risultato_emozioni, df_comparativo_emozioni


def confronto_coerenza_topic(StevenBasalari_coerenza, BiancaBalti_coerenza):
    """Questa funzione confronta la coerenza dei topic tra le due interviste utilizzando per ciascuna di esse un numero
     che rappresenta il livello di coerenza complessivo dei topic. Il livello di coerenza va da 0 a 1. La funzione indica
     l'intervista che ha un maggiore livello di coerenza"""

    if StevenBasalari_coerenza > BiancaBalti_coerenza:
        risultato_coerenza = "I topic di Steven Basalari sono più coerenti rispetto a quelli di Bianca Balti"
    elif StevenBasalari_coerenza < BiancaBalti_coerenza:
        risultato_coerenza = "I topic di Bianca Balti sono più coerenti rispetto a quelli di Steven Basalari"
    else:
        risultato_coerenza = "I topic di Bianca Balti sono coerenti quanto quelli di Steven Basalari"

    return f"\n{risultato_coerenza}"


if __name__ == "__main__":
    BiancaBalti_csv = r'C:\Users\39392\Desktop\progettoWebAnalytics\file_csv\BiancaBalti_model_large.csv'
    StevenBasalari_csv = r'C:\Users\39392\Desktop\progettoWebAnalytics\file_csv\StevenBasalari_model_large.csv'

    # esegue la funzione per confrontare i sentiment negativi tra le due interviste
    risultato_t, p_value_t, risultato_confronto_sentiment = t_test_sentiment_negativo(BiancaBalti_csv,
                                                                                      StevenBasalari_csv)

    # stampa i risultati del test t per il sentiment negativo
    print("Statistiche del t-test per il sentiment negativo:")
    print("Statistica t:", risultato_t)
    print("Valore p_value:", p_value_t)
    print(risultato_confronto_sentiment)

    # esegue la funzione per confrontare le emozioni predette tra le due interviste
    risultato_chi2, p_value_chi2, risultato_confronto_emozioni, df_comparativo_emozioni = confronto_emozioni_predette(
        BiancaBalti_csv, StevenBasalari_csv)

    # stampa i risultati del test del chi-quadro per le distribuzioni delle emozioni predette
    print("\nRisultati del test del chi-quadro per confronto delle emozioni predette:")
    print("Chi-quadro:", risultato_chi2)
    print("Valore p_value:", p_value_chi2)
    print(risultato_confronto_emozioni)

    # il livello di coerenza per ciascuna intervista preso dal file TopicModelling.py, passato come parametro degli
    # argomenti della funzione 'confronto_coerenza_topic' che viene chiamata
    print(confronto_coerenza_topic(0.3883500353042295, 0.4253803064229412))
