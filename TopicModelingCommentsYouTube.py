import os
import re
import random
import numpy as np
import pandas as pd
import spacy
import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim_models
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from gensim.models import CoherenceModel


class TopicModel:
    """La classe TopicModel è stata costruita per eseguire l'analisi dei topic sui file CSV contenenti i commenti delle
    interviste. I commenti vengono elaborati attraverso un processo di pulizia, a cui segue l'uso del modello LDA per
    individuare i topic principali per tutti i commenti di ogni intervista"""


    def __init__(self, file_path, output_folder):
        """Il costruttore della classe carica il file CSV di ogni intervista in un DataFrame di Pandas. Estrae il titolo
        per ogni file e inizializza gli attributi per il modello LDA e i topic con None per il momento. Carica il
        modello in italiano per la libreria Spacy e le stopwords predefinite. Viene anche definita una lista aggiuntiva
        da aggiungere alle stopwords"""

        self.df = pd.read_csv(file_path)
        self.csv_title = os.path.splitext(os.path.basename(file_path))[0]
        self.lda_model = None
        self.topics = None
        self.nlp = spacy.load('it_core_news_sm')
        self.stop_words_italian = self.nlp.Defaults.stop_words
        self.additional_stopwords = ['come', 'quando', 'sono', 'abbiamo', 'avete', 'hanno', 'sia', 'resto', 'vicenda',
                                     'siamo', 'siete', 'essere', 'avere', 'fare', 'dire', 'detto', 'fatto', 'del',
                                     'lha', 'lucere', 'dovere', 'totale', 'english', 'they', 'have', 'him', 'her',
                                     'with', 'nonaltola', 'aprile', 'grazie', 'dei', 'della', 'delle', 'nell', 'nella',
                                     'nelle', 'nello', 'negli', 'come', 'about', 'conto', 'volere', 'stare', 'spessore',
                                     'genere', 'scusa', 'mano', 'versione', 'paio', 'periodo', 'lupo', 'anch', 'gatto',
                                     'lha' 'abbraccio', 'cuore', 'abbraccio', 'persona', 'domanda', 'risposta', 'parola',
                                     'grazia', 'spazio', 'italiano', 'parentesi', 'ammazza', 'subtitles', 'fabrizio', 'vista',
                                     'mammabellissima', 'will', 'niro', 'grazi', 'metà',  'sunsplash', 'livello', 'parola',
                                     'laltro', 'quando', 'alla', 'allo', 'alle', 'agli', 'senza', 'sotto', 'sopra',
                                     'mora', 'bocca', 'resto', 'venire', 'gesto', 'situazione', 'surry', 'braccio',
                                     'tipo', 'los', 'lodo', 'boh', 'sai', 'cioè', 'cavolo', 'lo', 'adoro', 'cerare',
                                     'bello', 'mettere', 'totale', 'concentro', 'minuto', 'mese', 'settimana',
                                     'sacco', 'vero', 'cosa', 'punto', 'giusto', 'super', 'perchè', 'finché', 'vabbè',
                                     'ragazzo', 'uomo', 'donna', 'signore', 'nonantola', 'conor', 'maynard', 'boccone',
                                     'inizio', 'fine']
        self.output_folder = output_folder

    def preprocess_texts(self):
        """In questo metodo viene applicata la  funzione tokenize_text per tokenizzare e preprocessare il testo, che
        verrà definita successivamente, per ogni commento di ciascun file CSV. IIl risultato viene memorizzato in una
        nuova colonna chiamata 'Tokens'"""

        self.df['Tokens'] = self.df['Commento'].apply(self.tokenize_text)

    def tokenize_text(self, text):
        """In questo metodo avviene il preprocessamento dei commenti. Vengono sostituiti gli apostrofi con degli spazi,
        viene eliminata la punteggiatura e i numeri, e il testo dei commenti diventa tutto minuscolo. Attraverso il
        modello linguistico di Spacy si tokenizza il testo. Viene creata una lista vuota per memorizzare i token che
        hanno più di tre caratteri e che non sono pronomi, articoli, verbi, preposizioni, avverbi o aggettivi. Alla fine
        i token vengono lemmatizzati e aggiunti alla lista 'tokens'"""

        text = re.sub(r"(\w+)['’](\w+)", r"\1 \2", text.lower())
        text = re.sub(r'[^\w\s]', '', text.lower())
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not token.is_punct and not token.like_num and len(token.text) > 3 and token.pos_ not in ['PRON', 'DET',
                                                                                                        'VERB', 'ADP',
                                                                                                        'ADV', 'ADJ']:
                lemma = token.lemma_
                if lemma not in self.stop_words_italian and lemma not in self.additional_stopwords:
                    tokens.append(lemma)
        return tokens

    def lda(self):
        """Nel metodo lda si esegue l'addestramento del modello LDA sui dati preprocessati e si memorizzano i risultati
        negli attributi della classe. Prima vengono impostati dei seed fissi per la riproducibilità dei risultati, poi
        vengono preparati i dati per il modello LDA, convertendo la lista di token in un corpus in formato bag-of-words
        e creando un dizionario che mappa le parole agli ID. Quindi, viene addestrato il modello LDA utilizzando il
        corpus bag-of-words e il dizionario. I topic individuati dal modello vengono memorizzati nell'attributo
        self.topics come un dizionario che associa l'indice del topic alla stringa che rappresenta il topic stesso (le
        10 parole più rilevanti). Infine, vengono stampati i topic individuati per il file CSV corrente."""

        random.seed(10)
        np.random.seed(10)
        tokens = self.df['Tokens'].tolist()
        dict = corpora.Dictionary(tokens)
        bow_corpus = [dict.doc2bow(t) for t in tokens]
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=bow_corpus,
            id2word=dict,
            num_topics=5,
            random_state=10,
            update_every=1,
            chunksize=100,
            passes=200,
            alpha='auto',
            iterations=3000,
            per_word_topics=True
        )
        self.topics = {idx: topic for idx, topic in self.lda_model.print_topics(num_words=10)}
        print(f"Topic per intervista tratti dal file CSV di {self.csv_title}: ")
        print(self.topics)

    def save_visualization(self):
        """In questo metodo si salva la visualizzazione interattiva del modello LDA preparata in precedenza in un file
        HTML"""

        name_html = os.path.join(self.output_folder, f'{self.csv_title}_Visualizzazione_LDA.html')
        bow_corpus = [self.lda_model.id2word.doc2bow(t) for t in self.df['Tokens']]
        lda_html = pyLDAvis.gensim_models.prepare(self.lda_model, bow_corpus, self.lda_model.id2word)
        pyLDAvis.save_html(lda_html, name_html)

    def topic_plot(self):
        """Questo metodo crea un grafico che visualizza i 5 topic individuati dal modello LDA per ogni file CSV. Per
        ogni topic, viene generato un grafico a barre che mostra i token e le loro probabilità rispetto a quel topic.
        Il grafico viene salvato come immagine PNG."""

        name_plot = os.path.join(self.output_folder, f'{self.csv_title}_Grafico_Topic.png')
        plt.figure(figsize=(30, 30))
        sns.set_theme(style="darkgrid")
        for i in range(5):
            df = pd.DataFrame(self.lda_model.show_topic(i), columns=['Token', 'Probabilità']).set_index('Token')
            df = df.sort_values('Probabilità')
            plt.subplot(5, 2, i + 1)
            plt.title('Topic ' + str(i))
            sns.barplot(x='Probabilità', y=df.index, data=df, palette='Greens_d')
            plt.xlabel('Probabilità')
        plt.suptitle(f'Modello di Topic - Commenti da {self.csv_title}', fontsize=50)
        plt.savefig(name_plot)
        plt.show()
        print("Il topic plot è disponibile!")

    def word_cloud(self):
        """Questo metodo crea un grafico, salvato in formato PNG, che visualizza le word clouds per ciascuno dei 5 topic
        individuati dal modello LDA. Ogni word cloud mostra i termini più rappresentativi di un topic, con la dimensione
        delle parole proporzionale alla loro probabilità (importanza) nel topic."""

        name_cloud = os.path.join(self.output_folder, f'{self.csv_title}_Word_Cloud.png')
        plt.figure(figsize=(30, 30))
        for i in range(5):
            df = pd.DataFrame(self.lda_model.show_topic(i), columns=['Token', 'Probabilità']).set_index('Token')
            df = df.sort_values('Probabilità', ascending=False)
            wordcloud = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(
                dict(df['Probabilità']))
            plt.subplot(5, 2, i + 1)
            plt.title('Topic ' + str(i))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
        plt.suptitle(f'Word Clouds - Topics - {self.csv_title}', fontsize=50)
        plt.savefig(name_cloud)
        plt.show()
        print("Le word cloud per ogni topic sono disponibili!")

    def calculate_coherence(self):
        """Questo metodo calcola il livello di coerenza, da 0 a 1, dei topic generati dal modello LDA. Questo punteggio
         misura quanto i termini all'interno dei singoli topic siano semanticamente coerenti."""

        tokens = self.df['Tokens'].tolist()
        dict = corpora.Dictionary(tokens)
        topics = [[token for token, weight in self.lda_model.show_topic(topic_id)] for topic_id in
                  range(self.lda_model.num_topics)]
        coherence_model = CoherenceModel(topics=topics, texts=tokens, dictionary=dict, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        return coherence_score

    def run_topic_analysis(self):
        """Con questo metodo si esegue l'intero processo di analisi dei topic sui commenti e restituisce il livello di
        coerenza dei topic per valutare la qualità del modello"""

        self.preprocess_texts()
        self.lda()
        self.save_visualization()
        self.topic_plot()
        self.word_cloud()

        coherence_score = self.calculate_coherence()
        print(f"Coerenza dei Topic: {coherence_score}")
        return coherence_score


if __name__ == "__main__":
    # cartella contenente i file CSV delle interviste con i rispettivi commenti
    input_folder = r"C:\Users\39392\Desktop\progettoWebAnalytics\file_commenti_csv"
    # Cartella dove verranno salvati i risultati dell'analisi. Viene creata se non esiste già
    output_folder = r"C:\Users\39392\Desktop\progettoWebAnalytics\risultati_LDA_commenti"
    os.makedirs(output_folder, exist_ok=True)
    # creazione di un elenco di tutti i file CSV presenti nella cartella di input
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # lista dove viene inserito il livello di coerenza per ogni file intervista
    coherence_scores = []
    # passaggi da effettuare per ogni file CSV con i rispettivi commenti per eseguire l'analisi di topic
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        topic_model = TopicModel(file_path, output_folder)
        coherence_score = topic_model.run_topic_analysis()
        coherence_scores.append({'File': csv_file, 'Coerenza': coherence_score})

    # creazione di un DataFrame con i livelli di coerenza dei topic per ciascuna intervista
    coherence_df = pd.DataFrame(coherence_scores)
    # creazione di un file CSV dove vengono inseriti i risultati del DataFrame creato prima di questo commento
    coherence_df.to_csv(os.path.join(output_folder, 'coerenza_topic.csv'), index=False)
    print("Analisi completata e risultati salvati!")

   # # funzione creata successivamente alla parte precedente, inserita nello script per evitare di creare un altro file

    def process_and_plot_coherence_scores():
        """Funzione creata per ordinare (ordine decrescente) il livello di coerenza dei topic delle interviste,
        attraverso il percorso del file CSV creato appositamente per la coerenza, contenente le informazioni che
        determinato per ciascuna intervista un punteggio che può andare tra 0 e 1, che stabilisce coerenza tra i topic
        estrappolati tramite il modello LDA. Creazione di un grafico boxplot che indica il livello di distribuzione
        dei vari livelli di coerenza"""

        output_folder = r"C:\Users\39392\Desktop\progettoWebAnalytics\risultati_LDA_commenti"
        save_image_path = r"C:\Users\39392\Desktop\progettoWebAnalytics\grafici_immagini_commentiYT"
        file_path = os.path.join(output_folder, 'coerenza_topic.csv')
        if not os.path.isfile(file_path):
            print(f"Il file '{file_path}' non esiste.")
            return

        coherence_df = pd.read_csv(file_path)
        coherence_df['File'] = coherence_df['File'].str.replace('.csv', '')
        coherence_df = coherence_df.sort_values(by='Coerenza', ascending=False)
        print("\nPunteggi di coerenza in ordine decrescente:")
        for file, coherence in zip(coherence_df['File'], coherence_df['Coerenza']):
            print(f"{file}: {coherence}")

        sns.set_style("darkgrid")
        sns.set_context("talk", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.boxplot(data=coherence_df['Coerenza'], orient='h', palette="Blues_d", whis=[0, 0.8])

        mean = np.mean(coherence_df['Coerenza'])
        median = np.median(coherence_df['Coerenza'])
        variance = np.var(coherence_df['Coerenza'])
        std_dev = np.std(coherence_df['Coerenza'])
        percentile_25 = np.percentile(coherence_df['Coerenza'], 25)
        percentile_75 = np.percentile(coherence_df['Coerenza'], 75)

        ax.set_title('Distribuzione dei punteggi del livello di coerenza\n dei topic sui commenti delle interviste',
                     fontsize=18, fontweight='bold')
        ax.set_xlabel('Livello di coerenza da 0 a 1', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.tick_params(axis='x', labelsize=12)

        # Aggiungi la legenda con le informazioni sui percentili e la mediana
        legend_labels = [
            'Xmin: 0',
            f'Q1: {percentile_25}',
            f'Me: {median}',
            f'Q2: {percentile_75}',
            'Xmax: 1'
        ]
        legend_text = '\n'.join(legend_labels)
        ax.legend([legend_text], loc='upper right', frameon=True, fontsize=12, framealpha=0.9)

        plt.tight_layout()
        image_file_path = os.path.join(save_image_path, 'boxplot_LivelloCoerenzaInterviste.png')
        plt.savefig(image_file_path)
        plt.close()

        print(f"Immagine salvata in: {image_file_path}")
        print(f"Media: {mean}")
        print(f"Mediana: {median}")
        print(f"25° percentile: {percentile_25}")
        print(f"75° percentile: {percentile_75}")
        print(f"Varianza: {variance}")
        print(f"Scarto quadratico medio: {std_dev}")


    process_and_plot_coherence_scores()


    print("Fine")