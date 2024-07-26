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
from scipy.stats import chi2_contingency


class TopicModel:
    """ La classe TopicModel è progettata per analizzare i topic all'interno di file csv che contengono la trascrizione
    audio convertita in testo per frasi. Effettua il preprocessing testuale in lingua italiana utilizzando la libreria
    spacy e gestisce l'estrazione dei topic utilizzando il modello Latent Dirichlet Allocation (LDA) di gensim. I
    risultati verranno salvati in una cartella che conterrà le analisi rappresentative sotto forma di wordclouds e HTML"""

    def __init__(self, file_path, output_folder):
        """Nel costruttore vengono inizializzati gli attributi di istanza della classe che gestiscono la lettura del file
        CSV, l'estrazione del titolo utilizzato per i grafici e l'inizializzazione del modello LDA e dei topic,
        inizialmente impostati a None per essere utilizzati successivamente durante l'analisi. Si inizializza
        l'insieme predefinito di stopwords in lingua italiana da spacy e l'aggiunta di altre stopwords per avere maggiore
        accuratezza nell'analisi. Infine si inizializza la cartella dove verranno importati i risultati dei topic in formato
        di immagini e html."""

        self.df = pd.read_csv(file_path)
        self.csv_title = os.path.splitext(os.path.basename(file_path))[0]
        self.lda_model = None
        self.topics = None
        self.nlp = spacy.load('it_core_news_sm')
        self.stop_words_italian = self.nlp.Defaults.stop_words
        self.additional_stopwords = [
            'che', 'per', 'con', 'come', 'quando', 'sono', 'abbiamo', 'avete', 'hanno', 'sia',
            'siamo', 'siete', 'essere', 'avere', 'fare', 'dire', 'detto', 'fatto', 'del', 'lha',
            'dei', 'della', 'delle', 'nell', 'nella', 'nelle', 'nello', 'negli', 'come',
            'laltro', 'quando', 'alla', 'allo', 'alle', 'agli', 'senza', 'sotto', 'sopra',
            'tipo', 'los', 'lodo', 'boh', 'sai', 'cioè', 'parlarepadre', 'cavolo', 'lo', 'milo',
            'cerare', 'bello', 'mettere', 'dare', 'dici', 'mille', 'sacco', 'vero', 'cosa',
            'lo', 'buttare', 'sentire', 'andare', 'chiedere', 'punto', 'giusto', 'super',
            'perchè', 'ventanni', 'maturo', 'ventenne', 'finché', 'grazia', 'domanda', 'stare',
            'volere', 'potere', 'priorità', 'mese', 'labbiamo', 'dovere', 'possibilità',
            'base', 'giro', 'effetto', 'venire', 'minuto', 'periodo', 'mese', 'settimana', 'dieci',
            'cinque', 'settimana', 'lha', 'laltra', 'problema', 'maniera', 'roba', 'discorso',
            'nome', 'struttura', 'bisogno', 'cacao', 'aprile']
        self.output_folder = output_folder

    def preprocess_texts(self):
        """Metodo che esegue il preprocessing dei testi contenuti nel DataFrame self.df['Frase'] dove ogni frase viene
        trasformata in una lista di token significativi e il risultato viene memorizzato nella nuova colonna ciamata
        Tokens"""

        self.df['Tokens'] = self.df['Frase'].apply(self.tokenize_text)

    def tokenize_text(self, text):
        """Metodo che  esegue la tokenizzazione e il preprocessing utilizzando spacy. Viene eliminata la punteggiatura e
        il testo viene convertito in minuscolo. Viene creato un oggetto doc per il testo tokenizzato e c'è l'iterazione
        su ciascun token applicando determinati criteri: i token non devono essere segni di punteggiatura o numeri,
        devono avere almeno 2 caratteri e non devono essere pronomi, articoli, verbi, preposizioni, avverbi o aggettivi.
        I token validi vengono sottoposti alla lemmatizzazione per ottenere il lemma e verificare che non siano stopwords.
        Infine, i token preprocessati vengono restituiti come risultato del metodo attraverso una lista"""

        text = re.sub(r'[^\w\s]', '', text.lower())
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not token.is_punct and not token.like_num and len(token.text) > 2 and token.pos_ not in ['PRON', 'DET',
                                                                                                        'VERB', 'ADP',
                                                                                                        'ADV', 'ADJ']:
                lemma = token.lemma_
                if lemma not in self.stop_words_italian and lemma not in self.additional_stopwords:
                    tokens.append(lemma)
        return tokens

    def lda(self):
        """Metodo che esegue l'estrazione dei topic utilizzando il modello LDA. Vengono impostati i seed per fare in modo
        che escano sempre gli stessi risultati. I token preprocessati vengono convertiti in un Dictionary e poi in un
        bag-of-words corpus. Successivamente, viene creato un modello LDA con parametri specifici come il numero di
        topic (5) e il numero di iterazioni (3000). Le informazioni sui topic vengono memorizzate in un dizionario e per
        ciascun topic vengono stampate le parole chiave"""

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
        print("Topic per intervista: ")
        print(self.topics)

    def save_visualization(self):
        """Metodo per crea un grafico a barre per i topic estratti dal modello LDA. Viene creato un'immagine png che
        mostra la distribuzione delle parole chiave per ciascun topic, ordinato per probabilità. Il grafico viene
        salvato nella cartella dei risultati dell'analisi sui topic"""

        name_html = os.path.join(self.output_folder, f'{self.csv_title}_Visualizzazione_LDA.html')
        bow_corpus = [self.lda_model.id2word.doc2bow(t) for t in self.df['Tokens']]
        lda_html = pyLDAvis.gensim_models.prepare(self.lda_model, bow_corpus, self.lda_model.id2word)
        pyLDAvis.save_html(lda_html, name_html)

    def topic_plot(self):
        """Metodo che crea le word cloud relative ai topic estratti dal modello LDA. Per ogni topic viene creata
        un'immagine basata sulla probabilità delle parole chiave (più è grande la parola più quella parola ha
        probabilità maggiore). Le immagini vengono salvate in un file png all'interno della cartella dei risultati
        dell'analisi sui topic"""

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
        plt.suptitle(f'Modello di Topic - Frasi da {self.csv_title}', fontsize=50)
        plt.savefig(name_plot)
        plt.show()
        print("Il topic plot è disponibile!")

    def word_cloud(self):
        """Metodo che calcola la coerenza dei topic estratti usando il modello LDA attraverso un punteggio che va
        da 0 a 1. L'interpretazione della coerenza risulta soggettiva e dipende dal contesto specifico in cui viene
        effettuata l'analisi di topic"""

        name_cloud = os.path.join(self.output_folder, f'{self.csv_title}_Word_Cloud.png')
        plt.figure(figsize=(30, 30))
        for i in range(5):
            df = pd.DataFrame(self.lda_model.show_topic(i), columns=['Token', 'Probabilità']).set_index('Token')
            df = df.sort_values('Probabilità', ascending=False)
            wordcloud = WordCloud(width=800,
                                  height=300,
                                  background_color='white').generate_from_frequencies(dict(df['Probabilità']))
            plt.subplot(5, 2, i + 1)
            plt.title('Topic ' + str(i))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
        plt.suptitle('Word Clouds - Topics', fontsize=50)
        plt.savefig(name_cloud)
        plt.show()
        print("Le word cloud per ogni topic sono disponibili!")

    def calculate_coherence(self):
        """Metodo che calcola il livello di coerenza complessivo dei topic generati dal modello LDA. La coerenza di un
         topic è una misura di quanto le parole che compongono il topic siano semanticamente simili tra loro. Un
         punteggio di coerenza più alto indica che le parole all'interno di un topic sono più semanticamente correlate."""

        tokens = self.df['Tokens'].tolist()
        dict = corpora.Dictionary(tokens)
        topics = [[token for token, weight in self.lda_model.show_topic(topic_id)] for topic_id in
                  range(self.lda_model.num_topics)]
        coherence_model = CoherenceModel(topics=topics, texts=tokens, dictionary=dict, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        return coherence_score

    def compare_dominant_topics(self, other_model):
        """Metodo che serve per eseguire l'analisi comparativa dei topic dominanti tra due istanze di TopicModel.
        Utilizza il test del Chi-quadrato per valutare se c'è una relazione tra i testi delle interviste e i topic
        dominanti."""

        # le liste di token vengono convertite in bag-of-words
        bow_corpus_1 = [self.lda_model.id2word.doc2bow(doc) for doc in self.df['Tokens']]
        bow_corpus_2 = [other_model.lda_model.id2word.doc2bow(doc) for doc in other_model.df['Tokens']]

        # si ottengono i topic dominanti per ogni documento
        doc_lda_1 = [self.lda_model.get_document_topics(bow) for bow in bow_corpus_1]
        doc_lda_2 = [other_model.lda_model.get_document_topics(bow) for bow in bow_corpus_2]
        dominant_topics_1 = [max(topic, key=lambda x: x[1])[0] for topic in doc_lda_1]
        dominant_topics_2 = [max(topic, key=lambda x: x[1])[0] for topic in doc_lda_2]

        # viene creato un DataFrame con i testi e i topic dominanti
        data = {'text': ['doc1'] * len(dominant_topics_1) + ['doc2'] * len(dominant_topics_2),
                'dominant_topic': dominant_topics_1 + dominant_topics_2}
        df = pd.DataFrame(data)

        # viene applicato il test del Chi-quadrato
        contingency_table = pd.crosstab(df['text'], df['dominant_topic'])
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)

        print("\nAnalisi comparativa dei topic dominanti:")
        print("Statistica del test del Chi-quadrato:", chi2)
        print("P-value:", p_val)

        # interpretazione dei risultati
        if p_val < 0.05:
            print("C'è una relazione significativa tra i testi e i topic dominanti.")
        else:
            print("Non c'è una relazione significativa tra i testi e i topic dominanti.")

    def run_topic_analysis(self):
        """Metodo che richiama gli altri metodi, stampa il livello di coerenza dei topic per ogni intervista e crea un
        file CSV in cui a ogni frase viene accostato il topic con la probabilità più alta"""

        self.preprocess_texts()
        self.lda()
        self.save_visualization()
        self.topic_plot()
        self.word_cloud()

        # viene mandato a schermo il livello di coerenza di ciascun testo
        coherence_score = self.calculate_coherence()
        print(f"Coerenza dei Topic: {coherence_score}")


if __name__ == "__main__":

    # percorso dei file csv da cui verrà estratta, tramite pandas, la colonna "Frase" per l'analisi dei topic
    file_path_1 = r'C:\Users\39392\Desktop\progettoWebAnalytics\file_csv\StevenBasalari_model_large.csv'
    file_path_2 = r'C:\Users\39392\Desktop\progettoWebAnalytics\file_csv\BiancaBalti_model_large.csv'

    # cartella di riferimento in cui verranno salvate le immagini png e html (creata se non esiste)
    output_folder = r'C:\Users\39392\Desktop\progettoWebAnalytics\risultati_analisi_lda'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Attraverso un messaggio si fa riferimento al file csv che si sta analizzando
    # Creazione istanza per il file specificato e la cartella di riferimento per i risultati dell'analisi dei topic
    # Chiamata del metodo run_topic_analysis() che permette attraverso tutta una serie di chiamate di altri metodi
    # della classe lo sviluppo del codice
    print("Analisi Topic per l'intervista a Steven Basalari:")
    topic_model_1 = TopicModel(file_path_1, output_folder)
    topic_model_1.run_topic_analysis()

    print("\nAnalisi Topic per l'intervista a Bianca Balti:")
    topic_model_2 = TopicModel(file_path_2, output_folder)
    topic_model_2.run_topic_analysis()

    # Chiamata del metodo compare_dominant_topics dell'istanza topic_model_1 della classe TopicModel: passa come
    # argomento l'istanza topic_model_2. Il metodo esegue un'analisi comparativa dei topic dominanti tra i due testi
    # rappresentati dalle due istanze di TopicModel, utilizzando il test del Chi-quadrato.
    topic_model_1.compare_dominant_topics(topic_model_2)