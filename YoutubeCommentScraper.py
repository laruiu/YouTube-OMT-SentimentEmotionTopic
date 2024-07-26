import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import os
import csv


class ScraperComment:
    """Classe creata per estrarre i commenti di video su YouTube. Permette di interagire con l'API di YouTube e
    recuperare i commenti dei video. Definisce tre costanti: api_service_name, api_version e DEVELOPER_KEY. La prima è
    impostata su 'youtube' e indica il servizio con cui interagire, la seconda indica la versione API di YouTube e
    l'ultima indica la chiave di API di sviluppatore valida, necessaria per autenticare le richieste all'API di
    YouTube"""

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyCO6QCasjpaK7FOe3lPpFU-zEMr5t1n2Ew"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    def __init__(self, video_id, ospite):
        """Metodo costruttore che richiama per ogni oggetto della classe l'attributi ID del video di Youtube e nome
        dell'ospite dell'intervista"""

        self.video_id = video_id
        self.ospite = ospite

    def get_comments_with_metadata(self):
        """Questo metodo recupera i commenti di un video YouTube insieme a metadati importanti, come il nome utente del
        commento, il testo di esso e la sua data di pubblicazione. Sfrutta la YouTube Data API per ottenere i dati
        richiesti in lotti (pagine) fino a esaurimento dei commenti disponibili"""

        comments_data = []
        request = self.youtube.commentThreads().list(
            part="snippet",
            videoId=self.video_id,
            maxResults=100
        )

        while request:
            response = request.execute()
            for item in response['items']:
                username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comment_date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                comments_data.append({'Username': username, 'Commento': comment_text, 'Data': comment_date})
            request = self.youtube.commentThreads().list_next(request, response)

        return comments_data

    def save_comments_to_csv(self):
        """Metodo che salva i commenti dei video su YouTube e tutte le informazioni annesse mediante un DataFrame Pandas,
        che successivamente viene trasformato in un file CSV e salvato nella cartella file_commenti_csv."""

        comments_data = self.get_comments_with_metadata()
        comments_df = pd.DataFrame(comments_data)

        folder_name = "file_commenti_csv"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filename = f"{folder_name}/{self.ospite}.csv"
        comments_df.to_csv(filename, index=False)

        return (f"Il DataFrame dell'ospite {self.ospite} è stato salvato come {filename} "
                f"nella cartella file_commenti_csv\n")


# Creazione di istanze della classe ScraperComment in cui, per ogni video di cui si vogliono scaricare i metadati dei
# commenti, si passano come attributi gli ID dei video YouTube per identificarli e i nomi degli ospiti. Alla fine, si
# chiama il metodo che permette di salvare il DataFrame creato nella classe come file CSV.
scraper1 = ScraperComment(video_id="NUOe8EgEj-0", ospite="AlexCotoia").save_comments_to_csv()
scraper2 = ScraperComment(video_id="-szezlsJvYQ", ospite="BiancaBalti").save_comments_to_csv()
scraper3 = ScraperComment(video_id="nYYWoKzJNvE", ospite="BenjaminMascolo").save_comments_to_csv()
scraper4 = ScraperComment(video_id="NXcaaQ4u00k", ospite="MicheleMorrone").save_comments_to_csv()
scraper5 = ScraperComment(video_id="zXSLNzQELKs", ospite="SusiGallesi").save_comments_to_csv()
scraper6 = ScraperComment(video_id="RWcXxprOOcM", ospite="MarcoBaldini").save_comments_to_csv()
scraper7 = ScraperComment(video_id="Uq96mBBzkbU", ospite="SaraTommasi").save_comments_to_csv()
scraper8 = ScraperComment(video_id="S3Yy0ZqGfoA", ospite="NicolettaAmato").save_comments_to_csv()
scraper9 = ScraperComment(video_id="P2cqM9-t7u8", ospite="MarioMaccione").save_comments_to_csv()
scraper10 = ScraperComment(video_id="QZiC-Z5nEOw", ospite="AndreaPresti").save_comments_to_csv()

if __name__ == "__main__":

    # chiamata degli oggetti della classe
    print(scraper1, scraper2, scraper3, scraper4, scraper5,
          scraper6, scraper7, scraper8, scraper9, scraper10)

    def remove_user_from_files(folder_path):
        """Dopo aver creato la classe ScraperComment, ho creato una funzione che prende la cartella contente i file csv
        e ho tolto tutti i commenti pubblicati dalla pagina ufficiale in quanto non servono ai fini del progetto, infine
        ho aggiornato i file"""

        user_to_remove = "@OneMoreTimePodcast"  # utente pagina ufficiale
        for file_name in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    reader = csv.reader(file)
                    content = list(reader)

                header = content[0]
                new_content = [header]
                for row in content[1:]:
                    if user_to_remove not in row[0]:
                        new_content.append(row)

                new_file_path = os.path.join(folder_path, file_name)
                with open(new_file_path, "w", encoding="utf-8", newline='') as new_file:
                    writer = csv.writer(new_file)
                    writer.writerows(new_content)

                print(f"Il contenuto del file {file_name} è stato modificato e salvato come {new_file_path}")


    remove_user_from_files(folder_path=r"C:\Users\39392\Desktop\progettoWebAnalytics\file_commenti_csv")
