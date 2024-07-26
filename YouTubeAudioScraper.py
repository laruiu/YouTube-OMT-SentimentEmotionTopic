from pytube import YouTube
import os


class DownloadAudioYouTube:
    """Creazione classe DownloadAudioYouTube usata per effettuare
     i download di audio da YouTube."""

    def __init__(self, URL):
        """Metodo costruttore richiamato ogni volta che viene creato un nuovo oggetto della classe,
        inizializza l'attributo URL con il link del video di YouTube da cui verrà scaricato l'audio"""

        self.URL = URL

    def download(self):
        """Metodo che effettua il download dal video su YouTube tramite l'URL fornito.
        Se il download dell'audio viene effettuato lo si trova nel percorso scritto nel codice (le cartelle della
        directory se non sono presenti vengono create automaticamente).
        Il file scaricato prende il nome del titolo del video su YouTube"""

        try:
            yt = YouTube(self.URL)
            audio = yt.streams.filter(only_audio=True).first()
            path = "C:/Users/39392/Desktop/progettoWebAnalytics/audio_mp3"
            out_file = audio.download(output_path=path)
            base, ext = os.path.splitext(out_file)
            new_file = base + '.mp3'
            try:
                os.rename(out_file, new_file)
                return "Il video su Youtube '" + yt.title + "' è stato scaricato correttamente in formato .mp3"
            except FileExistsError:
                return f"Attenzione! Il file {yt.title} esiste già nel path selezionato: sono presenti più file uguali!"
        except Exception as e:
            return "Si è verificato un errore durante il download: " + str(e)


# Instanzia oggetti per la classe DownloadAudioYouTube e chiama il metodo download,
# vengono utilizzati per l'attributo URL due link diversi
video1 = DownloadAudioYouTube(URL="https://www.youtube.com/watch?v=SO7Comn4zIk").download()
video2 = DownloadAudioYouTube(URL="https://www.youtube.com/watch?v=-szezlsJvYQ&t").download()

if __name__ == "__main__":

    # Chiamo gli oggetti della classe DownloadAudioYouTube per ogni video da scaricare
    print("Video su Steven Basalari:")
    print(video1)
    print("Video su Bianca Balti:")
    print(video2)
