# Classificazione-Immagini-CNN
Sviluppo e confronto di architetture CNN in Keras (da zero e Transfer Learning) per la classificazione di immagini aeree.


## Guida all'Utilizzo

**Configurazione Iniziale:** 
Prima di procedere, assicurarsi di estrarre e posizionare tutti i file scaricati all'interno di un'unica cartella principale.

### 1. Inferenza (Test del modello pre-addestrato)
Per testare il modello sulle tue immagini senza doverlo riaddestrare:
1. Inserire le immagini da classificare all'interno della cartella `Immagini_per_inferenza`.
2. Avviare lo script di inferenza.
3. I risultati della classificazione verranno stampati direttamente nel terminale.

*Nota sulle performance:* L'inizializzazione del modello e il caricamento dei pesi potrebbero richiedere da pochi secondi fino a un minuto prima di mostrare i risultati, a seconda dell'architettura hardware del proprio PC.

### 2. Addestramento da zero (Training completo)
Per replicare l'intero processo di addestramento:
1.  **Importante (Prerequisito):** Inserire nella directory principale le due cartelle contenenti i dataset. Queste devono mantenere rigorosamente i **nomi originali**. Il mancato rispetto di questo passaggio genererà un errore di percorso (Path Error) durante l'esecuzione.
2. Avviare lo script `Main`.
3. Il processo gestirà automaticamente la creazione di tutte le sottocartelle necessarie, il salvataggio dei file dei pesi e la generazione dei grafici delle performance (Loss/Accuracy).
