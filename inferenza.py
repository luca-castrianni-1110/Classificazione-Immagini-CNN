import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import configurazione as conf
import architettura

def predici_immagine(percorso_immagine, modello, classi_ucm):
    immagine = cv2.imread(percorso_immagine)
    if immagine is None:
        print(f"Errore: Impossibile caricare l'immagine: {percorso_immagine}")
        return
        
    # Pre-processing OpenCV speculare a quello del training
    immagine = cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)
    immagine = cv2.resize(immagine, (conf.LARGHEZZA_IMG, conf.ALTEZZA_IMG))
    immagine = immagine.astype(np.float32)
    immagine -= conf.MEDIA_IMAGENET
    
    immagine_lotto = np.expand_dims(immagine, axis=0)

    probabilita = modello.predict(immagine_lotto, verbose=0)[0]
    indice_predetto = np.argmax(probabilita)
    classe_predetta = classi_ucm[indice_predetto]
    sicurezza = probabilita[indice_predetto] * 100

    print(f"\nFile analizzato: {os.path.basename(percorso_immagine)}")
    print(f"\nClasse Predetta: {classe_predetta.upper()} ({sicurezza:.2f}%)")

    
    indici_top3 = probabilita.argsort()[-3:][::-1]
    for i in indici_top3:
        print(f"   - {classi_ucm[i]}: {probabilita[i]*100:.2f}%")

def avvia_inferenza(percorso_input, percorso_modello="modello_finale.keras"):
    try:
        # Caricamento del modello gestendo la classe custom BloccoResiduale
        modello = tf.keras.models.load_model(
            percorso_modello,
            custom_objects={'BloccoResiduale': architettura.BloccoResiduale}
        )
        print(f"Modello caricato perfettamente da: {percorso_modello}")
        
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return

    classi_ucm = [
        'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 
        'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 
        'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 
        'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 
        'storagetanks', 'tenniscourt'
    ]

    print(" INIZIO INFERENZA MODELLO ")

    # Logica ibrida: gestisce in input sia singoli file che intere directory
    if os.path.isfile(percorso_input):
        predici_immagine(percorso_input, modello, classi_ucm)
        
    elif os.path.isdir(percorso_input):
        print(f"Trovata cartella. Analizzo tutte le immagini in: {percorso_input}")
        immagini_trovate = [f for f in os.listdir(percorso_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        if not immagini_trovate:
            print("Nessuna immagine trovata nella cartella.")
            return
            
        for img_nome in immagini_trovate:
            path_completo = os.path.join(percorso_input, img_nome)
            predici_immagine(path_completo, modello, classi_ucm)
            
    else:
        print(f"Errore: Il percorso '{percorso_input}' non esiste.")

if __name__ == "__main__":
    cartella_predefinita = "./immagini_per_inferenza"

    if len(sys.argv) > 1:
        avvia_inferenza(sys.argv[1])
    else:
        print(f"Controllo la cartella: {cartella_predefinita}")
        avvia_inferenza(cartella_predefinita)