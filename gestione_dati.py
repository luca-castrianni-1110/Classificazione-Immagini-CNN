import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import configurazione as conf

def crea_divisioni_stratificate(cartella_grezza, cartella_destinazione, quota_add, quota_val, quota_test=0.0):
    if os.path.exists(cartella_destinazione):
        print(f"Cartella di split {cartella_destinazione} già esistente. Divisione saltata.")
        return

    print(f"Creazione split stratificati in {cartella_destinazione}...")
    os.makedirs(cartella_destinazione, exist_ok=True)
    
    classi = sorted([d for d in os.listdir(cartella_grezza) if os.path.isdir(os.path.join(cartella_grezza, d))])
    
    for classe in classi:
        os.makedirs(os.path.join(cartella_destinazione, 'addestramento', classe), exist_ok=True)
        os.makedirs(os.path.join(cartella_destinazione, 'validazione', classe), exist_ok=True)
        if quota_test > 0:
            os.makedirs(os.path.join(cartella_destinazione, 'test', classe), exist_ok=True)

        cartella_classe = os.path.join(cartella_grezza, classe)
        immagini = [f for f in os.listdir(cartella_classe) if f.endswith(('.jpg', '.tif', '.png'))]
        
        # Suddivisione stratificata fissa
        imm_addestramento, imm_temp = train_test_split(immagini, train_size=quota_add, random_state=conf.SEME, shuffle=True)
        
        if quota_test > 0:
            quota_val_relativa = quota_val / (quota_val + quota_test)
            imm_validazione, imm_test = train_test_split(imm_temp, train_size=quota_val_relativa, random_state=conf.SEME, shuffle=True)
        else:
            imm_validazione = imm_temp
            imm_test = []

        for imm in imm_addestramento:
            shutil.copy(os.path.join(cartella_classe, imm), os.path.join(cartella_destinazione, 'addestramento', classe, imm))
        for imm in imm_validazione:
            shutil.copy(os.path.join(cartella_classe, imm), os.path.join(cartella_destinazione, 'validazione', classe, imm))
        for imm in imm_test:
            shutil.copy(os.path.join(cartella_classe, imm), os.path.join(cartella_destinazione, 'test', classe, imm))

    print(f"Split completato con successo in {cartella_destinazione}")

def preelabora_immagine_opencv(percorso_tensore, etichetta):
    def elaborazione_interna(percorso):
        percorso_str = percorso.numpy().decode("utf-8")
        
        # Preprocessing nativo in OpenCV
        immagine = cv2.imread(percorso_str)
        if immagine is None:
            raise ValueError(f"Immagine non trovata o corrotta: {percorso_str}")
            
        immagine = cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)
        immagine = cv2.resize(immagine, (conf.LARGHEZZA_IMG, conf.ALTEZZA_IMG))
        
        immagine = immagine.astype(np.float32)
        immagine -= conf.MEDIA_IMAGENET 
        
        return immagine

    [immagine_elaborata] = tf.py_function(elaborazione_interna, [percorso_tensore], [tf.float32])
    immagine_elaborata.set_shape([conf.ALTEZZA_IMG, conf.LARGHEZZA_IMG, 3])
    
    return immagine_elaborata, etichetta

def ottieni_dataset(cartella_divisione, sottoinsieme, num_classi):
    cartella = os.path.join(cartella_divisione, sottoinsieme)
    classi = sorted([d for d in os.listdir(cartella) if os.path.isdir(os.path.join(cartella, d))])
    indici_classi = {nome: i for i, nome in enumerate(classi)}
    
    percorsi_file = []
    etichette = []
    
    for nome_classe in classi:
        cartella_classe = os.path.join(cartella, nome_classe)
        for nome_file in os.listdir(cartella_classe):
            if nome_file.endswith(('.jpg', '.tif', '.png')):
                percorsi_file.append(os.path.join(cartella_classe, nome_file))
                
                etichetta_codificata = np.zeros(num_classi, dtype=np.float32)
                etichetta_codificata[indici_classi[nome_classe]] = 1.0
                etichette.append(etichetta_codificata)
                
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(percorsi_file), tf.constant(etichette)))
    
    if sottoinsieme == 'addestramento':
        dataset = dataset.shuffle(buffer_size=len(percorsi_file), seed=conf.SEME)
        
    dataset = dataset.map(preelabora_immagine_opencv, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(conf.DIMENSIONE_LOTTO)
    
    if sottoinsieme == 'addestramento':
        livello_aumento = ottieni_aumento_dati()
        # Data augmentation applicata SOLO sul training
        dataset = dataset.map(lambda x, y: (livello_aumento(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def ottieni_aumento_dati():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=conf.SEME),
        tf.keras.layers.RandomRotation(0.3, seed=conf.SEME),
        tf.keras.layers.RandomZoom(0.2, seed=conf.SEME),
    ])

def prepara_tutti_i_dataset():
    crea_divisioni_stratificate(conf.CARTELLA_GREZZA_AID, conf.CARTELLA_DIV_AID, conf.RAPPORTO_ADD_AID, conf.RAPPORTO_VAL_AID)
    crea_divisioni_stratificate(conf.CARTELLA_GREZZA_UCM, conf.CARTELLA_DIV_UCM, conf.RAPPORTO_ADD_UCM, conf.RAPPORTO_VAL_UCM, conf.RAPPORTO_TEST_UCM)

    dati_add_aid = ottieni_dataset(conf.CARTELLA_DIV_AID, 'addestramento', conf.CLASSI_AID)
    dati_val_aid = ottieni_dataset(conf.CARTELLA_DIV_AID, 'validazione', conf.CLASSI_AID)

    dati_add_ucm = ottieni_dataset(conf.CARTELLA_DIV_UCM, 'addestramento', conf.CLASSI_UCM)
    dati_val_ucm = ottieni_dataset(conf.CARTELLA_DIV_UCM, 'validazione', conf.CLASSI_UCM)
    dati_test_ucm = ottieni_dataset(conf.CARTELLA_DIV_UCM, 'test', conf.CLASSI_UCM)

    return dati_add_aid, dati_val_aid, dati_add_ucm, dati_val_ucm, dati_test_ucm