import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.keras import TqdmCallback

def ottieni_controlli_addestramento(titolo):
    arresto_anticipato = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True, 
        verbose=1
    )
    
    riduci_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        min_lr=1e-6, 
        verbose=1
    )
    
    return [arresto_anticipato, riduci_lr]

def calcola_pesi_classi(cartella_addestramento, num_classi):
    """Calcolo dynamically dei pesi per compensare sbilanciamenti tra classi"""
    print("Calcolo dei pesi per il bilanciamento delle classi...")
    classi = sorted([d for d in os.listdir(cartella_addestramento) if os.path.isdir(os.path.join(cartella_addestramento, d))])
    
    etichette_vere = []
    for indice_classe, nome_classe in enumerate(classi):
        cartella_classe = os.path.join(cartella_addestramento, nome_classe)
        num_immagini = len([f for f in os.listdir(cartella_classe) if f.endswith(('.jpg', '.tif', '.png'))])
        etichette_vere.extend([indice_classe] * num_immagini)
        
    pesi = compute_class_weight('balanced', classes=np.arange(num_classi), y=etichette_vere)
    dizionario_pesi = {i: peso for i, peso in enumerate(pesi)}
    return dizionario_pesi

def compila_e_addestra(modello, dati_addestramento, dati_validazione, tasso_apprendimento, epoche, titolo, pesi_classi=None):
    print(f"\nInizio Addestramento: {titolo} ---")
    modello.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tasso_apprendimento),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    controlli = ottieni_controlli_addestramento(titolo)
    controlli.append(TqdmCallback(verbose=1))
    
    storico = modello.fit(
        dati_addestramento, 
        validation_data=dati_validazione,
        epochs=epoche, 
        callbacks=controlli,
        class_weight=pesi_classi,
        verbose=0 
    )
    return storico

def disegna_grafici_storico(storico, titolo):
    acc = storico.history['accuracy']
    val_acc = storico.history['val_accuracy']
    loss = storico.history['loss']
    val_loss = storico.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Accuratezza Addestramento')
    plt.plot(val_acc, label='Accuratezza Validazione')
    plt.legend(loc='lower right')
    plt.title(f'{titolo} - Accuratezza')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Perdita Addestramento')
    plt.plot(val_loss, label='Perdita Validazione')
    plt.legend(loc='upper right')
    plt.title(f'{titolo} - Perdita (Loss)')
    
    nome_file = f"{titolo.replace(' ', '_')}_grafico.png"
    plt.tight_layout()
    plt.savefig(nome_file)
    plt.close()

def prepara_modello_per_fine_tuning(modello_preaddestrato, num_classi_nuove):
    # Rimuove il layer finale a 30 classi e ne istanzia uno a 21
    strato_penultimo = modello_preaddestrato.layers[-2].output
    nuove_uscite = tf.keras.layers.Dense(num_classi_nuove, activation='softmax', name='classificatore_ucm')(strato_penultimo)
    
    modello_affinamento = tf.keras.models.Model(inputs=modello_preaddestrato.input, outputs=nuove_uscite, name="Modello_Affinato_UCM")
    
    # Congelamento del 60% della rete base
    limite_congelamento = int(len(modello_affinamento.layers) * 0.6)
    
    for strato in modello_affinamento.layers[:limite_congelamento]:
        strato.trainable = False
        
    for strato in modello_affinamento.layers[limite_congelamento:]:
        # Batch Normalization mantenuta congelata per preservare le medie apprese
        if isinstance(strato, tf.keras.layers.BatchNormalization):
            strato.trainable = False 
        else:
            strato.trainable = True
            
    return modello_affinamento

def calcola_f1_validazione(modello, dataset_validazione):
    etichette_vere = []
    probabilita_predette = []
    
    for immagini, etichette in dataset_validazione:
        etichette_vere.extend(np.argmax(etichette.numpy(), axis=1))
        predizioni = modello.predict(immagini, verbose=0)
        probabilita_predette.extend(predizioni)
        
    etichette_predette = np.argmax(probabilita_predette, axis=1)
    
    return f1_score(etichette_vere, etichette_predette, average='macro')

def valuta_sul_test(modello, dataset_test, nomi_classi):
    print("\n Valutazione sul Test Set")
    
    etichette_vere = []
    probabilita_predette = []
    
    for immagini, etichette in dataset_test:
        etichette_vere.extend(np.argmax(etichette.numpy(), axis=1))
        predizioni = modello.predict(immagini, verbose=0)
        probabilita_predette.extend(predizioni)
        
    etichette_predette = np.argmax(probabilita_predette, axis=1)
    
    report = classification_report(etichette_vere, etichette_predette, target_names=nomi_classi)
    
    print("\nReport di Classificazione:\n")
    print(report)
    
    with open("report_valutazione_finale.txt", "w", encoding="utf-8") as f:
        f.write("===================================================\n")
        f.write("5. VALUTAZIONE FINALE: Test del Modello Vincitore\n")
        f.write("===================================================\n\n")
        f.write("--- Valutazione sul Test Set ---\n\n")
        f.write(report)
        
    print("\nIl report finale è stato salvato in: report_valutazione_finale.txt")
    
    matrice = confusion_matrix(etichette_vere, etichette_predette)
    plt.figure(figsize=(16, 14))
    sns.heatmap(matrice, annot=True, fmt='d', cmap='Blues', xticklabels=nomi_classi, yticklabels=nomi_classi)
    plt.title("Matrice di Confusione")
    plt.ylabel('Etichetta Vera')
    plt.xlabel('Etichetta Predetta')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("matrice_di_confusione.png")
    plt.close()