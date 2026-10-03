import tensorflow as tf
import configurazione as conf
import gestione_dati
import architettura
import addestramento
import os

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.config.threading.set_intra_op_parallelism_threads(8) 
tf.config.threading.set_inter_op_parallelism_threads(8)

def esegui_esperimento():
    print("Preparazione dei Dataset")
    dati_add_aid, dati_val_aid, dati_add_ucm, dati_val_ucm, dati_test_ucm = gestione_dati.prepara_tutti_i_dataset()

    cartella_add_aid = os.path.join(conf.CARTELLA_DIV_AID, 'addestramento')
    cartella_add_ucm = os.path.join(conf.CARTELLA_DIV_UCM, 'addestramento')
    pesi_aid = addestramento.calcola_pesi_classi(cartella_add_aid, conf.CLASSI_AID)
    pesi_ucm = addestramento.calcola_pesi_classi(cartella_add_ucm, conf.CLASSI_UCM)

    print("\n2. STRATEGIA 1: Pre-Addestramento su AID e Fine-Tuning su UC Merced")
    modello_aid = architettura.crea_modello_aid()
    storico_aid = addestramento.compila_e_addestra(
        modello_aid, dati_add_aid, dati_val_aid, tasso_apprendimento=1e-3, epoche=30, 
        titolo="Pre_Addestramento_AID", pesi_classi=pesi_aid
    )
    addestramento.disegna_grafici_storico(storico_aid, "Pre_Addestramento_AID")

    modello_fine_tuning = addestramento.prepara_modello_per_fine_tuning(modello_aid, conf.CLASSI_UCM)
    storico_ft = addestramento.compila_e_addestra(
        modello_fine_tuning, dati_add_ucm, dati_val_ucm, tasso_apprendimento=1e-4, epoche=40, 
        titolo="Fine_Tuning_UCM", pesi_classi=pesi_ucm
    )
    addestramento.disegna_grafici_storico(storico_ft, "Fine_Tuning_UCM")

    print("\n3. STRATEGIA 2: Addestramento da zero su UC Merced")
    modello_da_zero = architettura.crea_modello_ucm_da_zero()
    storico_zero = addestramento.compila_e_addestra(
        modello_da_zero, dati_add_ucm, dati_val_ucm, tasso_apprendimento=1e-3, epoche=40, 
        titolo="Da_Zero_UCM", pesi_classi=pesi_ucm
    )
    addestramento.disegna_grafici_storico(storico_zero, "Da_Zero_UCM")

    print("\n4. MODEL SELECTION: Confronto sul Validation Set")
    f1_strat_1 = addestramento.calcola_f1_validazione(modello_fine_tuning, dati_val_ucm)
    f1_strat_2 = addestramento.calcola_f1_validazione(modello_da_zero, dati_val_ucm)
    
    print(f"F1-Score (Validazione) Strategia 1 (Fine-Tuning): {f1_strat_1:.4f}")
    print(f"F1-Score (Validazione) Strategia 2 (Da Zero):      {f1_strat_2:.4f}")
    
    if f1_strat_1 >= f1_strat_2:
        print("\nIl modello vincitore è la STRATEGIA 1 (Fine-Tuning)")
        modello_migliore = modello_fine_tuning
    else:
        print("\nIl modello vincitore è la STRATEGIA 2 (Da Zero)")
        modello_migliore = modello_da_zero

    print("\n5. VALUTAZIONE FINALE: Test del Modello Vincitore")
    cartella_test = os.path.join(conf.CARTELLA_DIV_UCM, 'test')
    nomi_classi_ucm = sorted([d for d in os.listdir(cartella_test) if os.path.isdir(os.path.join(cartella_test, d))])
    
    addestramento.valuta_sul_test(modello_migliore, dati_test_ucm, nomi_classi_ucm)
    
    percorso_salvataggio = os.path.join(conf.CARTELLA_PROGETTO, "modello_finale.keras")
    modello_migliore.save(percorso_salvataggio)
    print(f"\nFiito, il miglior modello è stato salvato in:\n {percorso_salvataggio}")

if __name__ == "__main__":
    esegui_esperimento()