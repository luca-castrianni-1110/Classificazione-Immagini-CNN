'''
COSA FA:
Questo script inizializza ed esegue il riepilogo testuale (summary) della struttura dei modelli legati alle due strategie di addestramento: il modello creato da zero per il dataset UC Merced (Strategia 2) e il modello pre-addestrato su AID e adattato per il fine-tuning parziale (Strategia 1).

A COSA SERVE:
Serve a scopi di ispezione, debug e verifica immediata delle reti neurali prima di avviare il vero e proprio ciclo di addestramento complessivo. Permette di visualizzare il numero totale di parametri (sia quelli addestrabili che quelli bloccati), la forma dell'output di ogni livello e l'organizzazione dei blocchi.

PERCHÉ LO FACCIAMO:
Lo facciamo per verificare che l'architettura custom, i blocchi residuali e i meccanismi di skip-connection siano stati assemblati correttamente dal punto di vista dimensionale, e per confrontare l'impatto e la complessità computazionale delle due diverse strategie basandosi sul numero di parametri attivi.
'''

import architettura
import addestramento
import configurazione as conf

print("\n" + "="*50)
print("1. PARAMETRI STRATEGIA 2 (DA ZERO)")
print("="*50)
modello_da_zero = architettura.crea_modello_ucm_da_zero()
modello_da_zero.summary()

print("\n" + "="*50)
print("2. PARAMETRI STRATEGIA 1 (FINE-TUNING PARZIALE)")
print("="*50)
modello_aid_base = architettura.crea_modello_aid()
modello_ft = addestramento.prepara_modello_per_fine_tuning(modello_aid_base, conf.CLASSI_UCM)
modello_ft.summary()