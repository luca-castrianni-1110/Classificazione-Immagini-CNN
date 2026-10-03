import tensorflow as tf
from keras import layers, models, regularizers
import configurazione as conf

class BloccoResiduale(layers.Layer):
    """Blocco Residuale Custom con skip connection"""
    def __init__(self, filtri, passi=1, usa_conv_1x1=False, **kwargs):
        super(BloccoResiduale, self).__init__(**kwargs)
        
        self.conv1 = layers.Conv2D(filtri, kernel_size=3, padding='same', strides=passi, 
                                use_bias=False, kernel_regularizer=regularizers.l2(1e-3))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        
        self.conv2 = layers.Conv2D(filtri, kernel_size=3, padding='same', 
                                use_bias=False, kernel_regularizer=regularizers.l2(1e-3))
        self.bn2 = layers.BatchNormalization()
        
        # Convoluzione 1x1 per allineamento dimensionale della skip connection
        if usa_conv_1x1:
            self.conv_salto = layers.Conv2D(filtri, kernel_size=1, strides=passi, 
                                            use_bias=False, kernel_regularizer=regularizers.l2(1e-3))
            self.bn_salto = layers.BatchNormalization()
        else:
            self.conv_salto = None

    def call(self, ingressi, training=False):
        x = self.conv1(ingressi)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.conv_salto is not None:
            salto = self.conv_salto(ingressi)
            salto = self.bn_salto(salto, training=training)
        else:
            salto = ingressi
            
        x = layers.Add()([x, salto])
        return self.relu(x)

def costruisci_resnet_personalizzata(forma_ingresso, num_classi, nome_modello="ResNet_Personalizzata"):
    ingressi = layers.Input(shape=forma_ingresso)

    x = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(ingressi)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = BloccoResiduale(64, passi=2, usa_conv_1x1=True)(x)
    x = BloccoResiduale(128, passi=2, usa_conv_1x1=True)(x)
    
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Regolarizzazione per spegnere canali convoluzionali interi
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Flatten()(x)
    
    x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.65)(x)
    
    uscite = layers.Dense(num_classi, activation='softmax')(x)

    return models.Model(inputs=ingressi, outputs=uscite, name=nome_modello)

def crea_modello_aid():
    return costruisci_resnet_personalizzata(
        forma_ingresso=(conf.ALTEZZA_IMG, conf.LARGHEZZA_IMG, 3), 
        num_classi=conf.CLASSI_AID, 
        nome_modello="Modello_Preaddestramento_AID"
    )

def crea_modello_ucm_da_zero():
    return costruisci_resnet_personalizzata(
        forma_ingresso=(conf.ALTEZZA_IMG, conf.LARGHEZZA_IMG, 3), 
        num_classi=conf.CLASSI_UCM, 
        nome_modello="Modello_Da_Zero_UCM"
    )