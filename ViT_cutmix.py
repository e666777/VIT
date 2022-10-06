import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt
import scipy
import pickle
import warnings

import tensorflow
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import AUC, Accuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from vit_keras import vit

from mixup_gen import MultiOutputDataGenerator

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


# Set the seed value for experiment reproduci.bility.
seed = 1842
tensorflow.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')


image_generator = MultiOutputDataGenerator(rescale=1/255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=5,
                                           cutmix_alpha=0.5)
train_dataset = image_generator.flow_from_directory(batch_size=256,
                                                    directory='./Alzheimer_s Dataset/train',
                                                    target_size=(224, 224),
                                                    class_mode='categorical')

image_generator_validation = ImageDataGenerator(rescale=1/255)
validation_dataset = image_generator_validation.flow_from_directory(batch_size=256,
                                                                    directory='./Alzheimer_s Dataset/test',
                                                                    shuffle=False,
                                                                    target_size=(224, 224),
                                                                    class_mode='categorical')


vit = vit.vit_b32(
    image_size=224,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    classes=4)
for layer in vit.layers:
    layer.trainable = False
x = Flatten()(vit.output)

x = Dense(512, activation='gelu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='gelu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
prediction = Dense(4, activation='softmax', kernel_regularizer=l2(1e-4))(x)

model = Model(inputs=vit.input, outputs=prediction)
for layer in model.layers[-16:]:
    layer.trainable = True
model.summary()

# optimizer = SGD(learning_rate=0.1, momentum=0.9)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.001)
optimizer = Adam(learning_rate=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)
checkpoint = ModelCheckpoint(filepath='./ViT_cutmix.h5',
                             monitor='val_Accuracy',
                             save_best_only='True')
model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['Accuracy', AUC()])

history = model.fit(train_dataset, epochs=200, validation_data=validation_dataset, callbacks=[reduce_lr, checkpoint])

with open('ViT_cutmix.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)