#!pip install --upgrade git+https://github.com/davin11/easy-cv-dataset keras-cv

#!wget -q -c https://download843.mediafire.com/us7lt2um4yog312RSfvjxVJqxmcg_CJqBjUDbBbgzJBNuVpMJmZJpuq402-VzxAMnxBp12j4knrzRuHxAuchddeRKhi3ZvkcQGwzT3M-_h8K8eohMmtr0AQVqoOWTpNqkI_yHGPeVCgtqvtKnmXv8ML6ToCzvt-xE47MSVXlR3Ae/eozeyy464wki1ug/drugsy.zip
#!unzip -q -n drugsy.zip

import skimage.io as io
import matplotlib.pyplot as plt
import easy_cv_dataset as ds
import keras
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras_cv.layers import Resizing, RandomColorDegeneration, RandomRotation
from tensorflow.keras import layers

# Train-table
train_table = ds.image_dataframe_from_directory('drugsy copia 2/train')

# Split
train_table, valid_table = train_test_split(
    train_table,
    test_size=0.2,
    random_state=34,
    stratify=train_table['class']
)

# Preparing training and validation sets
batch_size = 9
img_height, img_width = 224, 224
augmenter = keras.Sequential(layers=[RandomColorDegeneration(0.5), RandomRotation((-20, 20))])

train_dataset = ds.image_classification_dataset_from_dataframe(
    train_table,
    batch_size=batch_size,
    shuffle=True,
    pre_batching_processing=Resizing(img_height, img_width),
    post_batching_processing=augmenter,
    do_normalization=True,
    class_mode='categorical')

valid_dataset = ds.image_classification_dataset_from_dataframe(
    valid_table,
    batch_size=batch_size,
    shuffle=False,
    pre_batching_processing=Resizing(img_height, img_width),
    do_normalization=True,
    class_mode='categorical'
)

# Neural network
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width, img_height, 3)
)

model = keras.models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(8, activation='softmax'))  # modificare con numero effettivo di classi

train_after_layer = 25

for layer in base_model.layers[:train_after_layer]:
    layer.trainable = False

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
    metrics=['accuracy']
)

# Train on training set
model.fit(train_dataset, epochs=4, validation_data=valid_dataset, verbose=True)

# Combine training and validation sets
combined_train_table = pd.concat([train_table, valid_table], ignore_index=True)
combined_train_table = combined_train_table.sample(frac=1).reset_index(drop=True)

combined_train_dataset = ds.image_classification_dataset_from_dataframe(
    combined_train_table,
    batch_size=batch_size,
    shuffle=True,
    pre_batching_processing=Resizing(img_height, img_width),
    post_batching_processing=augmenter,
    do_normalization=True,
    class_mode='categorical'
)

# Retrain on combined dataset
model.fit(combined_train_dataset, epochs=4, verbose=True)

# Performance
test_table = ds.image_dataframe_from_directory('drugsy copia 2/test')

test_dataset = ds.image_classification_dataset_from_dataframe(
    test_table, batch_size=batch_size,
    shuffle=False,
    pre_batching_processing=Resizing(img_height, img_width),
    do_normalization=True,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=True)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Confusion matrix and class accuracy
predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = []
for images, labels in test_dataset:
    true_classes.extend(np.argmax(labels.numpy(), axis=1))
conf_matrix = confusion_matrix(true_classes, predicted_classes)
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
min_accuracy_class_index = np.argmin(class_accuracies)

# Print class accuracies
print("Accuratezza per classe:")
for i, accuracy in enumerate(class_accuracies):
    print(f"Classe {i}: {accuracy}")

# Print class with the lowest accuracy
print(f"Classe con minore accuratezza: {min_accuracy_class_index}")

classi = ['cocaine', 'ecstasy', 'marijuana', 'mushrooms', 'origan', 'pills', 'salt', 'shrooms']

image_names = test_table.image

df_pred = pd.DataFrame(columns=['filename'] + classi)

for i, image_name in enumerate(image_names):
    # Ottieni le predizioni binarie per l'immagine corrente
    binary_predictions = [1 if j == predicted_classes[i] else 0 for j in range(len(classi))]
    # Aggiungi il nome dell'immagine e le predizioni binarie al DataFrame
    df_pred.loc[i] = [image_name] + binary_predictions

df_pred.to_csv('Predictions.csv', index=False)

import os
import shutil
import csv
import numpy as np
import skimage.io as io

def blurring(image_path, save_path):
    # Leggi l'immagine
    x = np.float64(io.imread(image_path))
    
    # Se l'immagine è a colori (3 canali), processa ciascun canale separatamente
    if len(x.shape) == 3:
        channels = []
        for c in range(x.shape[2]):
            channel = x[:, :, c]
            M, N = channel.shape
            m = np.fft.fftshift(np.fft.fftfreq(M))
            n = np.fft.fftshift(np.fft.fftfreq(N))
            l, k = np.meshgrid(n, m)
            D = np.sqrt(k**2 + l**2)
            D0 = 0.1
            H = (D <= D0)
            X = np.fft.fft2(channel)
            X = np.fft.fftshift(X)
            X = np.log(1 + np.abs(X))

            Y = H * X
            Y = np.log(1 + np.abs(Y))

            Y = np.fft.ifftshift(Y)
            y = np.real(np.fft.ifft2(Y))
            channels.append(y)
        
        # Ricostruisci l'immagine a colori combinando i canali processati
        y = np.stack(channels, axis=-1)
    else:
        # Se l'immagine è in bianco e nero, processa direttamente
        M, N = x.shape
        m = np.fft.fftshift(np.fft.fftfreq(M))
        n = np.fft.fftshift(np.fft.fftfreq(N))
        l, k = np.meshgrid(n, m)
        D = np.sqrt(k**2 + l**2)
        D0 = 0.1
        H = (D <= D0)
        X = np.fft.fft2(x)
        X = np.fft.fftshift(X)
        X = np.log(1 + np.abs(X))

        Y = H * X
        Y = np.log(1 + np.abs(Y))

        Y = np.fft.ifftshift(Y)
        y = np.real(np.fft.ifft2(Y))
    
    # Salva l'immagine blurrata
    io.imsave(save_path, np.uint8(y))
    return save_path

output_dir = 'blurred_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Creata cartella: {output_dir}")
else:
    print(f"Cartella già presente: {output_dir}")

# Percorso del file CSV
csv_file = "Predictions.csv"

# Leggi il file CSV
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Estrai il nome del file e le classi di appartenenza
        filename = row['filename']
        classes = [key.strip() for key, value in row.items() if key != 'filename' and value.strip() == '1']

        # Crea una sottocartella per ogni classe
        for class_name in classes:
            class_folder = os.path.join(output_dir, class_name + "_blurred")
            # Se la cartella della classe non esiste, creala
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
                print(f"Creata cartella: {class_folder}")
            else:
                print(f"Cartella già presente: {class_folder}")

            # Percorso per l'immagine blurrata nella cartella principale
            blurred_image_path = os.path.join(output_dir, filename)
        
            # Applica il blurring e salva l'immagine nella cartella principale
            blurring(blurred_image_path, blurred_image_path)
            print(f"Immagine {filename} blurrata e salvata in {blurred_image_path}")

        # Copia il file nella sottocartella corrispondente alla classe
        for class_name in classes:
            dest_file = os.path.join(output_dir, class_name + "_blurred", filename)
            try:
                shutil.copy(os.path.join(output_dir, filename), dest_file)
                print(f"Copiato {filename} in {class_name}")
            except FileNotFoundError:
                print(f"Il file {filename} non è stato trovato, non è stato copiato in {class_name}.")
