import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import Models

# Charger les données depuis le CSV
data = pd.read_csv('bluetooth_hopping_sequence.csv')
X = data['channel'].values

# Paramètres
sequence_length = 7  # Longueur de la séquence

# Fonction pour créer les séquences
def create_sequences(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length):
        sequence = X[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Créer les séquences
sequences = create_sequences(X, sequence_length)

# Créer un DataFrame à partir des séquences
columns = [f'feature_{i+1}' for i in range(sequence_length)]
sequences_df = pd.DataFrame(sequences, columns=columns)

# Ajouter la colonne cible (la valeur qui suit la séquence)
sequences_df['target'] = X[sequence_length:]

# Sauvegarder le DataFrame dans un nouveau CSV
sequences_df.to_csv('bluetooth_sequences.csv', index=False)


# Charger les données depuis le nouveau CSV
data = pd.read_csv('bluetooth_sequences.csv')

# Supposons que le CSV a 'feature1', 'feature2', ..., 'featureN' comme colonnes
# et 'target' comme colonne cible
X = data.drop(columns=['target']).values
y = data['target'].values
num_classes = 37  # Nombre de classes
# Créer des classes pour la cible
def create_classes(y, margin=3):
    classes = []
    for value in y:
        # Définir la classe basée sur la marge
        class_label = (value) // 1  # Adapte la logique selon ta distribution
        classes.append(class_label)
    return np.array(classes)

# Encoder les cibles
y_classes = create_classes(y)
encoder = OneHotEncoder(categories=[np.arange(num_classes)], sparse_output=False)
y_encoded = encoder.fit_transform(y_classes.reshape(-1, 1))

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2)


# Modèle
model = Models.CNN(sequence_length, num_classes)

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',  # Surveiller la perte de validation
    patience=5,  # Nombre d'époques sans amélioration avant d'arrêter
    restore_best_weights=True  # Restaurer les poids du meilleur modèle obtenu pendant l'entraînement
)
# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

prediction = model.predict(X_test)

# Visualiser les performances
nb_display = 15
plt.subplots(1,3)
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Precision over epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.subplot(1, 3, 3)
plt.scatter(np.arange(nb_display), np.argmax(prediction, axis=1)[:nb_display], label='Prediction', marker='x')
plt.scatter(np.arange(nb_display), np.argmax(y_test, axis=1)[:nb_display], label='True Value', marker='x')
plt.legend()
plt.title('Predictions vs True values')
plt.xlabel('Index')
plt.ylabel('Class')
# Pour obtenir les indices de classe prédits (l'indice de la probabilité maximale pour chaque prédiction)
predicted_classes = np.argmax(prediction, axis=1)

# Obtenir les vraies classes (l'indice de la vraie classe)
true_classes = np.argmax(y_test, axis=1)

# Calculer la différence entre les classes prédites et les vraies classes
differences = predicted_classes - true_classes

# Calculer l'écart type des différences
standard_deviation = np.std(differences)

# Afficher l'écart type
print(f"L'écart type entre les prédictions et les vraies classes est de : {standard_deviation:.2f}")
plt.show()
