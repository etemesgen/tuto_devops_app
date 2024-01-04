import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Charger le modèle
model = pickle.load(open('model.pkl', 'rb'))

# Charger ou préparer les données de test
data = pd.read_csv('sample.csv',sep=';')
x_test = data.drop('target', axis=1) 
y_test = data['target']

# Faire des prédictions
y_pred = model.predict(x_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Définir un seuil de classification
seuil_classification = 0.90

# Vérifier si le seuil est atteint
if accuracy >= seuil_classification:
  print("Le seuil de classification est atteint.")
else:
  print("Le seuil de classification n'est pas atteint.")
