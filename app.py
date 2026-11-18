from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Noms des 10 classes Fashion-MNIST (indice = class_id)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# Schéma de l'entrée : une liste de pixels
class ImageInput(BaseModel):
    pixels: list[float]


# Création de l'API
app = FastAPI(
    title="Fashion-MNIST API",
    description="API de classification des vêtements Fashion-MNIST",
    version="1.0"
)

# Chemin du dossier contenant app.py
PROJECT_DIR = Path(__file__).resolve().parent

# Chemin du modèle
MODEL_PATH = PROJECT_DIR / "fashion_mnist.keras"

print("Chemin du modèle :", MODEL_PATH)
print("Fichier trouvé :", MODEL_PATH.exists())

# Chargement du modèle : une seule fois, au démarrage du serveur
model = tf.keras.models.load_model(MODEL_PATH)

print("Modèle chargé avec succès")
print("Forme d'entrée :", model.input_shape)
print("Forme de sortie :", model.output_shape)


@app.get("/")
def home():
    return {
        "message": "Bienvenue dans Fashion-MNIST API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_PATH.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape)
    }


@app.post("/predict")
def predict(data: ImageInput):
    # 1. Réception de la liste nommée "pixels"
    pixels = data.pixels

    # 2. Vérification : exactement 784 valeurs (28 x 28)
    if len(pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail="L'entrée doit contenir exactement 784 pixels."
        )

    # 3. Conversion en tableau NumPy de type float32
    image = np.array(pixels, dtype=np.float32)

    # 4. Reconstruction d'un batch de forme (1, 28, 28)
    image = image.reshape(1, 28, 28)

    # 5. Normalisation si les pixels sont entre 0 et 255
    if image.max() > 1.0:
        image = image / 255.0

    # 10. Début de la mesure de latence
    start = time.perf_counter()

    # 6. Exécution du modèle en mode inférence
    logits = model(image, training=False)

    # 7. Transformation des logits en probabilités
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]

    # 10. Fin de la mesure de latence (en millisecondes)
    latency_ms = (time.perf_counter() - start) * 1000

    # 8. Classe la plus probable
    class_id = int(np.argmax(probabilities))

    # 9. Score de confiance
    confidence = float(probabilities[class_id])

    # 11. Retour des résultats au format JSON
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2)
    }
