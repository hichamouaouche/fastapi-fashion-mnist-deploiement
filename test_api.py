"""Tests de l'API Fashion-MNIST.

Lancer d'abord le serveur :  uvicorn app:app --reload
puis exécuter :              python test_api.py
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf

# URL de la route /predict (modifiable via la variable API_URL)
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
PROJECT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = PROJECT_DIR / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ---------------------------------------------------------------
# 1. Chargement de Fashion-MNIST
# ---------------------------------------------------------------
(_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


def predict_image(image):
    """Envoie une image 28x28 à l'API et retourne la réponse JSON."""
    # Transformation de l'image en une liste de 784 pixels
    payload = {"pixels": image.reshape(784).astype(float).tolist()}
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------
# 2. Test d'une image
# ---------------------------------------------------------------
print("=" * 70)
print("TEST D'UNE IMAGE")
print("=" * 70)

index = 0
image = test_images[index]
true_label = int(test_labels[index])

payload = {"pixels": image.reshape(784).astype(float).tolist()}
print("Nombre de pixels envoyés :", len(payload["pixels"]))

response = requests.post(
    API_URL,
    json=payload
)
result = response.json()

print("Réponse de l'API :")
print(json.dumps(result, indent=4, ensure_ascii=False))
print("Classe réelle  :", CLASS_NAMES[true_label])
print("Classe prédite :", result["class_name"])
print("Résultat       :",
      "Correct" if result["class_id"] == true_label else "Incorrect")

# Affichage de l'image testée
plt.figure(figsize=(4, 4.4))
plt.imshow(image, cmap="gray")
plt.title(f"Image {index}\nRéelle : {CLASS_NAMES[true_label]} | "
          f"Prédite : {result['class_name']} ({result['confidence']:.2f})")
plt.axis("off")
plt.tight_layout()
plt.savefig(CAPTURES_DIR / "image_testee.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 3. Test de plusieurs images (20)
# ---------------------------------------------------------------
print()
print("=" * 70)
print("TEST DE 20 IMAGES")
print("=" * 70)

N_IMAGES = 20
rows = []

print(f"{'Image':<7}{'Classe réelle':<15}{'Classe prédite':<16}"
      f"{'Confiance':<11}{'Latence':<11}{'Résultat'}")
print("-" * 70)

for i in range(N_IMAGES):
    true_id = int(test_labels[i])
    res = predict_image(test_images[i])
    correct = res["class_id"] == true_id

    rows.append({
        "image": i,
        "true_id": true_id,
        "true_name": CLASS_NAMES[true_id],
        "pred_id": res["class_id"],
        "pred_name": res["class_name"],
        "confidence": res["confidence"],
        "latency_ms": res["latency_ms"],
        "correct": correct,
    })

    print(f"{i:<7}{CLASS_NAMES[true_id]:<15}{res['class_name']:<16}"
          f"{res['confidence']:<11.2f}{res['latency_ms']:>6.2f} ms   "
          f"{'Correct' if correct else 'Incorrect'}")

n_correct = sum(r["correct"] for r in rows)
latencies = [r["latency_ms"] for r in rows]
accuracy = n_correct / N_IMAGES
mean_latency = float(np.mean(latencies))

print("-" * 70)
print(f"Prédictions correctes : {n_correct}/{N_IMAGES}")
print(f"Taux de réussite      : {accuracy * 100:.0f} %")
print(f"Latence moyenne       : {mean_latency:.2f} ms")

# Grille des 20 images testées (vert = correct, rouge = incorrect)
fig, axes = plt.subplots(4, 5, figsize=(10, 9))
for ax, r in zip(axes.flat, rows):
    ax.imshow(test_images[r["image"]], cmap="gray")
    ax.set_title(f"{r['image']} : {r['pred_name']}\n({r['true_name']})",
                 fontsize=9, color="green" if r["correct"] else "red")
    ax.axis("off")
plt.suptitle("20 images testées : prédite (réelle)")
plt.tight_layout()
plt.savefig(CAPTURES_DIR / "predictions_20.png", dpi=120)
plt.show()

# ---------------------------------------------------------------
# 4. Test d'une entrée incorrecte : seulement trois pixels
# ---------------------------------------------------------------
print()
print("=" * 70)
print("TEST D'UNE ENTRÉE INCORRECTE (3 pixels)")
print("=" * 70)

bad_response = requests.post(API_URL, json={"pixels": [0.0, 0.5, 1.0]})
print("Code HTTP :", bad_response.status_code)
print("Réponse   :", json.dumps(bad_response.json(), ensure_ascii=False))

# ---------------------------------------------------------------
# Sauvegarde des résultats (utilisés dans le compte rendu)
# ---------------------------------------------------------------
with open(PROJECT_DIR / "resultats.json", "w", encoding="utf-8") as f:
    json.dump({
        "single": {"image": index, "true_name": CLASS_NAMES[true_label],
                   "response": result},
        "rows": rows,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "mean_latency_ms": round(mean_latency, 2),
        "bad_input": {"status_code": bad_response.status_code,
                      "response": bad_response.json()},
    }, f, indent=2, ensure_ascii=False)

print("\nRésultats enregistrés dans resultats.json")
