from ultralytics import YOLO

# 1. Charger un modèle pré-entraîné léger et rapide (yolov8n ou yolov8s)
model = YOLO("yolov8s.pt")  # ou "yolov8n.pt" pour un modèle plus rapide mais un peu moins précis

# 2. Entraînement sur ton dataset
model.train(
    data="C:/Users\hassnae/Downloads/garbage claassification.v1i.yolov8/data.yaml",  # Ex : data.yaml
    epochs=50,              # Nombre d'époques (ajuste selon la taille du dataset)
    imgsz=640,              # Taille des images (640 est un bon compromis)
    batch=16,               # Taille de lot (augmente si tu as un GPU puissant)
    patience=10,            # Early stopping si pas d'amélioration
    device=0,               # Utiliser le GPU (0 = premier GPU dispo)
    workers=2,              # Nombre de processus de chargement de données
    name="pythonproject",  # Nom du dossier de sauvegarde
    optimizer="SGD",        # Optimiseur (SGD donne souvent de bons résultats)
    lr0=0.01,               # Taux d’apprentissage initial
    weight_decay=0.0005,    # Régularisation L2 pour éviter l’overfitting
    augment=True,           # Data augmentation activée
    cache=True              # Mise en cache pour un entraînement plus rapide
)
