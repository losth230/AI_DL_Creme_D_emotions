# 🎵 Projet Deep Learning : Reconnaissance d'Émotions Audio (SER)

**Année :** 2025-2026
**Domaine :** Audio

---

## 📋 Description du Projet
Ce projet a pour but de construire un pipeline complet de Deep Learning capable de classifier l'émotion d'un locuteur à partir d'un fichier audio. Le système distingue 6 émotions : Colère, Dégoût, Peur, Joie, Neutre et Tristesse.

Nous avons mis en œuvre une approche comparative testant plusieurs architectures de réseaux de neurones (CNN, LSTM, CRNN) et utilisé des techniques avancées de traitement du signal (Mel-Spectrogrammes) et d'augmentation de données pour maximiser la précision.

### 🎯 Objectifs
* **ETL Audio :** Chargement, nettoyage et transformation des fichiers `.wav`.
* **Modélisation :** Conception et comparaison de modèles CNN 1D, LSTM et CRNN.
* **Optimisation :** Utilisation de la Data Augmentation et du Fine-Tuning.
* **Déploiement :** Inférence sur des fichiers audio externes.

---

## 💾 Jeu de Données (Dataset)

Le projet s'appuie sur le dataset **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset).

* **Contenu :** 7 442 clips originaux provenant de 91 acteurs différents.
* **Diversité :** Acteurs de 20 à 74 ans, issus de diverses ethnies (Afro-Américains, Asiatiques, Caucasiens, Hispaniques).
* **Labels :** Les phrases sont prononcées avec 6 émotions différentes (Anger, Disgust, Fear, Happy, Neutral, Sad) et 4 niveaux d'intensité.
* **Format Nom Fichiers** : `ID_Acteur_Phrase_Emotion_Intensité.wav` (ex: `1001_MAD_HAP_HIGH.wav`).

---

## ⚙️ Pipeline et Prétraitement

Pour transformer les signaux audio bruts en données intelligibles pour le réseau, nous avons appliqué le pipeline suivant :

1.  **Uniformisation :** Tous les fichiers audio sont chargés à une fréquence de 22050 Hz et ajustés (padding/truncating) à une durée fixe de **2.5 secondes**.
2.  **Data Augmentation :** Pour pallier la taille limitée du dataset, nous avons multiplié les données par 3 en générant des variantes :
    * *Noise Injection :* Ajout de bruit blanc.
    * *Time Shifting :* Décalage temporel.
3.  **Feature Extraction :**
    * Conversion du signal en **Mel-Spectrogrammes** (128 bandes de fréquences).
    * Passage à une échelle logarithmique (dB) pour mieux représenter la perception humaine du son.

---

## 🧠 Architectures des Modèles

Trois architectures ont été implémentées et comparées :

| Modèle | Type | Description |
| :--- | :--- | :--- |
| **CNN 1D** | Spatial | Traite le spectrogramme comme une image pour détecter des motifs locaux (pics de fréquence, énergie). Rapide mais manque de contexte temporel. |
| **LSTM** | Temporel | Réseau récurrent bidirectionnel. Analyse la séquence audio dans les deux sens pour comprendre l'évolution de l'émotion dans le temps. |
| **CRNN** | **Hybride** | **(Meilleur Modèle)** Combine l'extraction de caractéristiques du CNN avec la mémoire séquentielle du LSTM. Utilise la `BatchNormalization` pour stabiliser l'apprentissage. |

---

## 📈 Entraînement et Résultats

* **Stratégie :** Entraînement sur 40 époques avec `Adam` (lr=0.0005).
* **Régularisation :** Utilisation de `Dropout` élevé (0.3 - 0.4) et d'`EarlyStopping` pour éviter le surapprentissage.
* **Résultats :**
    * Les modèles de base stagnaient autour de 60%.
    * L'ajout de la **Data Augmentation** et des **Mel-Spectrogrammes** a permis une nette amélioration de la généralisation.
    * Le modèle hybride **CRNN** offre le meilleur compromis entre précision et stabilité.

---

## 🤖 Pistes d'Amélioration & IA Symbolique

Pour enrichir le projet avec une dimension symbolique (règles logiques) :

1.  **Logique de Seuil :** Si la probabilité de la classe prédite est `< 40%`, le système classe le résultat comme "Incertain" plutôt que de donner une réponse fausse.
2.  **Règles de Cohérence :** Si nous avions accès à la vidéo, nous pourrions implémenter des règles multimodales (ex: `SI Audio=Triste ET Vidéo=Sourire ALORS Sortie=Ironie`).

---

## 🚀 Installation et Utilisation

### Prérequis
Le projet nécessite Python 3.x et les librairies suivantes :
```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn tqdm