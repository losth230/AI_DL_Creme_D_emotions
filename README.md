# Projet Deep Learning 2025-2026  
Reconnaissance des émotions avec TensorFlow (Audio + CSV, CREMA-D)

## 1. Description générale

Ce projet met en place un pipeline complet de Deep Learning avec TensorFlow pour la reconnaissance d’émotions à partir du dataset CREMA-D. Il est constitué de deux volets complémentaires :

- Partie I – Audio : classification des émotions à partir des fichiers audio (.wav).
- Partie II – CSV : prédiction de l’émotion affichée à partir des réponses d’annotateurs et des métadonnées des acteurs.

Les deux volets illustrent comment une même tâche (reconnaissance d’émotions) peut être abordée à partir de représentations différentes : signal audio brut d’un côté, données tabulaires (CSV) de l’autre.

---

## 2. Objectifs

- Concevoir un pipeline complet de deep learning avec TensorFlow.
- Manipuler un dataset réel (CREMA-D) et le prétraiter de bout en bout.
- Entraîner et évaluer plusieurs architectures de réseaux de neurones.
- Comparer différentes représentations de données (audio vs. métadonnées).
- Mettre en place les éléments classiques d’un projet reproductible : notebooks, scripts, rapport, README.
- Optionnel : préparer l’intégration d’éléments d’IA symbolique (règles, post-traitement logique).

---

## 3. Organisation du projet

### 3.1. Notebooks principaux

- `AudioTreatment.ipynb`  
  Pipeline de traitement audio : chargement des fichiers `.wav`, data augmentation, extraction de caractéristiques (Mel-Spectrogrammes), entraînement de modèles CNN / LSTM / CRNN, évaluation et inférence.

- `CSV_Training.ipynb`  
  Pipeline sur les données tabulaires : prétraitement de `finishedResponses.csv` et `VideoDemographics.csv`, feature engineering, encodage des variables, entraînement d’un modèle MLP avec TensorFlow.

## 4. Dataset : CREMA-D

Le projet repose sur le dataset CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) :

* Clips audio d’acteurs exprimant différentes émotions.
* Plusieurs acteurs, genres, groupes d’âge et origines ethniques.
* Six émotions principales :

  * Anger (A)
  * Disgust (D)
  * Fear (F)
  * Happiness (H)
  * Neutral (N)
  * Sadness (S)

Convention de nommage typique des fichiers audio :

```text
ActorID_PhraseType_Emotion_Intensité.wav
ex : 1001_DFA_ANG_HIGH.wav
```

Les fichiers CSV associés apportent des informations supplémentaires :

* `finishedResponses.csv` : réponses d’annotateurs, scores de perception, identifiant du clip, etc.
* `VideoDemographics.csv` : informations démographiques sur les acteurs (Genre, Age, Race, Ethnicity, ActorID, etc.).

---
## Installation et Utilisation

### Prérequis
Le projet nécessite Python 3.x et les librairies suivantes :
```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn tqdm
```
---

## 5. Partie I – Reconnaissance d’émotions à partir de l’audio

### 5.1. Objectif

Construire un modèle de Speech Emotion Recognition (SER) capable de prédire l’émotion exprimée dans un fichier audio parmi les six classes : A, D, F, H, N, S.

### 5.2. Pipeline audio dans `AudioTreatment.ipynb`

Pour transformer les signaux audio bruts en données intelligibles pour le réseau, nous avons appliqué le pipeline suivant :

1.  **Uniformisation :** Tous les fichiers audio sont chargés à une fréquence de 22050 Hz et ajustés (padding/truncating) à une durée fixe de **2.5 secondes**.
2.  **Data Augmentation :** Pour pallier la taille limitée du dataset, nous avons multiplié les données par 3 en générant des variantes :
    * *Noise Injection :* Ajout de bruit blanc.
    * *Time Shifting :* Décalage temporel.
3.  **Feature Extraction :**
    * Conversion du signal en **Mel-Spectrogrammes** (128 bandes de fréquences).
    * Passage à une échelle logarithmique (dB) pour mieux représenter la perception humaine du son.

---

### 5.3. Architectures de modèles audio

Trois familles de modèles sont explorées :

| Modèle                 | Type | Description                                                                                                                                                                  |
|:-----------------------| :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CNN 1D**             | Spatial | Traite le spectrogramme comme une image pour détecter des motifs locaux (pics de fréquence, énergie). Rapide mais manque de contexte temporel. Capte des motifs locaux (pics de fréquence, transitions rapides).                              |
| **LSTM**               | Temporel | Réseau récurrent bidirectionnel. Analyse la séquence audio dans les deux sens pour comprendre l'évolution de l'émotion dans le temps.                                        |
| **CRNN** (CNN + LSTM)  | **Hybride** | **(Meilleur Modèle)** Combine l'extraction de caractéristiques du CNN avec la mémoire séquentielle du LSTM. Utilise la `BatchNormalization` pour stabiliser l'apprentissage. |

## 5.4. Entraînement et Résultats

* **Stratégie :** Entraînement sur 40 époques avec `Adam` (lr=0.0005).
* **Régularisation :** Utilisation de `Dropout` élevé (0.3 - 0.4) et d'`EarlyStopping` pour éviter le surapprentissage.
* **Résultats :**
    * Les modèles de base stagnaient autour de 60%.
    * L'ajout de la **Data Augmentation** et des **Mel-Spectrogrammes** a permis une nette amélioration de la généralisation.
    * Le modèle hybride **CRNN** offre le meilleur compromis entre précision et stabilité.
Les modèles CNN et LSTM stagnaient autour de 60% alors que le CRNN atteignait 35% de précision. 
L'ajout de la Data Augmentation et des Mel-Spectrogrammes a permis une nette amélioration de la généralisation.
Le modèle hybride CRNN offre le meilleur compromis entre précision et stabilité avec une précision de 70%.

---

### 5.5. Inférence sur de nouveaux fichiers audio

Le notebook permet également :

* de charger un fichier audio externe,
* d’appliquer le même pipeline de prétraitement (resampling, padding, spectrogramme),
* de produire la prédiction du modèle (classe d’émotion et probabilités associées).

---

### 5.6. Pistes d'Amélioration & IA Symbolique

Pour enrichir le projet avec une dimension symbolique (règles logiques) :

1.  **Logique de Seuil :** Si la probabilité de la classe prédite est `< 40%`, le système classe le résultat comme "Incertain" plutôt que de donner une réponse fausse.
2.  **Règles de Cohérence :** Si nous avions accès à la vidéo, nous pourrions implémenter des règles multimodales (ex: `SI Audio=Triste ET Vidéo=Sourire ALORS Sortie=Ironie`).

---

## 6. PARTIE II : CSV

### 6.1. Objectif

Prédire l’émotion affichée (`dispEmo`) pour chaque clip à partir :

* des réponses des annotateurs (`finishedResponses.csv`),
* des métadonnées des acteurs (`VideoDemographics.csv`),
* de caractéristiques dérivées du nom du fichier (`clipName`).

Les classes cibles sont les mêmes que dans la partie audio : A, D, F, H, N, S.

---

### 6.2. Prétraitement dans `CSV_Training.ipynb`

1. Chargement et nettoyage des données

   * Chargement des deux fichiers CSV avec pandas.
   * Suppression des colonnes techniques (indices, IDs internes, colonnes inutiles pour l’apprentissage).
   * Suppression de la première colonne si elle ne contient que l’index.

2. Extraction de caractéristiques à partir de `clipName`

   * `Actor` : extrait des premiers caractères du nom de fichier, converti en entier, utilisé pour joindre les deux CSV.
   * `PhraseType` : extrait de la partie centrale du nom de fichier (ex. IEO, TIE, IOM, DFA, etc.).

3. Fusion des deux CSV

   * Fusion sur l’identifiant commun (`Actor` ou `ActorID`).
   * Suppression des colonnes redondantes après la fusion.

4. Gestion des valeurs manquantes et des valeurs aberrantes

   * Lignes contenant des valeurs manquantes critiques (ex. `dispVal`) supprimées.
   * Traitement des outliers sur `ttr` (Type/Token Ratio) via l’IQR :

     * calcul des quartiles Q1 et Q3,
     * découpage des valeurs au-delà de [Q1 - 1,5 × IQR, Q3 + 1,5 × IQR].

5. Construction des features et labels

   * `labels = dispEmo` (colonne cible).
   * `features = toutes les autres colonnes pertinentes`.
   * Encodage de `dispEmo` en entiers :

     * A → 0, D → 1, F → 2, H → 3, N → 4, S → 5.
   * Encodage des variables catégorielles (Gender, Age, Race, Ethnicity, PhraseType, etc.) en entiers.
   * Normalisation de la colonne `ttr` :

     ```python
     features['ttr'] = (features['ttr'] - features['ttr'].mean()) / features['ttr'].std()
     ```

---

### 6.3. Modèle TensorFlow pour les CSV

Un modèle MLP simple est utilisé :

```python
model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax)  # 6 classes d'émotions
])
```

* La loss utilisée est la sparse_categorical_crossentropy.
* Les labels (0 à 5) sont traités comme des valeurs numériques.

---

### 6.4. Entraînement et Évaluation
* Entraînement sur 100 époques avec `Adam` (lr=0.001)
* On évalue en regardant la quantité de prédictions justes sur l’ensemble de validation.

---

### 6.5. Résultats et Améliorations
* Le modèle atteint environ 18% de précision sur l’ensemble de validation ce qui est très peu.
* Cela s'explique par le fait que les données tabulaires contiennent peu d’informations discriminantes pour la tâche de reconnaissance des émotions.
En effet, les réponses ne permettent pas de réellement différencier une émotion d'une autre et les caractéristiques d'un acteur (âge, sexe, ethnie) n'ont pas d'impact direct sur l'émotion exprimée
étant capable de varier indépendamment.
* En conclusion, les données tabulaires seules sont insuffisantes pour une classification efficace des émotions.


