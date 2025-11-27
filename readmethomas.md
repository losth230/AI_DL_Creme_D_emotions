````markdown
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

## 5. Partie I – Reconnaissance d’émotions à partir de l’audio

### 5.1. Objectif

Construire un modèle de Speech Emotion Recognition (SER) capable de prédire l’émotion exprimée dans un fichier audio parmi les six classes : A, D, F, H, N, S.

### 5.2. Pipeline audio dans `AudioTreatment.ipynb`

1. Chargement et uniformisation du signal

   * Chargement des fichiers `.wav`.
   * Remise à une fréquence d’échantillonnage fixe (par exemple 22050 Hz).
   * Tronquage ou padding des signaux pour obtenir une durée fixe (par exemple 2,5 secondes).

2. Data augmentation

   * Injection de bruit (noise injection) pour simuler un environnement plus varié.
   * Décalage temporel (time shifting) pour augmenter la diversité des exemples.
   * L’objectif est d’augmenter la taille effective du dataset et de réduire l’overfitting.

3. Extraction de caractéristiques

   * Transformation du signal audio en Mel-Spectrogrammes (nombre de bandes Mel fixe, ex. 128).
   * Passage en échelle logarithmique (dB) pour mieux refléter la perception humaine du son.
   * Les spectrogrammes obtenus sont utilisés comme entrée des modèles.

### 5.3. Architectures de modèles audio

Trois familles de modèles sont explorées :

* CNN 1D / 2D

  * Traite le spectrogramme comme une image ou une carte de caractéristiques.
  * Capte des motifs locaux (pics de fréquence, transitions rapides).

* LSTM

  * Réseau récurrent pour exploiter la dimension temporelle.
  * Prend en compte la dynamique de la séquence audio sur la durée.

* CRNN (CNN + LSTM)

  * Combinaison d’un CNN (extraction des caractéristiques locales) et d’un LSTM (mémoire temporelle).
  * Ajout possible de Batch Normalization et Dropout pour stabiliser et régulariser l’entraînement.
  * Dans la pratique, c’est souvent le modèle le plus performant et stable sur ce type de tâche.

### 5.4. Entraînement

* Optimiseur typique : Adam, avec un taux d’apprentissage de l’ordre de 5e-4.
* Entraînement sur plusieurs époques (par exemple 30 à 40), avec :

  * validation sur un sous-ensemble des données,
  * Early Stopping pour arrêter l’entraînement en cas de surapprentissage,
  * Dropout pour limiter l’overfitting.

Les métriques principales sont la loss (par exemple cross-entropy) et l’accuracy sur l’ensemble de validation / test. Une matrice de confusion peut être utilisée pour analyser les erreurs par classe.

### 5.5. Résultats :
Les modèles CNN et LSTM stagnaient autour de 60% alors que le CRNN atteignait 35% de précision. 
L'ajout de la Data Augmentation et des Mel-Spectrogrammes a permis une nette amélioration de la généralisation.
Le modèle hybride CRNN offre le meilleur compromis entre précision et stabilité avec une précision de 70%.

### 5.6. Inférence sur de nouveaux fichiers audio

Le notebook permet également :

* de charger un fichier audio externe,
* d’appliquer le même pipeline de prétraitement (resampling, padding, spectrogramme),
* de produire la prédiction du modèle (classe d’émotion et probabilités associées).

---

## 6. Partie II – Prédiction d’émotions à partir des CSV

### 6.1. Objectif

Prédire l’émotion affichée (`dispEmo`) pour chaque clip à partir :

* des réponses des annotateurs (`finishedResponses.csv`),
* des métadonnées des acteurs (`VideoDemographics.csv`),
* de caractéristiques dérivées du nom du fichier (`clipName`).

Les classes cibles sont les mêmes que dans la partie audio : A, D, F, H, N, S.

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

### 6.3. Modèle TensorFlow pour les CSV

Un modèle MLP simple est utilisé :

```python
model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu')
])
```

Dans la version de base :

* La loss utilisée est la MSE (erreur quadratique moyenne).
* Les labels (0 à 5) sont traités comme des valeurs numériques.

Amélioration recommandée :

* Ajouter une couche de sortie adaptée à la classification :

  ```python
  model.add(layers.Dense(6, activation='softmax'))
  ```
* Utiliser une loss de classification :

  ```python
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
  )
  ```

### 6.4. Entraînement et évaluation

* Découpage train/test avec `train_test_split` (par exemple 80 % / 20 %).
* Entraînement sur plusieurs époques avec suivi des métriques.
* Évaluation sur le jeu de test (accuracy, éventuellement matrice de confusion).

---

## 7. Pistes d’intégration d’IA symbolique

Les deux volets peuvent être enrichis par des éléments symboliques :

* Post-traitement logique sur les prédictions :

  * si la probabilité maximale est trop faible, renvoyer une classe "incertain" plutôt qu’une émotion aléatoire.
* Règles sur les métadonnées (CSV) :

  * utiliser l’intensité, le type de phrase ou certains profils d’acteurs pour corriger ou filtrer certaines prédictions.
* Fusion audio + CSV (multimodal) :

  * combiner un score audio et un score "métadonnées" et appliquer des règles de cohérence entre les deux sources.

Ces éléments peuvent être décrits et discutés dans le rapport, même s’ils ne sont pas entièrement implémentés.

---

## 8. Installation

### 8.1. Prérequis

* Python 3.x
* Bibliothèques Python principales :

  * `tensorflow`
  * `librosa` (pour le traitement audio)
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`
  * `tqdm` (facultatif pour les barres de progression)

### 8.2. Installation des dépendances

```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

## 9. Utilisation

### 9.1. Lancer la partie audio

1. Placer les fichiers audio CREMA-D dans `CREMA-D/Audio/`.
2. Ouvrir le notebook :

```bash
jupyter notebook AudioTreatment.ipynb
```

3. Exécuter les cellules dans l’ordre :

   * chargement des données,
   * data augmentation,
   * extraction des spectrogrammes,
   * définition et entraînement des modèles,
   * évaluation et visualisation des résultats,
   * inférence sur de nouveaux fichiers audio.

### 9.2. Lancer la partie CSV

1. Placer `finishedResponses.csv` et `VideoDemographics.csv` dans `CREMA-D/`.
2. Ouvrir le notebook :

```bash
jupyter notebook CSV_Training.ipynb
```

3. Exécuter les cellules dans l’ordre :

   * chargement et nettoyage des CSV,
   * fusion et feature engineering,
   * encodage et normalisation,
   * définition et entraînement du modèle,
   * évaluation et prédictions.

---

## 10. Livrables attendus

Dans le cadre du projet de module, les livrables typiques sont :

1. Une implémentation complète du pipeline :

   * notebooks et/ou scripts exploitables,
   * modèles entraînés sauvegardés (dans `models/` par exemple).

2. Un rapport détaillé :

   * description du dataset,
   * détail du prétraitement (audio et CSV),
   * architectures de modèles,
   * résultats expérimentaux (métriques, courbes, matrices de confusion),
   * analyse critique et pistes d’amélioration,
   * éventuellement description de la partie IA symbolique.

3. Une démonstration :

   * via un notebook interactif,
   * ou via un script simple permettant de tester le modèle sur un nouveau fichier audio ou un exemple CSV.