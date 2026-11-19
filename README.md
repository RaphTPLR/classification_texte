# FilmSentiment

Application Streamlit de classification de critiques de films en français.
Elle charge un modèle BiLSTM entraîné, applique le même prétraitement que lors
de l’entraînement et renvoie un sentiment positif ou négatif avec les deux
scores associés.

## Prérequis

- Python 3.11
- Environ 1 Go d’espace disponible pour l’installation de TensorFlow et de ses dépendances

## Installation locale

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement

```powershell
streamlit run app.py
```

L’application est ensuite disponible sur `http://localhost:8501`.

## Fichiers nécessaires

Les fichiers suivants doivent rester à la racine du dépôt :

- `app.py` : interface Streamlit ;
- `inference.py` : prétraitement et prédiction ;
- `bilstm.keras` : modèle entraîné ;
- `lstm_word2idx.json` : vocabulaire du modèle ;
- `preprocessing_params.json` : paramètres de prétraitement ;
- `requirements.txt` : dépendances Python ;
- `.streamlit/config.toml` : thème et configuration Streamlit.

Les fichiers `tfidf_*.joblib` sont des artefacts expérimentaux qui ne sont pas
utilisés par l’application. Ils restent disponibles localement mais sont exclus
du dépôt GitHub afin de réduire son poids.

## Prétraitement

1. Normalisation du texte avec `preprocess_common`.
2. Tokenisation en conservant les préfixes apostrophés comme `l'`, `d'` et `qu'`.
3. Conversion des tokens en indices avec `lstm_word2idx.json`.
4. Remplacement des mots inconnus par `<OOV>`.
5. Conservation des 310 premiers tokens, puis padding à droite avec `<PAD>`.
6. Classe positive lorsque la probabilité positive est supérieure ou égale à `0,5`.

Les fichiers texte sont décodés en UTF-8, avec Windows-1252 comme solution de repli.

## Déploiement gratuit sur Streamlit Community Cloud

1. Publier ce dossier dans un dépôt GitHub.
2. Ouvrir [share.streamlit.io](https://share.streamlit.io/).
3. Connecter le compte GitHub.
4. Créer une application à partir d’un dépôt existant.
5. Choisir le dépôt et la branche `main`.
6. Indiquer `app.py` comme fichier principal.
7. Dans les paramètres avancés, sélectionner Python 3.11.
8. Lancer le déploiement.

Les changements poussés ensuite sur la branche `main` seront automatiquement
redéployés par Streamlit Community Cloud.

## Structure envoyée sur GitHub

```text
classification_texte/
├── .streamlit/
│   └── config.toml
├── tests/
│   └── test_preprocessing.py
├── .gitattributes
├── .gitignore
├── app.py
├── bilstm.keras
├── inference.py
├── lstm_word2idx.json
├── preprocessing_params.json
├── README.md
└── requirements.txt
```
