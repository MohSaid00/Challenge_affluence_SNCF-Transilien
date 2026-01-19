# Challenge : Prédiction de l'Affluence - Gares SNCF Transilien

##  Introduction

Ce projet a pour but de prédire l'affluence journalière (le nombre de validations) dans les gares du réseau SNCF Transilien. L'objectif est de créer un modèle de machine learning capable d'anticiper le trafic en se basant sur des données historiques et calendaires.

##  Contexte du Challenge

Ce projet a été réalisé dans le cadre du challenge "Anticipez l'affluence au sein des gares SNCF-Transilien !" disponible sur la plateforme challengedata.

- **[Lien vers le challenge](https://challengedata.ens.fr/challenges/149)**

##  Données Utilisées

Le modèle est entraîné à partir des fichiers de données suivants, qui doivent être placés dans un dossier nommé `data/` :

-   `x_train.csv` : Données d'entraînement contenant les features (date, nom de la gare, jours fériés, vacances scolaires, etc.).
-   `y_train.csv` : Données cibles contenant le nombre de validations journalières pour la période d'entraînement.
-   `x_test.csv` : Données de test avec les mêmes features que `x_train.csv`, pour la période à prédire.

##  Approches Modélisées

Ce dépôt contient deux implémentations différentes pour résoudre ce problème :

1.  **`predict.py`** : Première approche utilisant un modèle de Deep Learning de type **LSTM** (Long Short-Term Memory), spécialisé dans l'analyse de séquences temporelles.
2.  **`predict_lightgbm.py`** : Seconde approche utilisant un modèle de **LightGBM**, un algorithme de Gradient Boosting très performant et rapide sur les données tabulaires.