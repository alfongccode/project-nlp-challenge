# Real or Fake? — News Classification with NLP

## About

NLP project for the Ironhack bootcamp. We built a classifier that predicts if a news article is fake or real, using Bag of Words + Logistic Regression. Final accuracy: 99% on the test set.

Sample predictions on the validation set:

```
Predicted FAKE:
BUCKLE UP AMERICA: Dinesh D'Souza's Is About To EXPOSE The Democrat Party ...WATCH
ONLY IN DETROIT: Entitled Squatter Gets Squatted On [VIDEO]

Predicted REAL:
Barcelona balances security and freedom after deadly attacks
Putin calls tougher North Korea sanctions senseless, warns of 'global catastrophe'
```

## Problem statement

Predict if an article is fake (0) or real (1) from its title and text. Train on `data.csv`, then predict the labels of `validation_data.csv` (which comes with all labels set to 2) and save the result in the same CSV format.

## Dataset

- Provided by Ironhack. News articles from 2016-2017, mostly US politics.
- 39,942 labeled articles. After removing 3,859 duplicate titles: 36,083.
- Classes are almost balanced: 54% real, 46% fake.
- Columns: `label`, `title`, `text`, `subject`, `date`.
- Validation set: 4,956 articles without labels.

Important: we don't use `subject` as a feature. It separates the classes perfectly (each subject value only appears in one class), so using it would be learning the labeling process instead of the language. Also, 99.8% of real articles contain "(Reuters)" — the model has easy shortcuts in this dataset, which is why the 99% has to be taken with some context.

## Model

```
title + text -> preprocessing.py -> CountVectorizer (BoW) -> LogisticRegression -> 0/1
```

Preprocessing: lowercase, remove punctuation and stopwords, lemmatize (NLTK).

- Stratified 80/20 train/test split, done before fitting anything.
- The test set is only used once, at the end.
- The trained pipeline is saved to `nlp_model.pkl` with joblib.

We also compared BoW / TF-IDF / TF-IDF bigrams with LogReg / Naive Bayes / LinearSVC (3-fold CV). All results were within ~1 point of each other, so we kept the simplest and most interpretable option.

## Results

- Test accuracy: **0.99** (7,217 articles)
- Precision and recall: 0.99 for both classes

Confusion matrix (test set):

```
                predicted
                fake    real
true  fake      3291      22
      real        19    3885
```

The strongest fake indicators are words like *video, image, breaking, featured*. The strongest real indicators are weekday names (from datelines like "on Tuesday"), *washington* and *reuters*.

On the validation set the model predicted 3,495 fake / 1,461 real (71% fake). That matches what the split arithmetic of the original corpus predicts (~3,538 / ~1,418), so we estimate ~99% accuracy on the validation data.

## Setup & installation

```bash
git clone <repo-url>
cd project-nlp-challenge

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter lab model.ipynb
```

`requirements.txt`:

```
pandas
scikit-learn
matplotlib
nltk
joblib
jupyterlab
```

NLTK data (stopwords, wordnet) downloads automatically the first time `preprocessing.py` is imported.

To use the saved model:

```python
import joblib
from preprocessing import get_preprocessing_attributes

model = joblib.load('nlp_model.pkl')
df = get_preprocessing_attributes(new_data)   # needs 'title' and 'text' columns
labels = model.predict(df['preprocessed_text'])
```

## Project structure

```
├── dataset/
│   ├── data.csv
│   └── validation_data.csv
├── model.ipynb                   # EDA, training, evaluation, predictions
├── preprocessing.py              # text cleaning and lemmatization
├── nlp_model.pkl                 # saved pipeline
├── validation_predictions.csv    # predicted labels
├── fake_news_presentation.pptx
└── README.md
```

## Tech stack

Python · pandas · scikit-learn · NLTK · matplotlib · Jupyter

## Future improvements

- Try TF-IDF bigrams + LinearSVC (scored slightly higher in our comparison)
- Fine-tune DistilBERT
- Deploy as a Hugging Face Space
- Test on news from other years/sources — the model learned the style of these sources, it doesn't detect "truth"

## Authors

Alfon and Nico — Ironhack AI/Data Science Bootcamp, Madrid
