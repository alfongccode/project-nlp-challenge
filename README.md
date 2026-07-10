### Reptilian Fact Checker
## About
- Reptilian fact checker is a fake news locator.
## Usage and Examples
- APP on HuggingFace Space -> [AI_ventura](https://huggingface.co/spaces/Bitnick42/reptilian-fact-checker)

## Problem statement
* **Task**: NLP news classificator (2 classes)
* **Goal**: Given a corpus of an article gets if is fake or real
* **Approach**: Fine-tune a pretrained MobileNetV2 backbone (ImageNet weights) with a custom classification head, using data augmentation and class weighting to handle class imbalance

## Dataset
- **Source:** data.csv
- **Columns (5):** label, title, text, subject, date
- **Size:** 36,083 registers total (after removing 3,859 repeated articles)
- **Split:** 80% train / 20% test
  - Train: 28,867 articles
  - Test: 7,216 articles
- **Class balance:** Balanced
- **License:** [GPL2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

## Model architecture

- **Optimizer:** Adam (learning_rate: 0.001)
- **Loss:** Sparse categorical crossentropy
- **Callbacks:** EarlyStopping (patience 5), ReduceLROnPlateau (factor 0.5, patience 2)

Results – accuracy/loss metrics, confusion matrix, sample outputs
## Setup & installation

| Dependency | Version |
| :--- | :---: |
| keras | ^3.15.0 |
| matplotlib | ^3.10.6 |
| tensorflow | ^2.21.0 |
| numpy | ^2.4.4 |
| pillow | ^12.1.1 |

- **Download repository**: `git clone https://github.com/alfongccode/project-1-brief-CNN.git`
- **Environment setup**:
    - Run model_2.ipynb notebook

## Project structure

```
├── dataset
|   ├──data.csv
|   └── validation_data.csv
├── main.ipynb
├── model.ipynb
├── model_2.ipynb
├── requirements.txt
├── preprocessing.py
└── README.md
``` 

## Tech stack
- **Language:** Python 3.13
- **Deep Learning:** TensorFlow / Keras (MobileNetV2 pretrained on ImageNet)
- **Data processing:** NumPy, Pillow (PIL)
- **Machine Learning utilities:** scikit-learn (train/test split, class weights, metrics)
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook 

## Authors
  - **Nicolas Mooney** - [https://github.com/NIKK014](https://github.com/NIKK014)
  - **Alfonso García Cortijo** - [https://github.com/alfongccode](https://github.com/alfongccode)