# Personalized Healthcare & Medicine Recommendation System

A small Flask web app that takes a patient's age, gender, blood pressure, cholesterol and symptoms, then shows likely conditions (from one machine learning model and one deep learning model), a risk level, medicines from a drug list, and simple care tips. There is no login: open the page and use it.

Built for the second data science internship project (Python, Flask, scikit-learn, TensorFlow/Keras).

> **Important:** this is a learning project. It is not a doctor and it does not give a diagnosis or a prescription. Always check medicines and doses with a doctor or pharmacist.

## What it does

- **Health check form** - enter age, gender, BP level, cholesterol level and symptoms. Optionally pick a diagnosis you already have.
- **Result page** - likely conditions with the first pick of each model, risk level (Low / Medium / High), medicines from the drug list, and care tips (food, daily habits, what to watch, when to see a doctor).
- **Patient records** - all 349 patients from the dataset, each with their own medicine and care recommendation.
- **Data & models page** - EDA charts, the scores of the two models and plain-language notes on how accurate they really are.

## Project structure

```
healthcare_reco/
├── data_processing.py   # load + clean data, EDA, train + test the 2 models, save them
├── app.py               # Flask web app
├── requirements.txt
├── data/
│   ├── Cleaned_Dataset.csv     # patient records (input)
│   ├── medicine.csv            # drug list (input)
│   ├── medicine_clean.csv      # created by data_processing.py
│   ├── patients_clean.csv      # created by data_processing.py
│   └── eda/                    # charts created by data_processing.py
├── models/                     # ml_model, dl_model, metrics.json (created automatically)
└── templates/                  # HTML pages (base, index, result, patients, insights)
```

## How to run

You need Python 3.9 to 3.13.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install the libraries
pip install -r requirements.txt

# 3. train the 2 models (shows the EDA and the model scores in the terminal)
python data_processing.py

# 4. start the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Step 3 is optional. If the `models/` folder is empty, or the saved models were made with different scikit-learn / TensorFlow versions, `app.py` trains them by itself on start-up (under a minute).

`tensorflow` is a big download. If it cannot be installed on your computer, the project still runs: the neural network is then trained with scikit-learn's `MLPClassifier` using the same layer sizes, and the Data & models page says which one was used.

## How it works

1. **Cleaning** - Yes/No symptoms become 1/0, gender becomes one column, some disease names that were cut off (for example "Chronic Obstructive Pulmonary...") are fixed, and the drug names in `medicine.csv` are tidied.
2. **Models** - only two are kept, the best of the candidates that were tried:
   - **Machine learning model:** Support Vector Classifier with a linear kernel (features are scaled first).
   - **Deep learning model:** a small neural network in TensorFlow / Keras: 8 inputs, Dense 64 (ReLU), Dropout, Dense 32 (ReLU), Dropout, softmax over the diseases. Trained for 30 epochs.
   - Both predict the **disease**. The app ranks diseases by the average of the two models' probabilities and also shows each model's own first pick.
   - Risk level is almost fixed for each disease in this dataset, so it is looked up from the disease instead of predicted separately.
3. **Medicine recommendation** - the disease is matched to a medicine group in `medicine.csv` (for example Diabetes, Infection, Pain). Inside that group, drugs are ranked by TF-IDF cosine similarity between the patient's text and the drug description.
4. **Care tips** - rule-based advice for 13 condition groups, plus extra notes for high BP, high cholesterol, age 60+ and fever with breathing trouble.

## Model results

Scores come from 5-fold cross-validation on the 300 unique patient rows (114 diseases).

| Model | Top-1 | Top-3 |
|---|---|---|
| Machine learning: SVC (linear) | 8.3% | 13.3% |
| Deep learning: neural network | 5.3% | 14.0% |
| Both combined (average) | 6.0% | 14.7% |
| Guessing the most common disease | 5.3% | - |

Top-1 means the first guess is right. Top-3 means the right disease is among the three best guesses.

All three are close to guessing. There are 114 diseases, 60 of them have a single patient, and only 8 inputs are available, so the data cannot separate them. That is a limit of the dataset, not a bug. The gaps between the models are within normal run-to-run variation (the neural network changes by a few points with a different random seed), so none of them is clearly better. The models were chosen on this same cross-validation, so treat the scores as approximate.

This is why the form lets the user pick a known diagnosis, and why a symptoms-only result shows a low-certainty warning.

## Pages

| URL | What it shows |
|---|---|
| `/` | Health check form |
| `/predict` | Result for the submitted form (POST) |
| `/patients` | Searchable table of all patient records |
| `/patient/<id>` | Recommendation for one patient record |
| `/insights` | Data charts, model comparison, accuracy notes |

## Troubleshooting

**An `AttributeError` or other error when the models load (for example "object has no attribute ...")**
The saved model files were made with a different scikit-learn or TensorFlow version. Stop the app, delete everything inside the `models/` folder, and run `python app.py` again (or `python data_processing.py`). The newest `app.py` checks the versions and retrains by itself.

**`ModuleNotFoundError`**
Run `pip install -r requirements.txt` in the same Python you use to start the app. If only `tensorflow` fails to install, you can remove that line; the app falls back to scikit-learn's neural network.

**The page still shows an old error after fixing it**
Open `http://127.0.0.1:5000` fresh. Refreshing an error page on `/predict` resends the same failing request.

## Limitations

- Medicines come only from `medicine.csv`. Some conditions (cancers, asthma, genetic conditions) have no matching drug group, so the app says so and suggests a specialist.
- Some descriptions inside `medicine.csv` are inconsistent, and the app shows them as they are.
- Blood pressure and cholesterol are stored as 0 / 1 / 2 in the dataset. The app shows them as Low / Normal / High.
- The care tips are general advice written for this project, not medical guidance.

## Possible future work

- A larger dataset with more patients per disease and more measurements (glucose, heart rate, lab values).
- Collaborative and hybrid filtering once there is data about many users and their outcomes.
- Login and roles (Admin / User / Analyst) if the app is ever deployed for real users.
- Drug-allergy and drug-interaction checks.
