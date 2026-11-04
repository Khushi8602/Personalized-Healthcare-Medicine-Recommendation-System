"""
data_processing.py
------------------
Everything that happens before the web app: load the two CSV files, clean them,
run the EDA, train the two final models and save them to /models.

The two models (picked after comparing several candidates, see the report):
  * ML model : Support Vector Classifier (linear kernel)
  * DL model : small neural network (TensorFlow / Keras)

Run it once:   python data_processing.py
The Flask app (app.py) only loads what this script saves.
"""

import json
import os
import re
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no window needed, we only save the charts to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep TensorFlow quiet

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EDA_DIR = DATA_DIR / "eda"

PATIENT_FILE = DATA_DIR / "Cleaned_Dataset.csv"
MEDICINE_FILE = DATA_DIR / "medicine.csv"

SYMPTOMS = ["fever", "cough", "fatigue", "difficulty_breathing"]
# the order here matters, the app builds its input rows in exactly this order
FEATURES = SYMPTOMS + ["age", "gender_male", "blood_pressure", "cholesterol_level"]

# The patient file stores some disease names twice or cut off with "...".
NAME_FIXES = {
    "Urinary Tract Infection (UTI)": "Urinary Tract Infection",
    "Chronic Obstructive Pulmonary...": "Chronic Obstructive Pulmonary Disease (COPD)",
    "Myocardial Infarction (Heart...": "Myocardial Infarction (Heart Attack)",
    "Obsessive-Compulsive Disorde...": "Obsessive-Compulsive Disorder (OCD)",
    "Systemic Lupus Erythematosus...": "Systemic Lupus Erythematosus (SLE)",
    "Eating Disorders (Anorexia": "Eating Disorders (Anorexia, Bulimia)",
}

# medicine.csv has one "Reason" per drug (Hypertension, Infection, Pain ...).
# This maps a disease name to the Reason groups that fit it. First keyword hit
# wins, so the more specific words are listed first. A disease with no match
# gets no medicine from this file (cancers, genetic conditions, asthma, etc.
# have no drug group in medicine.csv).
DISEASE_TO_REASONS = [
    ("hypertensive", ["Hypertension"]),
    ("hypertension", ["Hypertension"]),
    ("atherosclerosis", ["Hypertension", "Angina"]),
    ("coronary", ["Angina", "Hypertension"]),
    ("myocardial", ["Angina", "Thrombolysis"]),
    ("stroke", ["Thrombolysis", "Hypertension"]),
    ("diabetes", ["Diabetes"]),
    ("hyperglycemia", ["Diabetes"]),
    ("hypoglycemia", ["Diabetes"]),
    ("hypothyroidism", ["Hypothyroidism"]),
    ("hyperthyroidism", ["Hyperthyroidism"]),
    ("malaria", ["Malarial", "Fever"]),
    ("dengue", ["Fever", "Pyrexia", "Viral"]),
    ("typhoid", ["Infection", "Fever"]),
    ("influenza", ["Viral", "Fever", "Pyrexia"]),
    ("common cold", ["Fever", "Allergies", "Viral"]),
    ("allergic rhinitis", ["Allergies"]),
    ("sinusitis", ["Infection", "Allergies"]),
    ("gastroenteritis", ["Diarrhoea", "Infection"]),
    ("cholera", ["Diarrhoea", "Infection"]),
    ("amoeb", ["Amoebiasis"]),
    ("scabies", ["Scabies"]),
    ("hiv", ["Viral", "Infection"]),
    ("hepatitis", ["Viral", "Digestion"]),
    ("measles", ["Viral", "Fever"]),
    ("mumps", ["Viral", "Fever"]),
    ("rubella", ["Viral", "Fever"]),
    ("chickenpox", ["Viral", "Fever"]),
    ("zika", ["Viral", "Fever"]),
    ("ebola", ["Viral", "Fever"]),
    ("rabies", ["Viral", "Vaccines"]),
    ("polio", ["Viral", "Vaccines"]),
    ("tetanus", ["Infection", "Vaccines"]),
    ("tuberculosis", ["Infection"]),
    ("pneumonia", ["Infection", "Fever"]),
    ("bronchitis", ["Infection", "Fever"]),
    ("urinary tract", ["Infection"]),
    ("tonsillitis", ["Infection", "Fever"]),
    ("otitis", ["Infection", "Pain"]),
    ("conjunctivitis", ["Infection"]),
    ("osteomyelitis", ["Infection", "Pain"]),
    ("sepsis", ["Infection"]),
    ("lyme", ["Infection"]),
    ("cholecystitis", ["Infection", "Pain"]),
    ("diverticulitis", ["Infection", "Pain"]),
    ("appendicitis", ["Infection", "Pain"]),
    ("pancreatitis", ["Pain", "Digestion"]),
    ("colitis", ["Digestion", "Pain"]),
    ("crohn", ["Digestion", "Pain"]),
    ("cirrhosis", ["Digestion"]),
    ("liver disease", ["Digestion"]),
    ("hemorrhoid", ["Haemorrhoid"]),
    ("rheumatoid", ["Arthritis", "Pain"]),
    ("osteoarthritis", ["Arthritis", "Pain"]),
    ("arthritis", ["Arthritis", "Pain"]),
    ("fibromyalgia", ["Pain"]),
    ("gout", ["Gout"]),
    ("osteoporosis", ["Osteoporosis"]),
    ("endometriosis", ["Pain"]),
    ("migraine", ["Migraine", "Pain"]),
    ("parkinson", ["Parkinson"]),
    ("alzheimer", ["Alzheimer"]),
    ("dementia", ["Alzheimer"]),
    ("depression", ["Depression"]),
    ("anxiety", ["Anxiety"]),
    ("obsessive", ["Anxiety", "Depression"]),
    ("bipolar", ["Psychosis", "Depression"]),
    ("schizophrenia", ["Schizophrenia", "Psychosis"]),
    ("attention", ["Adhd"]),
    ("adhd", ["Adhd"]),
    ("anemia", ["Anaemia"]),
    ("sickle", ["Anaemia"]),
    ("glaucoma", ["Glaucoma"]),
    ("acne", ["Acne"]),
    ("eczema", ["Allergies"]),
    ("vertigo", ["Vertigo"]),
]


# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------

def load_patients():
    df = pd.read_csv(PATIENT_FILE)
    df["disease"] = df["disease"].str.strip().replace(NAME_FIXES)

    # Yes/No columns -> 1/0, gender -> one column
    for col in SYMPTOMS:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)
    df["gender_male"] = (df["gender"].str.strip().str.lower() == "male").astype(int)

    # The file already has scaled copies of age/bp/cholesterol. The tree models
    # don't care about scaling, and the app takes the real values, so I only
    # keep the raw columns.
    df = df.drop(columns=["age_scaled", "bp_scaled", "chol_scaled"])
    df["patient_id"] = np.arange(1, len(df) + 1)
    return df


def clean_drug_name(raw):
    """medicine.csv glues several pack sizes into one string, e.g.
    'A CN Gel(Topical) 20gmA CN Soap 75gm'. Keep only the first product."""
    raw = str(raw).strip()
    hit = re.search(r"(?:\d+(?:\.\d+)?\s?(?:gm|ml|mg|mcg|iu|IU|g)|\d+\s?'[sS])(?=[A-Z])", raw)
    return raw[: hit.end()].strip() if hit else raw


def load_medicines():
    med = pd.read_csv(MEDICINE_FILE)
    med["drug_name"] = med["Drug_Name"].apply(clean_drug_name)
    # the "Â" is leftover encoding junk from the original file
    med["description"] = (med["Description"].fillna("")
                          .str.replace("Â", "", regex=False)
                          .str.replace(r"\s+", " ", regex=True).str.strip())
    med["reason"] = med["Reason"].str.strip()
    med = med.drop_duplicates(subset=["drug_name", "reason"]).reset_index(drop=True)
    return med[["drug_name", "reason", "description"]]


def reasons_for(disease):
    """Return the medicine groups that fit a disease name (may be empty)."""
    name = disease.lower()
    for keyword, reasons in DISEASE_TO_REASONS:
        if keyword in name:
            return reasons
    return []


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------

def run_eda(patients, medicines):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("\n--- Quick look at the data ---")
    print("patients :", patients.shape, "| unique rows:",
          patients.drop(columns="patient_id").drop_duplicates().shape[0])
    print("diseases :", patients["disease"].nunique(),
          "| diseases with only 1 patient:",
          int((patients["disease"].value_counts() == 1).sum()))
    print("medicines:", medicines.shape, "| groups:", medicines["reason"].nunique())
    print(patients[["age", "blood_pressure", "cholesterol_level"]].describe().round(1))

    # most common diseases
    plt.figure(figsize=(8, 6))
    top = patients["disease"].value_counts().head(15)
    sns.barplot(x=top.values, y=top.index, color="#2a7f9e")
    plt.title("15 most common diagnoses")
    plt.xlabel("patients")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "top_diseases.png", dpi=110)
    plt.close()

    # risk level counts + age spread per risk level
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    order = ["Low", "Medium", "High"]
    colors = {"Low": "#4caf8a", "Medium": "#f0b429", "High": "#e05a5a"}
    sns.countplot(data=patients, x="risk_level", order=order,
                  palette=colors, ax=ax[0])
    ax[0].set_title("Patients per risk level")
    sns.boxplot(data=patients, x="risk_level", y="age", order=order,
                palette=colors, ax=ax[1])
    ax[1].set_title("Age by risk level")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "risk_levels.png", dpi=110)
    plt.close()

    # how often each symptom shows up, split by outcome
    rates = patients.groupby("outcome_variable")[SYMPTOMS].mean().T * 100
    rates.plot(kind="bar", figsize=(7, 4), color=["#e07a5f", "#3d85c6"])
    plt.title("Symptom rate by outcome (%)")
    plt.ylabel("% of patients")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "symptoms_by_outcome.png", dpi=110)
    plt.close()

    # correlations between the numeric columns
    plt.figure(figsize=(7, 5.5))
    corr = patients[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlations")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlations.png", dpi=110)
    plt.close()

    # medicine groups
    plt.figure(figsize=(8, 6))
    groups = medicines["reason"].value_counts().head(15)
    sns.barplot(x=groups.values, y=groups.index, color="#7b6fb0")
    plt.title("Biggest medicine groups in medicine.csv")
    plt.xlabel("drugs")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "medicine_groups.png", dpi=110)
    plt.close()
    print("EDA charts saved in", EDA_DIR)


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

def make_ml_model():
    """The machine learning model: a linear Support Vector Classifier.
    Features are scaled first (age is on a much bigger scale than the 0/1 columns).
    probability=True lets it give a probability for every disease."""
    return make_pipeline(StandardScaler(),
                         SVC(kernel="linear", C=1.0, probability=True, random_state=42))


class NeuralNet:
    """The deep learning model: a small fully connected neural network.

        8 inputs -> Dense 64 (ReLU) -> Dropout -> Dense 32 (ReLU) -> Dropout -> softmax

    It is built with TensorFlow / Keras. If TensorFlow is not installed, the same
    layer sizes are trained with scikit-learn's MLPClassifier instead, so the
    project still runs. self.backend says which one was used."""

    def __init__(self, epochs=30, seed=42):
        self.epochs = epochs
        self.seed = seed
        self.scaler = None
        self.classes_ = None
        self.net = None
        self.backend = None

    @staticmethod
    def _keras():
        try:
            from tensorflow import keras
            return keras
        except ImportError:
            return None

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.classes_ = np.array(sorted(set(y)))
        y_idx = np.searchsorted(self.classes_, np.asarray(y))

        keras = self._keras()
        if keras is not None:
            keras.utils.set_random_seed(self.seed)
            net = keras.Sequential([
                keras.layers.Input((Xs.shape[1],)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(len(self.classes_), activation="softmax"),
            ])
            net.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            net.fit(Xs, y_idx, epochs=self.epochs, batch_size=16, verbose=0)
            self.backend = "TensorFlow / Keras"
        else:
            net = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                random_state=self.seed).fit(Xs, y_idx)
            self.backend = "scikit-learn MLP"
        self.net = net
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype="float32"))
        if self.backend == "TensorFlow / Keras":
            return self.net.predict(Xs, verbose=0)
        out = np.zeros((len(Xs), len(self.classes_)))
        out[:, self.net.classes_] = self.net.predict_proba(Xs)
        return out

    def save(self, folder):
        meta = {"scaler": self.scaler, "classes": self.classes_, "backend": self.backend}
        if self.backend == "TensorFlow / Keras":
            self.net.save(Path(folder) / "dl_model.keras")
        else:
            meta["mlp"] = self.net
        joblib.dump(meta, Path(folder) / "dl_meta.joblib")

    @classmethod
    def load(cls, folder):
        meta = joblib.load(Path(folder) / "dl_meta.joblib")
        obj = cls()
        obj.scaler, obj.classes_, obj.backend = meta["scaler"], meta["classes"], meta["backend"]
        if obj.backend == "TensorFlow / Keras":
            from tensorflow import keras
            obj.net = keras.models.load_model(Path(folder) / "dl_model.keras")
        else:
            obj.net = meta["mlp"]
        return obj


def environment_signature():
    """Saved models only load safely with the same library versions that made them.
    app.py compares this with what is installed and retrains if it differs."""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = None
    return {"sklearn": sklearn.__version__, "tensorflow": tf_version}


def aligned_proba(model, X, classes):
    """Probabilities as a (rows x all diseases) table. In a CV fold a rare disease can be
    missing from the training part, so the columns are matched by name."""
    out = np.zeros((len(X), len(classes)))
    cols = np.searchsorted(classes, model.classes_)
    out[:, cols] = model.predict_proba(X)
    return out


def cross_validated_proba(make_model, X, y, classes, folds=5):
    """Out-of-fold probabilities: every row is predicted by a model that never saw it."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros((len(y), len(classes)))
    for train_idx, test_idx in kf.split(X):
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = aligned_proba(model, X.iloc[test_idx], classes)
    return oof


def top_k_scores(proba, y_idx):
    """Top-1 and top-3 accuracy.
    Top-3 matters because there are ~110 diseases and only 8 inputs, so 'is the real
    disease among the three best guesses' is the fairer question."""
    order = np.argsort(-proba, axis=1)
    top1 = float(np.mean(order[:, 0] == y_idx))
    top3 = float(np.mean([t in row[:3] for t, row in zip(y_idx, order)]))
    return top1, top3


def plot_model_scores(scores):
    names = list(scores)
    top1 = [scores[n][0] * 100 for n in names]
    top3 = [scores[n][1] * 100 if scores[n][1] is not None else 0 for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, top1, 0.4, label="Top-1", color="#2a7f9e")
    plt.bar(x + 0.2, top3, 0.4, label="Top-3", color="#9fd3c7")
    for i, (a, b) in enumerate(zip(top1, top3)):
        plt.text(i - 0.2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8)
        if b:
            plt.text(i + 0.2, b + 0.3, f"{b:.1f}", ha="center", fontsize=8)
    plt.xticks(x, names)
    plt.ylabel("% correct")
    plt.title("Disease prediction (5-fold CV, unique patients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "model_scores.png", dpi=110)
    plt.close()


def build_disease_profiles(patients):
    """For every disease: its usual risk level and how common each symptom is.
    Risk level in this dataset is almost fixed per disease, so the app looks
    it up here instead of training a separate model."""
    profiles = {}
    for disease, grp in patients.groupby("disease"):
        profiles[disease] = {
            "risk": grp["risk_level"].mode().iloc[0],
            "patients": int(len(grp)),
            "avg_age": round(float(grp["age"].mean()), 1),
            "symptom_rate": {s: round(float(grp[s].mean()), 2) for s in SYMPTOMS},
        }
    return profiles


def build_medicine_index(medicines):
    """TF-IDF over drug description + reason, used for the content-based
    ranking inside a medicine group."""
    text = (medicines["reason"] + " " + medicines["description"]).str.lower()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(text)
    return vectorizer, matrix


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    patients = load_patients()
    medicines = load_medicines()
    run_eda(patients, medicines)

    X = patients[FEATURES]
    y = patients["disease"]

    # Repeated rows in both the training and testing part would flatter the scores,
    # so the scores come from the de-duplicated table. The final models are fit on everything.
    unique = patients.drop(columns="patient_id").drop_duplicates().reset_index(drop=True)
    Xu, yu = unique[FEATURES], unique["disease"]
    classes = np.array(sorted(yu.unique()))
    y_idx = np.searchsorted(classes, yu.values)

    print("\n--- Testing the two final models (5-fold cross-validation) ---")
    oof_ml = cross_validated_proba(make_ml_model, Xu, yu, classes)
    oof_dl = cross_validated_proba(NeuralNet, Xu, yu, classes)
    ml_scores = top_k_scores(oof_ml, y_idx)
    dl_scores = top_k_scores(oof_dl, y_idx)
    both_scores = top_k_scores((oof_ml + oof_dl) / 2, y_idx)
    baseline = float(yu.value_counts(normalize=True).iloc[0])  # always guess the most common disease
    print(f"SVC (ML)            top-1 {ml_scores[0]:.3f}   top-3 {ml_scores[1]:.3f}")
    print(f"Neural net (DL)     top-1 {dl_scores[0]:.3f}   top-3 {dl_scores[1]:.3f}")
    print(f"ML + DL combined    top-1 {both_scores[0]:.3f}   top-3 {both_scores[1]:.3f}")
    print(f"Baseline (most common disease)  top-1 {baseline:.3f}")
    plot_model_scores({"ML (SVC)": ml_scores, "DL (neural net)": dl_scores,
                       "ML + DL": both_scores, "Guessing": (baseline, None)})

    print("\n--- Training the final models on all patients ---")
    ml_model = make_ml_model().fit(X, y)
    dl_model = NeuralNet().fit(X, y)
    print("Deep learning backend:", dl_model.backend)

    profiles = build_disease_profiles(patients)
    vectorizer, matrix = build_medicine_index(medicines)

    joblib.dump(ml_model, MODEL_DIR / "ml_model.joblib")
    dl_model.save(MODEL_DIR)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, MODEL_DIR / "medicine_index.joblib")
    medicines.to_csv(DATA_DIR / "medicine_clean.csv", index=False)
    patients.to_csv(DATA_DIR / "patients_clean.csv", index=False)

    metrics = {
        # app.py compares this with the installed libraries and retrains on a mismatch
        "environment": environment_signature(),
        "ml_name": "Support Vector Classifier (linear)",
        "dl_name": "Neural network (64-32)",
        "dl_backend": dl_model.backend,
        "ml_top1": ml_scores[0], "ml_top3": ml_scores[1],
        "dl_top1": dl_scores[0], "dl_top3": dl_scores[1],
        "combined_top1": both_scores[0], "combined_top3": both_scores[1],
        "baseline_top1": baseline,
        "n_patients": int(len(patients)),
        "n_unique_patients": int(len(unique)),
        "n_diseases": int(patients["disease"].nunique()),
        "n_medicines": int(len(medicines)),
        "profiles": profiles,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nDone. Models and metrics saved in", MODEL_DIR)


if __name__ == "__main__":
    main()
