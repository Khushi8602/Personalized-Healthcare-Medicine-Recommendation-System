"""
app.py
------
Flask app for the Personalized Healthcare & Medicine Recommendation System.
No login: open the page, fill the form, get a recommendation.

Models are trained automatically on the first start (or run data_processing.py
yourself to see the EDA and scores), then:
    python app.py
and open http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, abort, render_template, request, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity

import data_processing
from data_processing import FEATURES, SYMPTOMS, reasons_for

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load everything once at start-up
# ---------------------------------------------------------------------------
def models_are_current():
    """Saved models only load safely with the library versions that made them.
    If they are missing or came from other versions, we retrain."""
    info = MODEL_DIR / "metrics.json"
    needed = ["ml_model.joblib", "dl_meta.joblib", "medicine_index.joblib"]
    if not all((MODEL_DIR / f).exists() for f in needed) or not info.exists():
        return False
    try:
        saved = json.loads(info.read_text()).get("environment")
        return saved == data_processing.environment_signature()
    except ValueError:
        return False


def load_models():
    return (joblib.load(MODEL_DIR / "ml_model.joblib"),
            data_processing.NeuralNet.load(MODEL_DIR),
            joblib.load(MODEL_DIR / "medicine_index.joblib"))


if not models_are_current():
    print("Training the models for the installed libraries (one time, about a minute)...")
    data_processing.main()

try:
    ml_model, dl_model, med_index = load_models()
except Exception as err:  # a saved file from another version can still fail to load
    print("Could not load the saved models (%s). Retraining..." % err)
    data_processing.main()
    ml_model, dl_model, med_index = load_models()

metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
profiles = metrics["profiles"]
DISEASES = ml_model.classes_  # both models use the same sorted disease list

medicines = pd.read_csv(DATA_DIR / "medicine_clean.csv").fillna("")
patients = pd.read_csv(DATA_DIR / "patients_clean.csv")
ALL_DISEASES = sorted(profiles.keys())

BP_LABELS = {0: "Low", 1: "Normal", 2: "High"}
CHOL_LABELS = {0: "Low", 1: "Normal", 2: "High"}
SYMPTOM_LABELS = {
    "fever": "Fever",
    "cough": "Cough",
    "fatigue": "Fatigue",
    "difficulty_breathing": "Difficulty breathing",
}

# ---------------------------------------------------------------------------
# Condition groups and general health advice
# ---------------------------------------------------------------------------
# First group whose keyword appears in the disease name wins.
GROUP_KEYWORDS = [
    ("cancer", ["cancer", "lymphoma", "melanoma", "tumor"]),
    ("mental", ["depression", "anxiety", "bipolar", "schizophrenia", "obsessive",
                "eating disorder", "autism", "tourette"]),
    ("neuro", ["migraine", "epilepsy", "parkinson", "alzheimer", "dementia",
               "multiple sclerosis", "cerebral palsy", "spina"]),
    ("cardio", ["hypertension", "hypertensive", "heart", "coronary", "stroke",
                "atherosclerosis", "myocardial"]),
    ("respiratory", ["asthma", "bronchitis", "pneumonia", "pulmonary", "copd",
                     "cold", "influenza", "tuberculosis", "sinusitis", "rhinitis",
                     "pneumothorax", "cystic", "sleep apnea"]),
    ("endocrine", ["diabetes", "thyroid", "glycemia", "polycystic"]),
    ("kidney", ["kidney", "urinary", "bladder"]),
    ("digestive", ["colitis", "crohn", "pancreatitis", "liver", "cirrhosis",
                   "cholecystitis", "diverticulitis", "hemorrhoid", "appendicitis",
                   "gastroenteritis", "hepatitis"]),
    ("musculoskeletal", ["arthritis", "osteoporosis", "gout", "scoliosis",
                         "fibromyalgia", "muscular", "osteomyelitis"]),
    ("skin", ["eczema", "psoriasis", "acne"]),
    ("infection", ["dengue", "malaria", "typhoid", "cholera", "zika", "ebola",
                   "rabies", "measles", "mumps", "rubella", "chickenpox", "hiv",
                   "tetanus", "polio", "lyme", "sepsis", "tonsillitis", "otitis",
                   "conjunctivitis", "pneumocystis"]),
    ("genetic", ["syndrome", "hemophilia", "sickle", "anemia"]),
]

ADVICE = {
    "respiratory": {
        "title": "Breathing and chest care",
        "diet": ["Drink warm fluids through the day; they loosen mucus.",
                 "Add vitamin C foods (citrus, guava, amla) and keep meals light."],
        "lifestyle": ["Stay away from smoke, dust and strong fumes.",
                      "Sleep with your head slightly raised if coughing worsens at night."],
        "monitor": ["Note how many days the cough or fever has lasted.",
                    "Count breaths at rest if you feel breathless."],
        "doctor": ["Breathing gets harder or lips look bluish.",
                   "Fever stays above 3 days or chest pain starts."],
    },
    "cardio": {
        "title": "Heart and blood pressure care",
        "diet": ["Cut down salt, pickles and packaged snacks.",
                 "Choose oats, vegetables, fruit and nuts over fried food."],
        "lifestyle": ["Walk 30 minutes on most days if your doctor agrees.",
                      "Quit smoking and keep alcohol to a minimum."],
        "monitor": ["Check blood pressure at home at the same time each day.",
                    "Keep a record to show your doctor."],
        "doctor": ["Chest pain, pressure or pain spreading to the arm or jaw.",
                   "Sudden weakness on one side, slurred speech or a severe headache."],
    },
    "endocrine": {
        "title": "Sugar and hormone balance",
        "diet": ["Pick whole grains, pulses and vegetables; avoid sweet drinks.",
                 "Eat at regular times and avoid long gaps between meals."],
        "lifestyle": ["Regular walking helps blood sugar and thyroid-related weight changes.",
                      "Keep sleep hours steady."],
        "monitor": ["Get fasting sugar or thyroid tests repeated as your doctor advises.",
                    "Check feet for cuts or sores if sugar is high."],
        "doctor": ["Extreme thirst, frequent urination or confusion.",
                   "Shaking, sweating or fainting spells."],
    },
    "infection": {
        "title": "Infection recovery",
        "diet": ["Drink plenty of boiled or filtered water and oral rehydration solution.",
                 "Eat simple, freshly cooked food like khichdi, dal and soups."],
        "lifestyle": ["Rest at home and avoid close contact with others while unwell.",
                      "Wash hands often and cover coughs."],
        "monitor": ["Write down your temperature twice a day.",
                    "Watch for rashes, bleeding gums or very dark urine."],
        "doctor": ["Fever with severe headache, stiff neck or confusion.",
                   "Fever lasts more than 3 days or you cannot keep fluids down."],
    },
    "digestive": {
        "title": "Gut and liver care",
        "diet": ["Eat small, frequent, low-oil meals; skip spicy and fried food.",
                 "Stay hydrated and avoid alcohol completely."],
        "lifestyle": ["Do not lie down right after eating.",
                      "Avoid painkillers like ibuprofen on an empty stomach."],
        "monitor": ["Watch stool colour and any yellowing of eyes or skin.",
                    "Note which foods trigger pain."],
        "doctor": ["Vomiting blood or black stools.",
                   "Severe belly pain that does not ease."],
    },
    "musculoskeletal": {
        "title": "Bone and joint care",
        "diet": ["Include milk, curd, ragi and leafy greens for calcium.",
                 "Get some morning sunlight for vitamin D."],
        "lifestyle": ["Do gentle stretching and low-impact exercise like walking or swimming.",
                      "Keep a healthy weight to reduce joint load."],
        "monitor": ["Track morning stiffness and which joints swell.",
                    "Ask about a bone density or uric acid test if due."],
        "doctor": ["A joint becomes hot, red and very swollen.",
                   "A fall with sudden pain or inability to bear weight."],
    },
    "neuro": {
        "title": "Brain and nerve care",
        "diet": ["Eat regular meals; skipping meals can trigger headaches.",
                 "Include nuts, fish or flaxseed and plenty of water."],
        "lifestyle": ["Keep a fixed sleep and wake time.",
                      "Keep a diary of triggers such as light, stress or certain foods."],
        "monitor": ["Record how often episodes happen and how long they last.",
                    "Note changes in memory, balance or movement."],
        "doctor": ["Sudden worst-ever headache or a seizure.",
                   "New numbness, vision loss or trouble speaking."],
    },
    "mental": {
        "title": "Mental wellbeing",
        "diet": ["Eat regular balanced meals and limit caffeine.",
                 "Avoid alcohol, which usually worsens mood and sleep."],
        "lifestyle": ["Try daily sunlight and a short walk.",
                      "Talk to a trusted person and keep a sleep routine."],
        "monitor": ["Notice changes in sleep, appetite and interest in daily activities.",
                    "Keep track of mood for a couple of weeks to show a professional."],
        "doctor": ["Thoughts of harming yourself. Contact a doctor or a local emergency or helpline right away.",
                   "Symptoms stop you from working or looking after yourself."],
    },
    "cancer": {
        "title": "Cancer care",
        "diet": ["Aim for a balanced diet with enough protein; ask for a dietitian's plan during treatment.",
                 "Avoid tobacco and alcohol."],
        "lifestyle": ["Gentle activity as tolerated helps energy and mood.",
                      "Keep all scan and follow-up appointments."],
        "monitor": ["Keep a list of symptoms, weight changes and side effects.",
                    "Keep all reports in one folder."],
        "doctor": ["This needs an oncologist. Book a specialist visit rather than waiting.",
                   "Unexplained weight loss, bleeding or a lump that is growing."],
    },
    "kidney": {
        "title": "Kidney and urinary care",
        "diet": ["Drink water regularly unless a doctor has limited fluids.",
                 "Reduce salt; ask about protein and potassium limits if kidney function is low."],
        "lifestyle": ["Do not hold urine for long.",
                      "Avoid regular painkillers like NSAIDs without advice."],
        "monitor": ["Watch for burning, blood in urine or swelling in feet.",
                    "Repeat kidney function tests as advised."],
        "doctor": ["Fever with back pain or chills.",
                   "Very little urine or sudden swelling."],
    },
    "skin": {
        "title": "Skin care",
        "diet": ["Drink enough water and limit very oily or sugary food if it flares your skin.",
                 "Note any food that triggers a flare."],
        "lifestyle": ["Use a gentle cleanser and moisturiser; avoid scratching.",
                      "Use sunscreen and avoid harsh soaps."],
        "monitor": ["Take photos weekly to show progress.",
                    "Watch for spreading redness or oozing."],
        "doctor": ["Skin becomes painful, hot or starts to ooze.",
                   "No improvement after a few weeks."],
    },
    "genetic": {
        "title": "Long-term condition support",
        "diet": ["Follow a balanced diet planned with a dietitian.",
                 "Stay hydrated and keep meals regular."],
        "lifestyle": ["Stay up to date with vaccines and routine checks.",
                      "Join a support group if one is available."],
        "monitor": ["Keep records of tests, growth and symptoms.",
                    "Follow the specialist's review schedule."],
        "doctor": ["Any sudden change in symptoms.",
                   "Unusual bleeding, severe pain or breathing trouble."],
    },
    "other": {
        "title": "General care",
        "diet": ["Eat balanced home-cooked meals and drink enough water.",
                 "Limit junk food and sugary drinks."],
        "lifestyle": ["Sleep 7-8 hours and stay active.",
                      "Avoid self-medicating with leftover prescriptions."],
        "monitor": ["Write down your symptoms and when they started.",
                    "Keep your reports together for the doctor."],
        "doctor": ["Symptoms get worse or new ones appear.",
                   "Pain, bleeding or fever that does not settle."],
    },
}


def condition_group(disease):
    name = disease.lower()
    for group, words in GROUP_KEYWORDS:
        if any(w in name for w in words):
            return group
    return "other"


# ---------------------------------------------------------------------------
# Prediction and recommendation logic
# ---------------------------------------------------------------------------

def make_row(data):
    """data: dict with the 8 feature values. Returns a one-row DataFrame."""
    return pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)


def disease_probabilities(row):
    """Probabilities from both models, and their average (the combined view)."""
    p_ml = ml_model.predict_proba(row)[0]
    p_dl = dl_model.predict_proba(row)[0]
    return p_ml, p_dl, (p_ml + p_dl) / 2


def predict_diseases(row, top_n=5):
    """Top guesses ranked by the combined (ML + DL) probability, plus what each model
    gave on its own and each model's single best pick."""
    p_ml, p_dl, p_all = disease_probabilities(row)
    order = np.argsort(p_all)[::-1][:top_n]
    guesses = [{"name": DISEASES[i],
                "score": round(float(p_all[i]) * 100, 1),
                "ml": round(float(p_ml[i]) * 100, 1),
                "dl": round(float(p_dl[i]) * 100, 1),
                "risk": profiles[DISEASES[i]]["risk"]} for i in order]
    picks = {"ml": DISEASES[int(np.argmax(p_ml))], "dl": DISEASES[int(np.argmax(p_dl))]}
    return guesses, picks


def expected_risk(row):
    """Used when no diagnosis is given: weigh each disease's usual risk level
    by how likely the models think that disease is."""
    _, _, proba = disease_probabilities(row)
    totals = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    for p, name in zip(proba, DISEASES):
        totals[profiles[name]["risk"]] += p
    return max(totals, key=totals.get)


def recommend_medicines(disease, symptoms, per_group=3):
    """Pick the medicine groups that fit the disease, then rank drugs inside
    those groups by TF-IDF similarity between the patient's text and the drug
    description. Returns (list, note)."""
    reasons = reasons_for(disease)
    note = ""
    if not reasons:
        if symptoms.get("fever"):
            reasons = ["Fever"]
            note = ("medicine.csv has no drug group for this condition, so these are "
                    "fever-relief options only. The condition itself needs a doctor.")
        else:
            return [], ("medicine.csv has no drug group for this condition. "
                        "Please see a doctor or specialist for treatment.")

    query = disease + " " + " ".join(
        SYMPTOM_LABELS[s].lower() for s in SYMPTOMS if symptoms.get(s)) + " " + " ".join(reasons)
    q_vec = med_index["vectorizer"].transform([query.lower()])

    results = []
    for reason in reasons:
        idx = medicines.index[medicines["reason"] == reason].to_numpy()
        if len(idx) == 0:
            continue
        sims = cosine_similarity(q_vec, med_index["matrix"][idx]).ravel()
        seen = set()
        taken = 0
        for pos in np.argsort(-sims, kind="stable"):
            row = medicines.iloc[idx[pos]]
            key = row["drug_name"].lower()
            if key in seen:
                continue
            seen.add(key)
            desc = row["description"]
            results.append({
                "name": row["drug_name"],
                "group": reason,
                "use": desc[:140] + ("..." if len(desc) > 140 else "") if desc else "No description available",
                "match": round(float(sims[pos]) * 100),
            })
            taken += 1
            if taken >= per_group:
                break
    return results, note


def vitals_flags(inputs):
    flags = []
    if inputs["blood_pressure"] == 2:
        flags.append("Blood pressure is marked high. Limit salt and get it rechecked this week.")
    if inputs["blood_pressure"] == 0:
        flags.append("Blood pressure is marked low. Rise slowly from sitting and drink enough fluids.")
    if inputs["cholesterol_level"] == 2:
        flags.append("Cholesterol is marked high. Cut fried and fatty food and ask for a lipid profile.")
    if inputs["age"] >= 60:
        flags.append("At 60 or above, check with a doctor before starting any new medicine, since doses often change with age.")
    if inputs["fever"] and inputs["difficulty_breathing"]:
        flags.append("Fever together with difficulty breathing should be seen by a doctor the same day.")
    return flags


def build_result(inputs, diagnosis=None, source="symptoms"):
    """Full recommendation for one patient.
    diagnosis: a known disease name (patient records or the form's dropdown).
    Without it we take the model's top guess."""
    row = make_row(inputs)
    guesses, picks = predict_diseases(row)

    if diagnosis:
        disease, basis = diagnosis, "diagnosis"
        risk = profiles.get(diagnosis, {}).get("risk") or expected_risk(row)
    else:
        disease, basis = guesses[0]["name"], "model guess"
        risk = expected_risk(row)

    group = condition_group(disease)
    meds, med_note = recommend_medicines(disease, inputs)
    advice = ADVICE[group]

    if risk == "High":
        urgency = "High risk: book a doctor visit soon instead of managing this at home."
    elif risk == "Medium":
        urgency = "Medium risk: see a doctor in the next few days if symptoms continue."
    else:
        urgency = "Lower risk: home care is usually enough, but see a doctor if it gets worse."

    return {
        "inputs": inputs,
        "disease": disease,
        "basis": basis,
        "source": source,
        "guesses": guesses,
        "picks": picks,
        "risk": risk,
        "urgency": urgency,
        "group": group,
        "advice": advice,
        "medicines": meds,
        "med_note": med_note,
        "flags": vitals_flags(inputs),
        "symptom_text": [SYMPTOM_LABELS[s] for s in SYMPTOMS if inputs[s]] or ["None reported"],
        "bp_label": BP_LABELS[inputs["blood_pressure"]],
        "chol_label": CHOL_LABELS[inputs["cholesterol_level"]],
    }


def read_form(form):
    """Turn the submitted form into the 8 numbers the models expect.
    Returns (inputs, error)."""
    try:
        age = int(form.get("age", ""))
    except ValueError:
        return None, "Please enter your age as a whole number."
    if not 1 <= age <= 120:
        return None, "Age should be between 1 and 120."

    gender = form.get("gender", "")
    if gender not in ("male", "female"):
        return None, "Please choose a gender."
    try:
        bp = int(form.get("blood_pressure", "1"))
        chol = int(form.get("cholesterol_level", "1"))
    except ValueError:
        return None, "Blood pressure and cholesterol need a level."
    if bp not in (0, 1, 2) or chol not in (0, 1, 2):
        return None, "Blood pressure and cholesterol must be Low, Normal or High."

    inputs = {s: 1 if form.get(s) == "on" else 0 for s in SYMPTOMS}
    inputs.update({"age": age, "gender_male": 1 if gender == "male" else 0,
                   "blood_pressure": bp, "cholesterol_level": chol})
    return inputs, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.context_processor
def inject_metrics():
    return {"m": metrics}


@app.route("/")
def home():
    return render_template("index.html", diseases=ALL_DISEASES, form={}, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    inputs, error = read_form(request.form)
    if error:
        return render_template("index.html", diseases=ALL_DISEASES,
                               form=request.form, error=error), 400
    known = request.form.get("known_disease", "").strip()
    if known and known not in profiles:
        known = ""
    result = build_result(inputs, diagnosis=known or None)
    return render_template("result.html", r=result, patient=None)


@app.route("/patients")
def patient_list():
    rows = patients.to_dict("records")
    for r in rows:
        r["gender"] = "Male" if r["gender_male"] else "Female"
    return render_template("patients.html", rows=rows)


@app.route("/patient/<int:pid>")
def patient_detail(pid):
    match = patients[patients["patient_id"] == pid]
    if match.empty:
        abort(404)
    rec = match.iloc[0]
    inputs = {f: int(rec[f]) for f in FEATURES}
    result = build_result(inputs, diagnosis=rec["disease"], source="record")
    return render_template("result.html", r=result,
                           patient={"id": pid, "outcome": rec["outcome_variable"],
                                    "risk": rec["risk_level"]})


@app.route("/insights")
def insights():
    return render_template("insights.html", m=metrics)


@app.route("/eda/<path:filename>")
def eda_file(filename):
    return send_from_directory(DATA_DIR / "eda", filename)


@app.errorhandler(404)
def not_found(_):
    return render_template("index.html", diseases=ALL_DISEASES, form={},
                           error="That page or patient does not exist."), 404


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # reloader off: it would load TensorFlow twice
