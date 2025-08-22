import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
import joblib as j
import os
from flask import Flask, render_template, request, Response
from datetime import datetime
from transformers import pipeline

app = Flask(__name__)

ADMIN_USERNAME = "host"
ADMIN_PASSWORD = "admin123"

ai_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

solution_generator = pipeline(
    "text-generation",
    model="gpt2",
    do_sample=True,
    max_length=120
)

pt = PorterStemmer()

def preprocessing(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return " ".join([pt.stem(word) for word in text.split()])

def map_ai_to_status(label):
    mapping = {
        "joy": "Normal",
        "sadness": "Depression",
        "fear": "Anxiety",
        "anger": "Stress",
        "surprise": "Bipolar",
        "love": "Personality Disorder",
        "disgust": "Suicidal"
    }
    return mapping.get(label, "Normal")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        text = request.form['text']

        text_proc = preprocessing(text)

        if len(text_proc.strip()) < 3 or len(text_proc.split()) < 2:
            return render_template("predict.html", error="Please enter a meaningful sentence.")

        ai_results = ai_classifier(text_proc)[0]  # list of dicts with scores
        best = max(ai_results, key=lambda x: x['score'])
        result = map_ai_to_status(best['label'])

        prompt = (
            f"You are a helpful mental health assistant. "
            f"The user wrote: '{text}'. "
            f"The predicted mental health condition is: '{result}'. "
            f"Provide a clear, practical solution or guidance for this condition."
        )
        solution_text = solution_generator(prompt, max_length=120)[0]['generated_text']

        if not os.path.exists("user_data"):
            os.makedirs("user_data")
        filepath = "user_data/predictions.csv"
        new_entry = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "age": age,
            "text": text,
            "prediction": result,
            "solution": solution_text
        }])
        if os.path.exists(filepath):
            new_entry.to_csv(filepath, mode='a', header=False, index=False)
        else:
            new_entry.to_csv(filepath, index=False)

        return render_template(
            'success.html',
            name=name,
            age=age,
            text=text,
            prediction=result,
            solution=solution_text
        )

    return render_template("predict.html")


@app.route("/results", methods=["POST"])
def results():
    name = request.form.get("name")
    age = request.form.get("age")
    original = request.form.get("original")
    prediction = request.form.get("prediction")
    solution = request.form.get("solution")

    filepath = "user_data/predictions.csv"
    predictions_list = {}
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, on_bad_lines="skip")
        predictions_list = df['prediction'].value_counts().to_dict()

    return render_template(
        "results.html",
        name=name,
        age=age,
        original=original,
        prediction=prediction,
        solution=solution,
        predictions=predictions_list
    )


@app.route('/analysis')
def analysis():
    auth = request.authorization
    if not auth or not (auth.username == ADMIN_USERNAME and auth.password == ADMIN_PASSWORD):
        return Response(
            'Access Denied: Admins only.',
            401,
            {'WWW-Authenticate': 'Basic realm="Login Required"', 'Cache-Control': 'no-store'}
        )

    filepath = "user_data/predictions.csv"
    if not os.path.exists(filepath):
        return "<h2>No user data available yet. Please enter some predictions first.</h2>"

    df = pd.read_csv(filepath, on_bad_lines='skip')

    if not os.path.exists("static/charts"):
        os.makedirs("static/charts")

        plt.figure(figsize=(6,6))
    df['prediction'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True)
    plt.title("Prediction Distribution")
    plt.ylabel("")
    plt.savefig("static/charts/pie.png")
    plt.close()

    plt.figure(figsize=(8,5))
    sn.countplot(x="prediction", data=df)
    plt.title("Number of Users by Prediction")
    plt.xticks(rotation=45)
    plt.savefig("static/charts/bar.png")
    plt.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timeline = df.groupby(df['timestamp'].dt.date)['prediction'].count()
    plt.figure(figsize=(10,5))
    timeline.plot(kind='line', marker='o')
    plt.title("Predictions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Predictions")
    plt.savefig("static/charts/line.png")
    plt.close()

    return render_template("analysis.html", tables=df.to_dict(orient="records"))


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact-support')
def contact_support():
    return render_template("contact_support.html")


if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__" and False:
    pass

plt.figure(figsize=(10,5))
sn.countplot(df['status'])
plt.xlabel("Status of Patients")
plt.ylabel("No. of Patients")
plt.title("Patients With Respective Status")
plt.xticks(rotation=45)
plt.show()

status_counts = df['status'].value_counts()
colors = ['#87CEEB', '#2D3047', '#FF7F50', '#419D78', '#B3CDE0', '#D0D0D0', '#90EE90']
plt.figure(figsize=(7, 7))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, shadow=True)
plt.title('Mental Health Conditions')
plt.axis('equal')
plt.tight_layout()
plt.show()

def resmpl(df):
    max_count = df['status'].value_counts().max()
    df_resampled = pd.DataFrame()
    for status in df['status'].unique():
        df_class = df[df['status'] == status]
        if len(df_class) < max_count:
            df_class_resampled = resample(df_class, replace=True, n_samples=max_count, random_state=42)
            df_resampled = pd.concat([df_resampled, df_class_resampled])
        else:
            df_resampled = pd.concat([df_resampled, df_class])
    return df_resampled

df = resmpl(df)

pt = PorterStemmer()
def preprocessing(x):
    l = []
    text = re.sub(r'[^a-zA-Z0-9\s]', '', x.lower())
    for i in text.split():
        l.append(pt.stem(i.lower()))
    return " ".join(l)

df['statement'] = df['statement'].apply(preprocessing)

x = df['statement']
y = df['status']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

vec = TfidfVectorizer()
x_train_tfidf = vec.fit_transform(x_train)
x_test_tfidf = vec.transform(x_test)
print(x_train_tfidf.shape, x_test_tfidf.shape)

rf = RandomForestClassifier()
rf.fit(x_train_tfidf, y_train) 
ypred = rf.predict(x_test_tfidf)
ytr_pred = rf.predict(x_train_tfidf)
print("Test accuracy: ", accuracy_score(y_test, ypred)) 
print("Train accuracy: ", accuracy_score(y_train, ytr_pred))
print("Precision Test accuracy: ", precision_score(y_test, ypred, average='weighted'))
print("Precision Train accuracy: ", precision_score(y_train, ytr_pred, average='weighted'))

adb = AdaBoostClassifier() 
adb.fit(x_train_tfidf, y_train) 
ypred = adb.predict(x_test_tfidf)
ytr_pred = adb.predict(x_train_tfidf)
print("Test accuracy: ", accuracy_score(y_test, ypred))
print("Train accuracy: ", accuracy_score(y_train, ytr_pred))
print("Precision Test accuracy: ", precision_score(y_test, ypred, average='weighted'))
print("Precision Train accuracy: ", precision_score(y_train, ytr_pred, average='weighted'))

ext = ExtraTreesClassifier() 
ext.fit(x_train_tfidf, y_train)
ypred = ext.predict(x_test_tfidf)
ytr_pred = ext.predict(x_train_tfidf)
print("Test accuracy: ", accuracy_score(y_test, ypred))
print("Train accuracy: ", accuracy_score(y_train, ytr_pred))
print("Precision Test accuracy: ", precision_score(y_test, ypred, average='weighted'))
print("Precision Train accuracy: ", precision_score(y_train, ytr_pred, average='weighted'))

lr = LogisticRegression()
lr.fit(x_train_tfidf, y_train)
ypred = lr.predict(x_test_tfidf)
ytr_pred = lr.predict(x_train_tfidf)
print("Test accuracy: ", accuracy_score(y_test, ypred))
print("Train accuracy: ", accuracy_score(y_train, ytr_pred))
print("Precision Test accuracy: ", precision_score(y_test, ypred, average='weighted'))
print("Precision Train accuracy: ", precision_score(y_train, ytr_pred, average='weighted'))

j.dump(vec, 'vectorizer.pkl')
j.dump(rf, 'model.pkl')
vector = j.load('vectorizer.pkl')
model = j.load('model.pkl')

pt = PorterStemmer()
def preprocessing(x):
    l = []
    text = re.sub(r'[^a-zA-Z0-9\s]', '', x.lower())   
    for i in text.split():
        l.append(pt.stem(i.lower()))
    return " ".join(l) 

text = "trouble sleeping, confused mind, restless heart. All out of tune"
text = preprocessing(text)

def predi(x):
    vec = vector.transform([text])
    result = model.predict(vec)[0]
    return result

predi(text)
text = 'It shows I have posted but I cannot find my post back here Am I not allowed to post here?'
text = preprocessing(text)
predi(text)
