from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset and train model
df = pd.read_csv("symptom_disease_named.csv")
df.dropna(subset=['text', 'disease_name'], inplace=True)

X = df["text"]
y = df["disease_name"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        symptom_text = request.form["symptom"]
        prediction = model.predict([symptom_text])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
