from flask import Flask, render_template, request
import numpy as np
import pickle
import sqlite3

app = Flask(__name__)

# Load trained model
model = pickle.load(open('Liver2.pkl', 'rb'))

# Initialize database
def init_db():
    conn = sqlite3.connect("liver_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Age REAL,
        ALB REAL,
        ALP REAL,
        ALT REAL,
        AST REAL,
        BIL REAL,
        CHE REAL,
        CHOL REAL,
        CREA REAL,
        GGT REAL,
        PROT REAL,
        Sex_m INTEGER,
        Prediction TEXT
    )
    """)
    conn.commit()
    conn.close()

@app.route('/')
def home():
    conn = sqlite3.connect("liver_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    conn.close()
    return render_template('home.html', rows=rows)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Age = float(request.form['Age'])
        ALB = float(request.form['ALB'])
        ALP = float(request.form['ALP'])
        ALT = float(request.form['ALT'])
        AST = float(request.form['AST'])
        BIL = float(request.form['BIL'])
        CHE = float(request.form['CHE'])
        CHOL = float(request.form['CHOL'])
        CREA = float(request.form['CREA'])
        GGT = float(request.form['GGT'])
        PROT = float(request.form['PROT'])
        Sex_m = int(request.form['Sex_m'])

        # Create feature array
        features = np.array([[Age, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT, Sex_m]])
        
        # Predict
        prediction = model.predict(features)[0]

        # Convert prediction to label
        label_map = {
            0: "Blood Donor",
            1: "Suspect Blood Donor",
            2: "Hepatitis",
            3: "Fibrosis",
            4: "Cirrhosis"
        }
        prediction_label = label_map[prediction]

        # Store data in database
        conn = sqlite3.connect("liver_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (Age, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT, Sex_m, Prediction) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (Age, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT, Sex_m, prediction_label))
        conn.commit()
        conn.close()

        # Redirect to different pages based on prediction
        if prediction == 0:
            return render_template('blood_donor.html')
        elif prediction == 1:
            return render_template('suspect_blood_donor.html')
        elif prediction == 2:
            return render_template('hepatitis.html')
        elif prediction == 3:
            return render_template('fibrosis.html')
        elif prediction == 4:
            return render_template('cirrhosis.html')

    return render_template('predict.html')

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
