import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

MODEL_PATH = "naive_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Target Audience Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center font-sans antialiased text-slate-800">
    <div class="w-full max-w-lg m-4 bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <div class="bg-gradient-to-r from-violet-600 to-indigo-600 px-8 py-6 text-white">
            <h1 class="text-2xl font-bold tracking-tight">Predictive Analytics</h1>
            <p class="text-indigo-100 text-sm mt-1">Gaussian Naive Bayes Classification Model</p>
        </div>
        <form method="POST" action="/" class="p-8 space-y-6">
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-2" for="gender">Gender</label>
                <select id="gender" name="gender" required class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 bg-slate-50">
                    <option value="" disabled selected>Select Gender</option>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-2" for="age">Age</label>
                <input type="number" id="age" name="age" min="0" max="120" placeholder="e.g. 35" required class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 bg-slate-50">
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-2" for="salary">Estimated Annual Salary ($)</label>
                <input type="number" id="salary" name="salary" min="0" placeholder="e.g. 50000" required class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 bg-slate-50">
            </div>
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-xl shadow-md transition-all">
                Run Classification
            </button>
        </form>
        {% if prediction is not none %}
        <div class="border-t border-slate-100 bg-slate-50 p-8 text-center">
            <h3 class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1">Model Output</h3>
            {% if prediction == 1 %}
            <div class="inline-flex items-center px-4 py-2 rounded-full bg-emerald-100 text-emerald-800 font-semibold text-lg mb-2">🎯 Positive Class (1)</div>
            {% else %}
            <div class="inline-flex items-center px-4 py-2 rounded-full bg-rose-100 text-rose-800 font-semibold text-lg mb-2">🛑 Negative Class (0)</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            gender = float(request.form.get("gender"))
            age = float(request.form.get("age"))
            salary = float(request.form.get("salary"))
            features = np.array([[gender, age, salary]])
            if model:
                prediction = int(model.predict(features)[0])
            else:
                prediction = "Model not loaded."
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
