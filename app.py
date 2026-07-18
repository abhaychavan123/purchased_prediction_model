import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Naive Bayes model
MODEL_PATH = "naive_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found. Please upload it to the root directory.")

# Attractive HTML layout using Tailwind CSS for UI styling
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

    <div class="w-full max-w-lg m-4 bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden transition-all duration-300 hover:shadow-2xl">
        
        <!-- Header -->
        <div class="bg-gradient-to-r from-violet-600 to-indigo-600 px-8 py-6 text-white">
            <h1 class="text-2xl font-bold tracking-tight">Predictive Analytics</h1>
            <p class="text-indigo-100 text-sm mt-1">Gaussian Naive Bayes Classification Model</p>
        </div>

        <!-- Form Content -->
        <form method="POST" action="/" class="p-8 space-y-6">
            
            <!-- Gender Input -->
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-2" for="gender">Gender</label>
                <select id="gender" name="gender" required 
                        class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-slate-50 text-slate-700 transition-colors">
                    <option value="" disabled selected>Select Gender</option>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <!-- Age Input -->
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-2" for="age">Age</label>
                <input type="number" id="age" name="age" min="0" max="120" placeholder="e.g. 35" required
                       class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-slate-50 text-slate-700 transition-colors">
            </div>

            <!-- Estimated Salary Input -->
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-2" for="salary">Estimated Annual Salary ($)</label>
                <input type="number" id="salary" name="salary" min="0" placeholder="e.g. 50000" required
                       class="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-slate-50 text-slate-700 transition-colors">
            </div>

            <!-- Submit Button -->
            <button type="submit" 
                    class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-xl shadow-md shadow-indigo-200 transition-all duration-150 transform active:scale-[0.98]">
                Run Classification
            </button>
        </form>

        <!-- Prediction Result Display -->
        {% if prediction is not none %}
        <div class="border-t border-slate-100 bg-slate-50 p-8 text-center animate-fade-in">
            <h3 class="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1">Model Output</h3>
            
            {% if prediction == 1 %}
            <div class="inline-flex items-center px-4 py-2 rounded-full bg-emerald-100 text-emerald-800 font-semibold text-lg mb-2">
                🎯 Positive Class (1)
            </div>
            <p class="text-sm text-slate-500">The customer fits the targeted profile criteria.</p>
            {% else %}
            <div class="inline-flex items-center px-4 py-2 rounded-full bg-rose-100 text-rose-800 font-semibold text-lg mb-2">
                🛑 Negative Class (0)
            </div>
            <p class="text-sm text-slate-500">The customer does not meet the target criteria.</p>
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
            # Safely grab metrics from form data
            gender = float(request.form.get("gender"))
            age = float(request.form.get("age"))
            salary = float(request.form.get("salary"))
            
            # Format features identically to feature_names_in_: ['Gender', 'Age', 'EstimatedSalary']
            features = np.array([[gender, age, salary]])
            
            if model:
                # Execute prediction (returns 0 or 1)
                prediction = int(model.predict(features)[0])
            else:
                prediction = "Error: Model not loaded."
        except Exception as e:
            prediction = f"Error processing input: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
