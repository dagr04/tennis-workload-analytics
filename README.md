# Tennis Players Workload & Injury Risk Prediction System

This project is a sports analytics and machine learning system designed to simulate athlete training data, engineer sports science metrics, and provide actionable injury risk insights through a coach-facing dashboard.

## Project Overview
This system bridges the gap between raw training data and clinical decision support. It mimics the workflows used by professional sports organizations to monitor athlete health and optimize performance.

* Synthetic Data Generation: Simulates stochastic athlete training workloads and latent injury patterns using semi-randomized intensity variables.
* Feature Engineering: Calculates longitudinal sports science metrics including Acute:Chronic Workload Ratio (ACWR), fatigue accumulation, and high-intensity streaks.
* Algorithmic Benchmarking: Implements and compares a linear baseline (Logistic Regression) against a non-linear ensemble method (Random Forest).
* Decision Engine: Deploys a dashboard that maps model probabilities and workload ratios to specific clinical recommendations.

## System Components

### 1. Workload Simulation Engine
The WorkloadGenerator creates a synthetic dataset by simulating:
* Session Types: Practice, gym, and match sessions.
* Metrics: Intensity, fatigue accumulation, and chronic/acute workloads.
* ACWR: The Acute:Chronic Workload Ratio, a gold-standard metric in sports science.
* Injury Events: Realistic injury triggers based on overtraining thresholds.

### 2. Machine Learning Pipeline
The system evaluates two distinct approaches:
* Logistic Regression: A baseline medical-style risk model.
* Random Forest: A nonlinear ensemble model for complex pattern recognition.

Performance is measured via: ROC-AUC scores, Classification Reports, and ROC Curve visualizations.

### 3. Coach Decision Dashboard
The final output is a decision-support tool that translates data into coaching actions:
* Load Status: Categorizes athletes (Undertrained / Optimal / Overloaded).
* Risk Score: Probability percentage of injury.
* Recommendations: Clear instructions (e.g., "Recommend light recovery").
* Visuals: ACWR timelines with "Safe" and "Danger" zones.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Compare Models 
```bash
python3 main_ml_models.py
```

### 3. Run the System
```bash
python3 main_coach_planner.py
```

## Insights & Outputs

* Visualizations: The system generates ROC curves for data scientists and workload trend graphs for coaches.
* Actionable Data: Instead of just "High Risk," the system provides specific intervention advice based on the athlete's fatigue and workload history.

## Author 

Dev Agrawal
Pre-Engineering
Earlham College

## Future Improvements
* Integration of real-world athlete datasets (GPS/Wearable data).
* Implementation of Gradient Boosting (XGBoost).
* Seasonal planning optimizer to peak for specific competition dates.