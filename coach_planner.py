import matplotlib.pyplot as plt
from ml_models import build_features

def get_latest_status(df):
    latest = df.iloc[-1]
    
    acwr = latest['ACWR']
    if acwr < 0.8:
        status = "Undertrained"
    elif 0.8 <= acwr <= 1.3:
        status = "Optimal Load"
    else:
        status = "High Injury Risk"
        
    return latest, status

def get_injury_probability(df, rf_model):
    X, y = build_features(df)
    latest_X = X.iloc[-1:]
    prob = rf_model.predict_proba(latest_X)[0][1]
    return prob

def display_coach_report(df, rf_model):
    latest, status = get_latest_status(df)
    risk_prob = get_injury_probability(df, rf_model)
    
    print(f"COACH'S DAILY REPORT: {latest['Player']}")
    print("-"*40)
    print(f"Current Status:     {status}")
    print(f"Injury Risk Score:  {risk_prob:.2%}")
    print(f"Fatigue Level:      {latest['Fatigue']:.2f}")
    print(f"ACWR:               {latest['ACWR']:.2f}")
    print("-" * 40)
    
    if risk_prob > 0.30 or status == "High Injury Risk":
        print("ACTION REQUIRED: High risk detected. Recommend rest or light recovery.")
    else:
        print("ACTION: Athlete is cleared for full participation.")
    print("-"*40 + "\n")

    recent = df.tail(31) 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(recent['Date'], recent['Acute'], label='Acute (7-day)', color='orange', linewidth=2)
    ax1.plot(recent['Date'], recent['Chronic'], label='Chronic (28-day)', color='blue', linestyle='--')
    ax1.set_title("Workload Balance (Short-term vs Long-term)")
    ax1.set_ylabel("Workload Units")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(recent['Date'], recent['ACWR'], color='black', marker='o', markersize=4)
    ax2.axhspan(0.8, 1.3, facecolor='green', alpha=0.2, label='Sweet Spot')
    ax2.axhline(y=1.5, color='red', linestyle='--', label='Danger Zone')
    ax2.set_title("Acute:Chronic Workload Ratio (ACWR)")
    ax2.set_ylabel("Ratio")
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()