import numpy as np
import pandas as pd

class WorkloadGenerator:
    def __init__(self, player_name, start_date, days, seed = 35):
        np.random.seed(seed)
        self.player = player_name
        self.dates = pd.date_range(start = start_date, periods = days)
        self.rest_day = False
        self.rows = []
        self.forced_rest_days = 0
        self.prone_to_injury = np.random.uniform(0.0, 0.15)
        self.fatigue = 0

    def generate_day(self):
        if self.rest_day or self.forced_rest_days > 0:
            practice_h = 0
            gym_h = 0
            match_h = 0
            intensity = 1
            workload = 0

            if self.forced_rest_days > 0:
                self.forced_rest_days -= 1

            self.rest_day = False
    
        else:
            practice_h = np.random.randint(0, 3)
            gym_h = np.random.randint(0, 3)
    
            if practice_h <= 1 and gym_h <= 1:
                match_h = np.random.randint(2, 4)
            else:
                match_h = 0
    
            total_h = practice_h + gym_h + match_h
    
            if total_h > 0:
                intensity = np.random.randint(4, 10)
            else:
                intensity = np.random.randint(1, 3)
    
            workload = total_h * intensity
    
            if match_h > 0:
                self.rest_day = True
    
        return practice_h, gym_h, match_h, intensity, workload


    def run(self):
        for date in self.dates:
            practice_h, gym_h, match_h, intensity, workload = self.generate_day()

            self.fatigue = self.fatigue * 0.9 + workload * 0.1

            self.rows.append({
                "Player": self.player,
                "Date": date,
                "Practice": practice_h,
                "Gym": gym_h,
                "Match": match_h,
                "Intensity": intensity,
                "Workload": workload,
                "Fatigue": self.fatigue
            })
            
        df = pd.DataFrame(self.rows)
        
        df['Acute'] = df['Workload'].rolling(window = 7, min_periods = 1).mean()
        df['Chronic'] = df['Workload'].rolling(window = 28, min_periods = 1).mean()
        df['ACWR'] = df['Acute'] / (df['Chronic'] + 1e-6)

        df["HighIntensityStreak"] = (df["Intensity"] >= 8).rolling(3, min_periods = 1).sum()

        injuries = []

        for i, row in df.iterrows():
            risk = 0.01 + self.prone_to_injury + np.random.normal(0, 0.02)

            if row["ACWR"] > 1.3:
                risk += 0.05
            if row["ACWR"] > 1.5:
                risk += 0.10
            if row["ACWR"] > 1.8:
                risk += 0.15

            if row["Fatigue"] > df["Fatigue"].quantile(0.75):
                risk += 0.10

            if row["Intensity"] >= 8:
                risk += 0.05

            if row["HighIntensityStreak"] >= 3:
                risk += 0.10

            risk = np.clip(risk, 0, 0.7)

            injury = np.random.rand() < risk
            injuries.append(int(injury))

            if injury:
                self.forced_rest_days = np.random.randint(7, 14)

        df["Injury"] = injuries

        return df