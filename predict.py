import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import requests
import pandas as pd
import joblib
from dotenv import load_dotenv

# API設定
load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = { 'X-Auth-Token': API_KEY }

# 1. モデル定義（※train.pyと全く同じ構造にする）
class SoccerPredictor(nn.Module):
    def __init__(self, input_size):
        super(SoccerPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4) 
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4) 
        self.fc3 = nn.Linear(64, 32)    
        self.fc4 = nn.Linear(32, 3)     
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 直近の調子を取得する関数
def get_recent_points(team_id):
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=5"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return 7.0  
    
    matches = response.json().get('matches', [])
    points = 0
    for m in matches:
        is_home = m['homeTeam']['id'] == team_id
        winner = m['score']['winner']
        if winner == 'DRAW':
            points += 1
        elif (winner == 'HOME_TEAM' and is_home) or (winner == 'AWAY_TEAM' and not is_home):
            points += 3
    return float(points)

# 2. 予測の実行
def get_upcoming_matches():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    matches = response.json()['matches']
    
    team_df = pd.read_csv('team_ids.csv')
    team_to_id = dict(zip(team_df.TeamName, team_df.ID))
    
    # モデルとScalerの読み込み
    input_size = 6 
    model = SoccerPredictor(input_size)
    model.load_state_dict(torch.load('soccer_model.pth'))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    
    print("\n--- 今週末の試合予測結果 ---")
    
    for m in matches[:10]:
        home_name = m['homeTeam']['name']
        away_name = m['awayTeam']['name']
        
        try:
            home_id = team_to_id[home_name]
            away_id = team_to_id[away_name]
            
            # 生のデータを取得
            current_home_form = get_recent_points(home_id)
            current_away_form = get_recent_points(away_id)
            home_rest_days = 7 
            
            raw_data = [[
                home_id, 
                away_id, 
                current_home_form, 
                current_away_form, 
                home_rest_days, 
                0.0 
            ]]            
            
            # 学習時と同じScalerで変換
            scaled_data = scaler.transform(raw_data)
            input_data = torch.FloatTensor(scaled_data)
            
            # 予測
            with torch.no_grad():
                output = model(input_data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
            p_home = probabilities[0][0].item() * 100
            p_draw = probabilities[0][1].item() * 100
            p_away = probabilities[0][2].item() * 100
            
            print(f"{home_name} vs {away_name}")
            print(f"  [予測結果] ホーム勝: {p_home:.1f}% | 引き分け: {p_draw:.1f}% | アウェイ勝: {p_away:.1f}%")
            
        except KeyError:
            continue

if __name__ == "__main__":
    get_upcoming_matches()