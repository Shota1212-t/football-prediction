import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 重複エラー回避

import torch
import torch.nn as nn
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# --- 1. モデルの定義 (train.pyと同じ構造である必要があります) ---
class SoccerPredictor(nn.Module):
    def __init__(self, input_size):
        super(SoccerPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. 予測用データの準備 ---
load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = { 'X-Auth-Token': API_KEY }

def get_upcoming_matches():
    # 今後の試合スケジュールを取得
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    matches = response.json()['matches']
    
    # チームIDの対応表を読み込む
    team_df = pd.read_csv('team_ids.csv')
    team_to_id = dict(zip(team_df.TeamName, team_df.ID))
    
    # モデルのロード
    input_size = 6 # 特徴量の数
    model = SoccerPredictor(input_size)
    model.load_state_dict(torch.load('soccer_model.pth'))
    model.eval()
    
    print("\n--- 今週末の試合予測結果 ---")
    
    for m in matches[:10]: # 直近10試合を表示
        home_name = m['homeTeam']['name']
        away_name = m['awayTeam']['name']
        
        # 学習時と同じ特徴量を作る（簡易版: IDと、とりあえず平均的なFormを入れる）
        # ※本来は直近データを再計算して入れるのがベストですが、まずは動かします
        try:
            home_id = team_to_id[home_name]
            away_id = team_to_id[away_name]
            
            # 入力データ作成 (home_id, away_id, home_form, away_form, home_rest, h2h)
            # 仮の数値として、平均的な状態(Form:7, Rest:7, H2H:0)を代入
            input_data = torch.FloatTensor([[home_id, away_id, 7.0, 7.0, 7.0, 0.0]])
            
            # 予測
            with torch.no_grad():
                output = model(input_data)
                # Softmaxで確率に変換
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
            p_home = probabilities[0][0].item() * 100
            p_draw = probabilities[0][1].item() * 100
            p_away = probabilities[0][2].item() * 100
            
            print(f"{home_name} vs {away_name}")
            print(f"  [予測結果] ホーム勝: {p_home:.1f}% | 引き分け: {p_draw:.1f}% | アウェイ勝: {p_away:.1f}%")
            
        except KeyError:
            # 昇格組などでID表にないチームはスキップ
            continue

if __name__ == "__main__":
    get_upcoming_matches()