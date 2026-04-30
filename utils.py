import os
import streamlit as st
import requests
from dotenv import load_dotenv # 追加

# プロジェクト内の .env ファイルを読み込む
load_dotenv()

# APIキーの取得
API_KEY = None

try:
    # 1. 本番環境（Streamlit Cloud）用
    if hasattr(st, "secrets") and "FOOTBALL_API_KEY" in st.secrets:
        API_KEY = st.secrets["FOOTBALL_API_KEY"]
except Exception:
    pass

# 2. ローカル環境用（もしSecretsで取れなかったら）
if not API_KEY:
    API_KEY = os.getenv('FOOTBALL_API_KEY')

# デバッグ用（確認したら消してOK）
print(f"DEBUG: API_KEY is {'Found' if API_KEY else 'Not Found'}")

headers = { 'X-Auth-Token': API_KEY }

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

def get_upcoming_matches_api():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    
    data = response.json()
    # 試合データの中にエンブレムのURLが含まれているので、そのまま返せばOKです
    return data.get('matches', [])



# ...（中略）
if not API_KEY:
    API_KEY = os.getenv('FOOTBALL_API_KEY')

# 追加：キーが取れているか確認
print(f"DEBUG: API_KEY is {'Found' if API_KEY else 'Not Found'}")

headers = { 'X-Auth-Token': API_KEY }