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

@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=86400)
def get_upcoming_matches_api():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    
    data = response.json()
    # 試合データの中にエンブレムのURLが含まれているので、そのまま返せばOKです
    return data.get('matches', [])

@st.cache_data(ttl=86400)
def get_standings_api():
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return []
        
        data = response.json()
        table = data.get('standings', [{}])[0].get('table', [])
        
        # --- ここから修正：必要な情報を整理して抽出する ---
        all_data = []
        for item in table:
            all_data.append({
                "id": item['team']['id'],        # これが詳細表示に不可欠なIDです
                "順位": item['position'],
                "チーム": item['team']['name'],
                "勝ち点": item['points'],
                "試合数": item['playedGames'],
                "得失点": item['goalDifference']
            })
        return all_data
        # ----------------------------------------------
        
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return []

@st.cache_data(ttl=86400)
def get_top_scorers_api():
    # プレミアリーグ(PL)の得点者・アシスト者データを取得
    url = "https://api.football-data.org/v4/competitions/PL/scorers"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    # APIから選手データのリストを返す
    return response.json().get('scorers', [])

@st.cache_data(ttl=86400)
def get_team_form_api(team_id):
    # 特定のチームの直近5試合の完了済み試合を取得
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=5"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "-"
    
    matches = response.json().get('matches', [])
    form_list = []
    
    # 取得した試合を時系列（最新が右）に並べるために、古い順から処理
    for m in reversed(matches):
        is_home = m['homeTeam']['id'] == team_id
        winner = m['score']['winner']
        
        if winner == 'DRAW':
            form_list.append("D")
        elif (winner == 'HOME_TEAM' and is_home) or (winner == 'AWAY_TEAM' and not is_home):
            form_list.append("W")
        else:
            form_list.append("L")
            
    return " ".join(form_list) if form_list else "-"

@st.cache_data(ttl=3600)
def get_team_details_api(team_id):
    """
    指定したチームの詳細情報（監督・選手名簿）を取得します
    """
    url = f"https://api.football-data.org/v4/teams/{team_id}"
    try:
        # ファイル上部で定義済みの headers をそのまま使用
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

if not API_KEY:
    API_KEY = os.getenv('FOOTBALL_API_KEY')

# 追加：キーが取れているか確認
print(f"DEBUG: API_KEY is {'Found' if API_KEY else 'Not Found'}")

headers = { 'X-Auth-Token': API_KEY }