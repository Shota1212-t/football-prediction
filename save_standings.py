# save_standings.py (ローカル実行専用)
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = { 'X-Auth-Token': API_KEY }

def get_full_standings():
    print("順位表を取得中...")
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    response = requests.get(url, headers=headers)
    standings = response.json().get('standings', [{}])[0].get('table', [])
    
    table_data = []
    for i, s in enumerate(standings):
        team_id = s['team']['id']
        team_name = s['team']['name']
        print(f"[{i+1}/20] {team_name} の戦績を取得中...")
        
        # 9回ごとに1分強休んで、API制限(1分10回)を確実に回避[cite: 1, 3]
        if i > 0 and i % 9 == 0:
            print("API制限回避のため65秒待機します...")
            time.sleep(65)
            
        # 直近5試合取得
        f_url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=5"
        f_res = requests.get(f_url, headers=headers)
        form_list = []
        if f_res.status_code == 200:
            matches = f_res.json().get('matches', [])
            for m in reversed(matches):
                is_home = m['homeTeam']['id'] == team_id
                winner = m['score']['winner']
                if winner == 'DRAW': form_list.append("D")
                elif (winner == 'HOME_TEAM' and is_home) or (winner == 'AWAY_TEAM' and not is_home): form_list.append("W")
                else: form_list.append("L")
        
        table_data.append({
            "順位": s['position'],
            "チーム": team_name,
            "試合": s['playedGames'],
            "勝": s['won'],
            "分": s['draw'],
            "負": s['lost'],
            "点": s['points'],
            "直近5試合": " ".join(form_list) if form_list else "-"
        })
        
    df = pd.DataFrame(table_data)
    df.to_csv('standings_data.csv', index=False)
    print("✅ standings_data.csv を作成しました！")

if __name__ == "__main__":
    get_full_standings()