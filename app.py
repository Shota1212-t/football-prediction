import requests
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = { 'X-Auth-Token': API_KEY }

# 1. 過去3シーズン分のデータを取得して統合
def fetch_multi_season_data(seasons=[2023, 2024, 2025]):
    all_matches = []
    for season in seasons:
        print(f"{season}シーズンのデータを取得中...")
        url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            all_matches.extend(response.json()['matches'])
        else:
            print(f"Error {response.status_code}")
        time.sleep(6) # 無料プランのレート制限回避(1分10回)
    return pd.json_normalize(all_matches)

def build_advanced_dataset():
    df = fetch_multi_season_data()
    
    # 基本整形
    df = df[['utcDate', 'homeTeam.name', 'awayTeam.name', 'score.fullTime.home', 'score.fullTime.away', 'status']]
    df = df[df['status'] == 'FINISHED'].copy()
    df['utcDate'] = pd.to_datetime(df['utcDate'])
    df = df.sort_values('utcDate')

    # --- 特徴量計算開始 ---
    
    # A. 勝敗ラベル (Target)
    def get_res(row):
        if row['score.fullTime.home'] > row['score.fullTime.away']: return 0
        if row['score.fullTime.home'] == row['score.fullTime.away']: return 1
        return 2
    df['result'] = df.apply(get_res, axis=1)

    # B. 中何日 (Rest Days)
    # チームごとに最後に試合をした日からの差分を計算
    team_last_date = {}
    def get_rest_days(row, team_key):
        team = row[team_key]
        current_date = row['utcDate']
        if team in team_last_date:
            diff = (current_date - team_last_date[team]).days
            team_last_date[team] = current_date
            return min(diff, 14) # 最大14日としてクリップ
        team_last_date[team] = current_date
        return 7 # 初回データは1週間空きと仮定
    
    df['home_rest'] = df.apply(lambda r: get_rest_days(r, 'homeTeam.name'), axis=1)
    # ※同様にアウェイも計算（実際は時系列順に処理が必要）

    # C. 直近5試合の調子 (Form)
    # 勝ち=3, 分=1, 負=0 として移動合計
    df['home_pts'] = df['result'].map({0: 3, 1: 1, 2: 0})
    df['away_pts'] = df['result'].map({0: 0, 1: 1, 2: 3})
    
    # チーム別の直近成績を計算する関数
    def calculate_form(team_name, current_date):
        past = df[(df['utcDate'] < current_date) & ((df['homeTeam.name'] == team_name) | (df['awayTeam.name'] == team_name))].tail(5)
        # そのチームがホームだった時の勝ち点と、アウェイだった時の勝ち点を合算
        pts = past.apply(lambda r: r['home_pts'] if r['homeTeam.name'] == team_name else r['away_pts'], axis=1)
        return pts.sum()

    print("Form（調子）を計算中...（これには少し時間がかかります）")
    df['home_form'] = df.apply(lambda r: calculate_form(r['homeTeam.name'], r['utcDate']), axis=1)
    df['away_form'] = df.apply(lambda r: calculate_form(r['awayTeam.name'], r['utcDate']), axis=1)

    # D. 直接対決 (H2H)
    # 過去、この2チームが戦った時のホームチームの勝率
    def get_h2h(row):
        h, a = row['homeTeam.name'], row['awayTeam.name']
        past_h2h = df[(df['utcDate'] < row['utcDate']) & (df['homeTeam.name'] == h) & (df['awayTeam.name'] == a)]
        return past_h2h['result'].value_counts().get(0, 0) # 過去のホーム勝ち数

    df['h2h_home_wins'] = df.apply(get_h2h, axis=1)

    # E. チームID化
    teams = sorted(df['homeTeam.name'].unique())
    team_to_id = {name: i for i, name in enumerate(teams)}
    df['home_id'] = df['homeTeam.name'].map(team_to_id)
    df['away_id'] = df['awayTeam.name'].map(team_to_id)

    # 保存
    final_cols = ['home_id', 'away_id', 'home_form', 'away_form', 'home_rest', 'h2h_home_wins', 'result']
    df[final_cols].to_csv('final_training_data.csv', index=False)
    print("最強データセット 'final_training_data.csv' が完成しました！")

if __name__ == "__main__":
    build_advanced_dataset()