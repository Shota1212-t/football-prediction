import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import pandas as pd
import time
from dotenv import load_dotenv

# =========================
# APIキー取得
# =========================
load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")

headers = {'X-Auth-Token': API_KEY}

print("API_KEY:", API_KEY)


# =========================
# W杯出場チーム取得
# =========================
def get_world_cup_teams():
    url = "https://api.football-data.org/v4/competitions/WC/teams"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("チーム取得エラー:", response.status_code)
        return []

    data = response.json()
    return data["teams"]


# =========================
# チームごとの試合取得
# =========================


def fetch_world_cup_matches():
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code)
        return []

    data = response.json()
    return data["matches"]




# =========================
# データ収集メイン
# =========================
def collect_all_matches():
    print("W杯出場チーム取得中...")
    teams = get_world_cup_teams()

    all_matches = []

    for team in teams:
        team_id = team["id"]
        team_name = team["name"]

        print(f"取得中: {team_name}")

        matches = get_team_matches(team_id)
        all_matches.extend(matches)

        time.sleep(6)  # API制限対策（重要）

    return all_matches


# =========================
# 特徴量生成
# =========================
def extract_features(matches):

    dataset = []

    for match in matches:
        if match["status"] != "FINISHED":
            continue

        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]

        try:
            home_goals = match["score"]["fullTime"]["home"]
            away_goals = match["score"]["fullTime"]["away"]
        except:
            continue

        if home_goals is None or away_goals is None:
            continue

        # 簡易特徴量
        result = 1 if home_goals > away_goals else 0

        dataset.append({
            "home_goals": home_goals,
            "away_goals": away_goals,
            "goal_diff": home_goals - away_goals,
            "result": result
        })

    return pd.DataFrame(dataset)


# =========================
# 実行
# =========================
def main():
    print("データ収集中...")
    matches = collect_all_matches()

    print("総試合数:", len(matches))

    df = extract_features(matches)

    print("使用試合数:", len(df))

    # 重複削除
    df = df.drop_duplicates()

    df.to_csv("final_training_data.csv", index=False)

    print("✅ final_training_data.csv 保存完了")
    print(df.head())


if __name__ == "__main__":
    main()
