import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import pandas as pd
from dotenv import load_dotenv

# APIキーの取得
load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = { 'X-Auth-Token': API_KEY }

def fetch_world_cup_matches():
    """FIFA World Cupの過去試合データを取得"""
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return []
        
        matches = response.json().get('matches', [])
        print(f"取得した試合数: {len(matches)}")
        return matches
    except Exception as e:
        print(f"エラー: {e}")
        return []

def extract_features(matches):
    """試合データから特徴量と結果を抽出"""
    training_data = []
    
    for match in matches:
        # 完了した試合のみを対象
        if match['status'] != 'FINISHED':
            continue
        
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        
        home_score = match['score']['fullTime']['home']
        away_score = match['score']['fullTime']['away']
        
        # 結果を判定（0: ホーム勝, 1: 引き分け, 2: アウェイ勝）
        if home_score > away_score:
            result = 0
        elif home_score == away_score:
            result = 1
        else:
            result = 2
        
        training_data.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'result': result
        })
    
    return training_data

def get_country_ranking(country_name):
    """国のFIFAランキング情報を取得（簡易版：固定値）"""
    # 注：実際のランキングはAPIやスクレイピングで取得することをお勧めします
    # ここではダミー値を使用
    rankings = {
        'Argentina': 1,
        'France': 2,
        'Brazil': 3,
        'England': 4,
        'Belgium': 5,
        'Netherlands': 6,
        'Spain': 7,
        'Germany': 8,
        'Italy': 9,
        'Portugal': 10,
        'Uruguay': 11,
        'Croatia': 12,
        'Mexico': 13,
        'Denmark': 14,
        'Switzerland': 15,
        'Poland': 16,
        'Sweden': 17,
        'Norway': 18,
        'Czech Republic': 19,
        'Austria': 20,
        'Wales': 21,
        'Serbia': 22,
        'Turkey': 23,
        'Japan': 24,
        'South Korea': 25,
        'Canada': 26,
        'USA': 27,
        'Australia': 28,
        'Morocco': 29,
        'Ecuador': 30,
        'Senegal': 31,
        'Iran': 32,
        'Saudi Arabia': 33,
        'Tunisia': 34,
        'Qatar': 35,
        'Costa Rica': 36,
        'Ghana': 37,
        'Cameroon': 38,
        'Hungary': 39,
        'Romania': 40,
    }
    return rankings.get(country_name, 50)  # デフォルト: 50位

def create_training_dataset(training_data):
    """学習用データセットを作成"""
    dataset = []
    
    for match in training_data:
        home_team = match['home_team']
        away_team = match['away_team']
        result = match['result']
        
        # 各国のランキング取得
        home_rank = get_country_ranking(home_team)
        away_rank = get_country_ranking(away_team)
        
        # ランキング差を特徴量として使用
        rank_diff = home_rank - away_rank
        
        # シンプルな特徴量（6次元に統一）
        # [home_rank, away_rank, rank_diff, home_score_history, away_score_history, home_advantage]
        features = [
            home_rank,
            away_rank,
            rank_diff,
            match.get('home_score', 0),  # ホームチームの過去平均得点（簡易）
            match.get('away_score', 0),  # アウェイチームの過去平均失点（簡易）
            1.0  # ホーム優位性
        ]
        
        dataset.append(features + [result])
    
    return dataset

def save_world_cup_data():
    """Wカップデータを保存"""
    print("FIFA World Cupの試合データを取得中...")
    matches = fetch_world_cup_matches()
    
    if not matches:
        print("データ取得に失敗しました")
        return
    
    print("特徴量を抽出中...")
    training_data = extract_features(matches)
    
    print(f"抽出した完了試合数: {len(training_data)}")
    
    if len(training_data) < 10:
        print("警告: 学習用データが10件未満です。モデル学習に不十分な可能性があります。")
    
    print("学習用データセットを作成中...")
    dataset = create_training_dataset(training_data)
    
    # DataFrameに変換して保存
    df = pd.DataFrame(dataset, columns=['home_rank', 'away_rank', 'rank_diff', 'home_avg_goals', 'away_avg_goals', 'home_advantage', 'result'])
    df.to_csv('world_cup_training_data.csv', index=False)
    
    print(f"✅ world_cup_training_data.csv に {len(df)} 件のデータを保存しました")
    print(f"\n最初の5行:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    save_world_cup_data()
