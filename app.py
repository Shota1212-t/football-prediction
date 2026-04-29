import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # エラー回避

import streamlit as st
import torch
import joblib
import pandas as pd
from model import SoccerPredictor  # model.pyから読み込む
from utils import get_recent_points, get_upcoming_matches_api # utils.pyから読み込む

st.title("⚽ Premier League AI Predictor")

# リソースの読み込み
@st.cache_resource
def load_assets():
    model = SoccerPredictor(6)
    model.load_state_dict(torch.load('soccer_model.pth'))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    team_df = pd.read_csv('team_ids.csv')
    team_to_id = dict(zip(team_df.TeamName, team_df.ID))
    return model, scaler, team_to_id

model, scaler, team_to_id = load_assets()

if st.button('試合予測を更新'):
    matches = get_upcoming_matches_api()
    for m in matches[:10]:
        home_name = m['homeTeam']['name']
        away_name = m['awayTeam']['name']
        
        try:
            home_id = team_to_id[home_name]
            away_id = team_to_id[away_name]
            
            # 各種データの取得と正規化
            h_form = get_recent_points(home_id)
            a_form = get_recent_points(away_id)
            
            raw_data = [[home_id, away_id, h_form, a_form, 7.0, 0.0]]
            scaled_data = scaler.transform(raw_data)
            
            with torch.no_grad():
                output = model(torch.FloatTensor(scaled_data))
                prob = torch.nn.functional.softmax(output, dim=1)
            
            # 画面表示
            st.subheader(f"{home_name} vs {away_name}")
            cols = st.columns(3)
            cols[0].metric("HOME Win", f"{prob[0][0]*100:.1f}%")
            cols[1].metric("DRAW", f"{prob[0][1]*100:.1f}%")
            cols[2].metric("AWAY Win", f"{prob[0][2]*100:.1f}%")
            st.divider()
            
        except KeyError:
            continue