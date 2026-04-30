import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import joblib
import pandas as pd
from model import SoccerPredictor
from utils import get_recent_points, get_upcoming_matches_api, get_standings_api, get_top_scorers_api, get_team_form_api
st.set_page_config(page_title="Premier Predictor", page_icon="⚽")
st.title("⚽ Premier League AI Predictor")

# 1. タブの作成（機能3）
tab1, tab2, tab3 = st.tabs(["🔥 試合予測", "📊 順位表", "🏆 選手ランキング"])
@st.cache_resource
def load_assets():
    model = SoccerPredictor(6)
    model.load_state_dict(torch.load('soccer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    team_df = pd.read_csv('team_ids.csv')
    team_to_id = dict(zip(team_df.TeamName, team_df.ID))
    return model, scaler, team_to_id

model, scaler, team_to_id = load_assets()

with tab1:
    if st.button('試合予測を更新'):
        matches = get_upcoming_matches_api()
        for m in matches[:10]:
            home_name = m['homeTeam']['name']
            away_name = m['awayTeam']['name']
            home_crest = m['homeTeam'].get('crest') # エンブレムURL（機能1）
            away_crest = m['awayTeam'].get('crest') # エンブレムURL（機能1）
            
            try:
                home_id = team_to_id[home_name]
                away_id = team_to_id[away_name]
                
                h_form = get_recent_points(home_id)
                a_form = get_recent_points(away_id)
                
                raw_data = [[home_id, away_id, h_form, a_form, 7.0, 0.0]]
                scaled_data = scaler.transform(raw_data)
                
                with torch.no_grad():
                    output = model(torch.FloatTensor(scaled_data))
                    prob = torch.nn.functional.softmax(output, dim=1)
                
                # --- レイアウトの改善 ---
                with st.container():
                    # 3列作って、両端にロゴ、真ん中にチーム名
                    col_h, col_vs, col_a = st.columns([1, 2, 1])
                    with col_h:
                        st.image(home_crest, width=60)
                    with col_vs:
                        st.markdown(f"<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
                    with col_a:
                        st.image(away_crest, width=60)
                    
                    st.markdown(f"<p style='text-align: center;'><b>{home_name} vs {away_name}</b></p>", unsafe_allow_html=True)
                    
                    # 予測確率の表示
                    c1, c2, c3 = st.columns(3)
                    c1.metric("HOME Win", f"{prob[0][0]*100:.1f}%")
                    c2.metric("DRAW", f"{prob[0][1]*100:.1f}%")
                    c3.metric("AWAY Win", f"{prob[0][2]*100:.1f}%")
                    st.divider()
                    
            except KeyError:
                continue

# app.py の Tab 2 部分
with tab2:
    st.subheader("📊 Premier League Standings")
    
    # セッション状態を使って、ボタンを押すまでデータを保持する
    if 'standings_data' not in st.session_state:
        st.session_state.standings_data = None

    if st.button('順位表と戦績を最新にする（APIを消費します）'):
        with st.spinner('データを取得中...'):
            standings = get_standings_api()
            if standings:
                table_data = []
                # API制限を考慮し、上位10チームに絞る
                for s in standings[:8]: 
                    team_id = s['team']['id']
                    recent_form = get_team_form_api(team_id)
                    
                    # もしAPI制限で取得できなかったら "-" にする
                    if recent_form is None:
                        recent_form = "-"
                        
                    table_data.append({
                        "順位": s['position'],
                        "チーム": s['team']['name'],
                        "試合": s['playedGames'],
                        "勝": s['won'],
                        "分": s['draw'],
                        "負": s['lost'],
                        "点": s['points'],
                        "直近5試合": recent_form
                    })
                st.session_state.standings_data = pd.DataFrame(table_data)

    # 取得済みのデータがあれば表示
    if st.session_state.standings_data is not None:
        st.dataframe(st.session_state.standings_data, hide_index=True, use_container_width=True)
    else:
        st.info("「最新にする」ボタンを押すと順位表が表示されます。")

# --- Tab 3: 得点ランキング ---
with tab3:
    st.subheader("🏆 Player Stats Ranking")
    scorers = get_top_scorers_api()
    
    if scorers:
        all_stats = []
        for s in scorers:
            # 安全にデータを取得（キーがない場合に備える）
            p_name = s['player']['name']
            t_name = s['team']['name']
            goals = s.get('goals', 0)
            assists = s.get('assists', 0) if s.get('assists') else 0
            
            all_stats.append({
                "選手名": p_name,
                "チーム": t_name,
                "得点": goals,
                "アシスト": assists,
                "G+A": goals + assists,
                "出場試合": s.get('playedMatches', 0)
            })
        
        df_scorers = pd.DataFrame(all_stats)

        # ランキングの切り替え
        stat_choice = st.radio(
            "ランキング項目を選択:",
            ["得点", "アシスト", "G+A"],
            horizontal=True
        )

        # 選択された項目でソートして、上位20人を抽出
        df_sorted = df_scorers.sort_values(by=stat_choice, ascending=False).head(20)
        
        # 順位列を一番左に追加
        df_sorted.insert(0, '順位', range(1, len(df_sorted) + 1))

        # インデックス（0, 1...）を隠して表示
        st.dataframe(df_sorted, hide_index=True, use_container_width=True)
    else:
        st.warning("選手データを取得できませんでした。1分ほど待ってから再読み込みしてください。")