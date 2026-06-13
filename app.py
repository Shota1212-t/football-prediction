import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import joblib
import pandas as pd
from model import SoccerPredictor, WorldCupPredictor
from utils import get_recent_points, get_upcoming_matches_api, get_standings_api, get_top_scorers_api, get_team_form_api, get_team_details_api, get_upcoming_world_cup_matches_api, get_country_ranking

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽")
st.title("⚽ Football AI Predictor")

# タブの作成
tab1, tab2, tab3, tab4 = st.tabs(["🔥 PL試合予測", "🌍 Wカップ予測", "📊 順位表", "🏆 選手ランキング"])

@st.cache_resource
def load_assets():
    # Premier League用モデル
    model = SoccerPredictor(6)
    model.load_state_dict(torch.load('soccer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    team_df = pd.read_csv('team_ids.csv')
    team_to_id = dict(zip(team_df.TeamName, team_df.ID))
    return model, scaler, team_to_id

@st.cache_resource
def load_world_cup_assets():
    # World Cup用モデル
    try:
        model_wc = WorldCupPredictor(6)
        model_wc.load_state_dict(torch.load('world_cup_model.pth', map_location=torch.device('cpu')))
        model_wc.eval()
        scaler_wc = joblib.load('world_cup_scaler.pkl')
        return model_wc, scaler_wc
    except FileNotFoundError:
        return None, None

model, scaler, team_to_id = load_assets()
model_wc, scaler_wc = load_world_cup_assets()

# ヘルパー関数：試合予測を表示
def display_match_prediction(home_name, away_name, home_crest, away_crest, home_id=None, away_id=None, is_world_cup=False):
    """試合予測を表示する関数"""
    try:
        if is_world_cup:
            # World Cup用
            if model_wc is None or scaler_wc is None:
                st.warning("⚠️ Wカップモデルがまだ準備できていません。以下のコマンドを実行してください:")
                st.code("python world_cup_data.py\npython train_world_cup.py", language="bash")
                return False
            
            # ランキング情報から特徴量を作成
            home_rank = get_country_ranking(home_name)
            away_rank = get_country_ranking(away_name)
            rank_diff = home_rank - away_rank
            
            # World Cup用特徴量
            raw_data = [[home_rank, away_rank, rank_diff, 1.5, 1.0, 1.0]]
            scaled_data = scaler_wc.transform(raw_data)
            
            with torch.no_grad():
                output = model_wc(torch.FloatTensor(scaled_data))
                prob = torch.nn.functional.softmax(output, dim=1)
        else:
            # Premier League用
            if home_id is None or away_id is None:
                return False
            
            if home_id not in [v for v in team_to_id.values()] or away_id not in [v for v in team_to_id.values()]:
                h_form = 7.0
                a_form = 7.0
            else:
                h_form = get_recent_points(home_id)
                a_form = get_recent_points(away_id)
            
            raw_data = [[home_id, away_id, h_form, a_form, 7.0, 0.0]]
            scaled_data = scaler.transform(raw_data)
            
            with torch.no_grad():
                output = model(torch.FloatTensor(scaled_data))
                prob = torch.nn.functional.softmax(output, dim=1)
        
        # レイアウトの改善
        with st.container():
            col_h, col_vs, col_a = st.columns([1, 2, 1])
            with col_h:
                if home_crest:
                    st.image(home_crest, width=60)
            with col_vs:
                st.markdown(f"<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
            with col_a:
                if away_crest:
                    st.image(away_crest, width=60)
            
            st.markdown(f"<p style='text-align: center;'><b>{home_name} vs {away_name}</b></p>", unsafe_allow_html=True)
            
            # 予測確率の表示
            c1, c2, c3 = st.columns(3)
            c1.metric("HOME Win", f"{prob[0][0]*100:.1f}%")
            c2.metric("DRAW", f"{prob[0][1]*100:.1f}%")
            c3.metric("AWAY Win", f"{prob[0][2]*100:.1f}%")
            
            match_query = f"{home_name} vs {away_name}"
            st.link_button(
                label=f"🔍 {match_query} のニュースや対戦成績を調べる",
                url=f"https://www.google.com/search?q={match_query.replace(' ', '+')}",
                use_container_width=True
            )
            
            st.divider()
        return True
    except Exception as e:
        st.error(f"予測エラー: {e}")
        return False

# Tab 1: PL試合予測
with tab1:
    if st.button('試合予測を更新'):
        matches = get_upcoming_matches_api()
        if not matches:
            st.warning("試合予定が見つかりません")
        else:
            for m in matches[:10]:
                home_name = m['homeTeam']['name']
                away_name = m['awayTeam']['name']
                home_crest = m['homeTeam'].get('crest')
                away_crest = m['awayTeam'].get('crest')
                
                try:
                    home_id = team_to_id[home_name]
                    away_id = team_to_id[away_name]
                    display_match_prediction(home_name, away_name, home_crest, away_crest, home_id, away_id, is_world_cup=False)
                except KeyError:
                    continue

# Tab 2: Wカップ試合予測
with tab2:
    st.subheader("🌍 FIFA World Cup Upcoming Matches Prediction")
    
    if st.button('Wカップ試合予測を更新'):
        with st.spinner('Wカップのデータを取得中...'):
            wc_matches = get_upcoming_world_cup_matches_api()
        
        if not wc_matches:
            st.warning("Wカップのスケジュール情報を取得できませんでした。")
        else:
            match_count = 0
            for m in wc_matches:
                home_name = m['homeTeam']['name']
                away_name = m['awayTeam']['name']
                home_crest = m['homeTeam'].get('crest')
                away_crest = m['awayTeam'].get('crest')
                
                if display_match_prediction(home_name, away_name, home_crest, away_crest, is_world_cup=True):
                    match_count += 1
                    if match_count >= 15:
                        break

# Tab 3: 順位表
with tab3:
    st.header("📊 プレミアリーグ順位表 & チーム詳細")
    
    with st.spinner('データを読み込み中...'):
        standings_list = get_standings_api()
        
    if not standings_list:
        st.error("データの取得に失敗しました。しばらく待ってからリロードしてください。")
    else:
        df_standings = pd.DataFrame(standings_list)
        
        if 'id' not in df_standings.columns:
            st.error("データが古い形式です。しばらく経ってから再度お試しください。")
        else:
            st.write("📋 **チームをクリックすると下に詳細が表示されます**")
            event = st.dataframe(
                df_standings,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                use_container_width=True,
                key="standing_table_auto",
                column_config={"id": None}
            )

            if event.selection.rows:
                idx = event.selection.rows[0]
                t_id = df_standings.iloc[idx]['id']
                t_name = df_standings.iloc[idx]['チーム']

                st.markdown(f"---")
                st.subheader(f"🛡️ {t_name}")
                
                st.link_button(f"🔍 {t_name} をGoogleで調べる", 
                               f"https://www.google.com/search?q={t_name.replace(' ', '+')}",
                               use_container_width=True)

                with st.spinner('選手情報をロード中...'):
                    detail = get_team_details_api(t_id)
                
                if detail:
                    c1, c2 = st.columns(2)
                    c1.info(f"👤 **監督**: {detail.get('coach', {}).get('name', '情報なし')}")
                    c2.info(f"🎨 **カラー**: {detail.get('clubColors', '情報なし')}")

                    players = detail.get('squad', [])
                    if players:
                        st.write("#### 📋 登録選手一覧")
                        df_sq = pd.DataFrame(players)[['name', 'position', 'nationality']]
                        df_sq.columns = ['選手名', 'ポジション', '国籍']
                        st.dataframe(df_sq, hide_index=True, use_container_width=True, key=f"sq_list_{t_id}")
                    else:
                        st.warning("選手データが空でした。")
                else:
                    st.error("API制限などの理由で詳細データを取得できませんでした。1分ほど待ってから別のチームをお試しください。")
    
# Tab 4: 得点ランキング
with tab4:
    st.subheader("🏆 Player Stats Ranking")
    scorers = get_top_scorers_api()
    
    if scorers:
        all_stats = []
        for s in scorers:
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
                "出場試合": s.get('playedMatches', 0),
                "詳細": f"https://www.google.com/search?q={p_name.replace(' ', '+')}+{t_name.replace(' ', '+')}"
            })
        
        df_scorers = pd.DataFrame(all_stats)

        stat_choice = st.radio(
            "ランキング項目を選択:",
            ["得点", "アシスト", "G+A"],
            horizontal=True
        )

        df_sorted = df_scorers.sort_values(by=stat_choice, ascending=False).head(20)
        df_sorted.insert(0, '順位', range(1, len(df_sorted) + 1))

        st.dataframe(
            df_sorted,
            column_config={
                "詳細": st.column_config.LinkColumn(
                    "情報",
                    display_text="🔍 調べる"
                ),
            },
            hide_index=True, 
            use_container_width=True
        )
    else:
        st.warning("選手データを取得できませんでした。1分ほど待ってから再読み込みしてください。")
