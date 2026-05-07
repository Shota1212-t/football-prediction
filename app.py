import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import joblib
import pandas as pd
from model import SoccerPredictor
from utils import get_recent_points, get_upcoming_matches_api, get_standings_api, get_top_scorers_api, get_team_form_api, get_team_details_api
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
                    
                    match_query = f"{home_name} vs {away_name}"
                    st.link_button(
                        label=f"🔍 {match_query} のニュースや対戦成績を調べる",
                        url=f"https://www.google.com/search?q={match_query.replace(' ', '+')}",
                        use_container_width=True
                    )
                    
                    st.divider()
                    
            except KeyError:
                continue

with tab2:
    st.header("📊 プレミアリーグ順位表 & チーム詳細")
    
    # 1. APIから直接データを取得（@st.cache_dataが効くのでAPI制限の心配なし！）
    with st.spinner('データを読み込み中...'):
        standings_list = get_standings_api()
        
    if not standings_list:
        st.error("データの取得に失敗しました。しばらく待ってからリロードしてください。")
    else:
        df_standings = pd.DataFrame(standings_list)
        
        # 念のための安全装置：万が一 'id' がなくてもクラッシュさせない
        if 'id' not in df_standings.columns:
            st.error("データが古い形式です。しばらく経ってから再度お試しください。")
        else:
            # 2. 順位表の表示
            st.write("📋 **チームをクリックすると下に詳細が表示されます**")
            event = st.dataframe(
                df_standings,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                use_container_width=True,
                key="standing_table_auto",
                column_config={"id": None} # IDを隠す
            )

            # 3. チーム詳細の表示
            if event.selection.rows:
                idx = event.selection.rows[0]
                t_id = df_standings.iloc[idx]['id']
                t_name = df_standings.iloc[idx]['チーム']

                st.markdown(f"---")
                st.subheader(f"🛡️ {t_name}")
                
                # 検索ボタン
                st.link_button(f"🔍 {t_name} をGoogleで調べる", 
                               f"https://www.google.com/search?q={t_name.replace(' ', '+')}",
                               use_container_width=True)

                with st.spinner('選手情報をロード中...'):
                    # ここもキャッシュが効くのでAPI制限に引っかかりにくい
                    detail = get_team_details_api(t_id)
                
                if detail:
                    c1, c2 = st.columns(2)
                    c1.info(f"👤 **監督**: {detail.get('coach', {}).get('name', '情報なし')}")
                    c2.info(f"🎨 **カラー**: {detail.get('clubColors', '情報なし')}")

                    # 選手名簿
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
    
# --- Tab 3: 得点ランキング ---
with tab3:
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
                # 検索用URLを作成（選手名 + チーム名 で検索精度を上げる）
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

        # --- 表示設定の変更 ---
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