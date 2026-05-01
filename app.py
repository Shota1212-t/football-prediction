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
                    
                    match_query = f"{home_name} vs {away_name}"
                    st.link_button(
                        label=f"🔍 {match_query} のニュースや対戦成績を調べる",
                        url=f"https://www.google.com/search?q={match_query.replace(' ', '+')}",
                        use_container_width=True
                    )
                    
                    st.divider()
                    
            except KeyError:
                continue

# app.py の Tab 2 部分
with tab2:
    st.subheader("📊 Premier League Standings")
    
    if st.button('順位表データを更新'):
        with st.spinner('最新の順位表を取得中...'):
            standings_data = get_standings_api() # utils.pyで修正した、IDを含むリストが返ってくる
            if standings_data:
                df_new = pd.DataFrame(standings_data)
                df_new.to_csv('standings_data.csv', index=False)
                st.success("順位表を更新しました！(ID情報も保存されました)")
                st.rerun() # 画面を再読み込みしてCSVを読み直す

    # 1. データの読み込み
    try:
        df_standings = pd.read_csv('standings_data.csv')
    except Exception as e:
        st.error(f"CSVファイルの読み込みに失敗しました: {e}")
        st.stop()
    
    # 2. 順位表の表示（最新のハイフン形式 selection_mode）
    # これで「チーム名」の部分がクリック（選択）できるようになります
    event = st.dataframe(
        df_standings,
        on_select="rerun",
        selection_mode="single-row", # アンダースコア(_)ではなくハイフン(-)
        hide_index=True,
        use_container_width=True
    )

    # 3. チームが選択された時の処理
    if event.selection.rows:
        selected_row_index = event.selection.rows[0]
        
        # CSVの列名を確認しながらデータを取得
        try:
            # もし 'id' という列名がない場合のために .get を使うか、直接指定
            team_id = df_standings.iloc[selected_row_index]['id']
            team_name = df_standings.iloc[selected_row_index]['チーム']
        except KeyError:
            st.warning("CSVに 'id' 列が見つかりません。順位表データを更新するか、CSVを確認してください。")
            st.stop()

        st.divider()
        st.subheader(f"🛡️ {team_name} の詳細")

        # utils.pyで作った関数でAPIから詳細（選手一覧など）を取得
        with st.spinner('データを取得中...'):
            detail = get_team_details_api(team_id)
        
        if detail:
            # 監督情報
            coach_name = detail.get('coach', {}).get('name', '情報なし')
            st.info(f"👤 **監督**: {coach_name}")
            
            # 選手一覧のデータフレーム作成
            players = detail.get('squad', [])
            if players:
                st.write("### 📋 選手一覧")
                st.write("名前を選択すると、APIから取得できる全詳細データを確認できます。")
                
                df_squad = pd.DataFrame(players)[['name', 'position', 'nationality', 'dateOfBirth']]
                df_squad.columns = ['選手名', 'ポジション', '国籍', '生年月日']
                
                # 選手選択用のデータフレーム
                p_event = st.dataframe(
                    df_squad,
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True,
                    use_container_width=True
                )
                
                # 4. 選手が選択されたら全情報をJSONで表示
                if p_event.selection.rows:
                    p_idx = p_event.selection.rows[0]
                    selected_player_all_data = players[p_idx]
                    
                    with st.expander(f"🔍 {selected_player_all_data['name']} の全データ", expanded=True):
                        st.json(selected_player_all_data)
            else:
                st.write("選手情報が見つかりませんでした。")

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