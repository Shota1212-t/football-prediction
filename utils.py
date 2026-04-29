import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = { 'X-Auth-Token': API_KEY }

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

def get_upcoming_matches_api():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    response = requests.get(url, headers=headers)
    return response.json()['matches']