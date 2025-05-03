import numpy as np
import pandas as pd

ipl_matches = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRy2DUdUbaKx_Co9F0FSnIlyS-8kp4aKv_I0-qzNeghiZHAI_hw94gKG22XTxNJHMFnFVKsO4xWOdIs/pub?gid=1655759976&single=true&output=csv"
matches = pd.read_csv(ipl_matches)
def total_teams():
    response=list(set(list(matches["Team1"])+list(matches["Team2"])))
    return response
def teamvsteam(team1,team2):
    matches_played=matches[((matches["Team1"]==team1) & (matches["Team2"]==team2)) | ((matches["Team1"]==team2) & (matches["Team2"]==team1))]
    team1_wins=matches_played["WinningTeam"].value_counts()[team1]
    team2_wins=matches_played["WinningTeam"].value_counts()[team2]
    draws=matches_played.shape[0]-(team1_wins+team2_wins)
    dict1={
        "matches":str(matches_played.shape[0]),
        team1:str(team1_wins),
        team2:str(team2_wins),
        "draws":str(draws)
    }
    return dict1