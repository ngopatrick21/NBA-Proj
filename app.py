import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('nba_logistic_model.pkl')
scaler = joblib.load('nba_scaler.pkl')

st.title("🏀 NBA Matchup & Win Probability Predictor")
st.write("Select team parameters below to run real-time inference using your L2-regularized Logistic Regression pipeline.")

# User Input Controls
col1, col2 = st.columns(2)

# List of all 30 NBA teams for both home and away dropdowns
nba_teams = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

with col1:
    st.subheader("Home Team Settings")
    home_team = st.selectbox("Home Team", nba_teams)
    home_elo = st.slider("Home Elo Rating", 1300.0, 1750.0, 1550.0, 5.0)

with col2:
    st.subheader("Away Team Settings")
    away_team = st.selectbox("Away Team", nba_teams, index=6) # Default to DAL or any index
    away_elo = st.slider("Away Elo Rating", 1300.0, 1750.0, 1480.0, 5.0)

# Compute Features (matching your model's expected feature columns: ELO_DIFF, IS_HOME)
elo_diff = home_elo - away_elo
is_home = 1  # Home team flag is always 1 for the home perspective

# Prepare feature array
X_input = np.array([[elo_diff, is_home]])
X_input_scaled = scaler.transform(X_input)

# Predict Probability
if st.button("Predict Matchup Outcome"):
    prob = model.predict_proba(X_input_scaled)[0][1] * 100
    
    st.divider()
    st.subheader("Prediction Results")
    st.metric(label=f"{home_team} Win Probability vs {away_team}", value=f"{prob:.1f}%")
    
    if prob > 50:
        st.success(f"Advantage: **{home_team}** is projected to win.")
    else:
        st.warning(f"Advantage: **{away_team}** is projected to upset.")