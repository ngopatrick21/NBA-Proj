NBA Matchup & Win Probability Predictor
I built this project to accurately forecast NBA game outcomes and model team win probabilities using historical data, sequential machine learning pipelines, and an interactive decision-support dashboard.

What it does:
You select any two NBA home and away matchups and adjust their sequential Elo ratings, and the model instantly predicts the home team's win probability and projected matchup outcome using an L2-regularized logistic regression classifier.

How it works:
Data Ingestion:** Pulls multi-season NBA game logs using the `nba_api` package.
Feature Engineering: Builds a chronological sequential Elo rating engine with seasonal resets to track dynamic team strength without data leakage.
Model Training:** Trains an L2-regularized Logistic Regression model optimized via `GridSearchCV`, `StandardScaler`, and `TimeSeriesSplit` cross-validation.
Validation:** Evaluated strictly out-of-sample on unseen future temporal splits, achieving an 0.8110 ROC-AUC.
Interactive Dashboard: Serves real-time inference through a lightweight Streamlit web application.

Results:
Test ROC-AUC: 0.8110
Key Takeaway: Tracking team strength sequentially via dynamic Elo ratings captured massive predictive signal, outperforming static win-loss baselines while completely eliminating data leakage.

How to run it:
Run the following commands in your terminal to set up and launch the app locally:

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py
