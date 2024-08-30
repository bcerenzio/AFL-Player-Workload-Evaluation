<h1>
SAL 313 Final Project: Evaluating Player Workload in Practice
</h1>

<h4>
For my SAL 313 Final, my group was tasked with using anonymous AFL player data supplied by Edge10 to create and answer an prompt which could be used to better performance. Our group decided to evaluate whether players were undertraining or overtraining on any given training session based on their soreness levels and general wellness (Sleep, Energy, etc.) The code above is what I contributed to the project for my group. 

- Muscle_Soreness.R
    - OLS regression to see if Wellness or Soreness had any effect on total session load. I added Session Duration, Acute Workload, Time in Season, and a Session Type to control for other factors that impact Session Load. I created the Session Type variable using k-means clustering on the Session Load variable to account for practices that were meant to be lighter.
- XGBoost Code
    - Muscle Soreness XGBoost.R predicts the session load for a given player during a session, while Mean Muscle Soreness XGBoost.R predicts the expected session load for a given practice based previous practices. Both models take into account Session Duration, Session Type, Chronic Workload, and Acute Workload among other variables that help determine total session load. However, the player model also includes Muscle Soreness and Wellness variables to isolate those factors in relation to Session Load. We then created a Load Over Expected statistic using the two models to see which is a scaled metric to determine which players would be expected to overtrain or undertrain in a given training session.
</h4>
