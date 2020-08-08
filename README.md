# Cricket-Score-Prediction
A data science project which predicts cricket score of first innings of a team by giving few features

# Problem Statement

Indian Premier League (IPL) is a Twenty20 cricket format league in India. It is usually played in April and May every year. The league was founded by Board of Control for Cricket India (BCCI) in 2008.

We have to predict 1st innings score of a team which is still batting based on certain factors like wickets fallen, overs done, total runs, runs in last 5 overs etc.

# Dataset

The dataset for all the IPL matches details are taken from Kaggle.

# Webapp

Below is a sample webapp for Cricket Score Prediction on Heroku https://cricket-scores-prediction.herokuapp.com/

# How to use Webapp

Select Bat Team
Select Bowl Team
Put Overs for eg 5.1-5.6 (Please note: Overs should not be below 5.1 or in this format 6.0, it should be 5.6 and not 6.0)
Enter Runs
Enter Wickets Fallen
Enter Runs in last 5 overs eg 40
Enter Wickets in last 5 overs eg 4
Predict Score

# Conclusion

We can clearly see in Cricket-Score-Prediction.ipynb notebook that Ridge performed better than Lasso and CatBoost , 
thus we will be using Ridge as our Machine Learning model and exporting same for the webapp and later use.
