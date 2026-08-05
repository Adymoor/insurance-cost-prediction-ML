import joblib
import pandas
from sklearn.metrics import r2_score


model = joblib.load("AI_model.pkl")

X_test = pandas.read_csv("X_test_data.csv")
y_test = pandas.read_csv("y_test_data.csv")

predict_test = model.predict(X_test)
r2score = r2_score(y_test, predict_test)



print(f"R2 score : {r2score}")