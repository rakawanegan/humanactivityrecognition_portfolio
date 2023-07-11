import datetime
import joblib
import pandas as pd


start_date = datetime.date.now()
print(f"Start: {start_date}")
# do something

model = 0
study = 0

y_pred = 0
y_ture = 0

y_submit = pd.DataFrame([y_pred, y_ture])
y_submit.to_csv(f"output.csv")
joblib.dump(model, f"model.pkl")
joblib.dump(study, f"study.pkl")

end_date = datetime.date.now()
print(f"End: {end_date}")
date_diff = end_date - start_date
date_diff = f"{date_diff.hours} hours, {date_diff.seconds} seconds"
print(f"Time: {date_diff}")