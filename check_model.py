import joblib
obj = joblib.load("precip_model.joblib")
print(type(obj))
if isinstance(obj, dict):
    print(obj.keys())

import joblib
obj = joblib.load("precip_model.joblib")
print(obj["features"])