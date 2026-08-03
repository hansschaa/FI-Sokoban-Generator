import pickle
import json

with open("surrogate_models/results/regressor_calibration.pkl", "rb") as f:
    iso = pickle.load(f)

# IsotonicRegression properties
X = iso.X_thresholds_.tolist()
y = iso.y_thresholds_.tolist()

out_data = {
    "X_thresholds": X,
    "y_thresholds": y,
    "X_min": float(iso.X_min_),
    "X_max": float(iso.X_max_)
}

with open("surrogate_models/results/regressor_calibration.json", "w") as f:
    json.dump(out_data, f)
print("Exported to surrogate_models/results/regressor_calibration.json")
