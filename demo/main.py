from sklearn.model_selection import ParameterGrid
from config import params

# Simulated function to "fit" a model
def fit_model(alpha, l1_ratio, l2_ratio):
    print(f"Fitting model with alpha={alpha} and l1_ratio={l1_ratio}")
    # Simulate model fitting
    return f"model_{alpha}_{l1_ratio}_{l2_ratio}"

# Simulated function to "evaluate" the model
def evaluate_model(model):
    # Simulate evaluation
    print(f"Evaluating {model}")
    return {"score": 0.9}  # Simulated score

for param in ParameterGrid(params):
    # model = fit_model(**param)
    
    # # Simulate evaluating the model
    # metrics = evaluate_model(model)
    
    # # Simulate logging the parameters and metrics (in a real scenario, this would use MLflow or another logging tool)
    # print(f"Logged params: {params}")
    # print(f"Logged metrics: {metrics}")
    # print("---")
    print(param)
