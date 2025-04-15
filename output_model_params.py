#%%
import pickle
import pandas as pd

# Load the pickled model
with open('models\electricity_price_model.pkl', 'rb') as f:
    model_obj = pickle.load(f)

# If the pickle contains the ElectricityPriceModel object
# The actual LightGBM model is in the 'model' attribute
lgb_model = model_obj.model

# Get the model parameters
params = lgb_model.params

# Display the parameters in a readable format
print("Model Hyperparameters:")
for param, value in params.items():
    print(f"{param}: {value}")

# You can also see feature importances
feature_importance = lgb_model.feature_importance()
feature_names = model_obj.feature_names if hasattr(model_obj, 'feature_names') else lgb_model.feature_name()

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

# For best hyperparameter setting from random search
if hasattr(model_obj, 'best_score'):
    print(f"\nBest Score (MAE): {model_obj.best_score}")


#%%
sklearn_params = {
    'n_estimators': len(lgb_model.dump_model()['tree_info']),
    'num_leaves': lgb_model.params.get('num_leaves', None),
    'learning_rate': lgb_model.params.get('learning_rate', None),
    'min_child_samples': lgb_model.params.get('min_data_in_leaf', None),
    'subsample': lgb_model.params.get('bagging_fraction', None),
    'colsample_bytree': lgb_model.params.get('feature_fraction', None),
    'reg_alpha': lgb_model.params.get('lambda_l1', None),
    'reg_lambda': lgb_model.params.get('lambda_l2', None)
}

print("scikit-learn compatible parameters:")
for param, value in sklearn_params.items():
    print(f"{param}: {value}")
    
#%%
all_params = lgb_model.params
print("Full parameter set:")
for param, value in all_params.items():
    print(f"{param}: {value}")