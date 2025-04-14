from sklearn.metrics import r2_score
import numpy as np

def create_metrics(model):
    print("\nModellleistung wird ausgewertet...")
    y_pred = model.predict(model.X_test)
    y_true = model.y_test

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def smape(A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    
    r2 = r2_score(y_true, y_pred)

    print(f"Modellleistungsmetriken:")
    print(f"  MAE: {mae:.2f} EUR/MWh")
    print(f"  RMSE: {rmse:.2f} EUR/MWh")
    print(f"  MAPE: {mape:.2f} %")
    print(f"  SMAPE: {smape(y_true, y_pred): .2f} %")
    print(f"  R²-Wert: {r2:.2f}")
    
    
    
    # mae = np.mean(np.abs(y_pred - y_true))
    # rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # smape = (
    #     np.mean(np.abs((y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true)) / 2)))
    #     * 100
    # )
    # r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")

    # print(f"Modellleistungsmetriken:")
    # print(f"  MAE: {mae:.2f} EUR/MWh")
    # print(f"  RMSE: {rmse:.2f} EUR/MWh")
    # print(f"  MAPE: {mape:.2f} %")
    # print(f" SMAPE: {smape: .2f} %")
    # print(f" R²-Wert: {r2: .2f}")