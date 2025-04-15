# Elektrizitätspreisvorhersage mit robuster SHAP-Interaktionsvisualisierung
# =====================================================================

#%%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
from datetime import datetime
import traceback
import pickle
import shap
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import figure_3
import figure_4
import fuel_price_loader
import train
import types
import features
from excel_electricity_loader import ExcelElectricityDataLoader
from electricity_price_model import ElectricityPriceModel
import SHAP_modelling
import create_metrics

# Verzeichnisse festlegen
DATA_DIR = "./data/xlsx/"
MODEL_SAVE_PATH = "./models/electricity_price_model.pkl"

# Definiere Pfad für die Modellzusammenfassung
MODELS_DIR = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "consistent_models")
MODELS_SUMMARY_PATH = os.path.join(MODELS_DIR, "models_summary.pkl")

# Verzeichnisse erstellen falls nicht vorhanden
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Aktuelles Verzeichnis zum Pfad hinzufügen
current_dir = os.path.dirname(os.path.abspath("__file__"))
if current_dir not in sys.path:
    sys.path.append(current_dir)


#%%
# 1. Brennstoffpreisdaten aufbereiten
# Diese Funktion verarbeitet Öl- und Erdgaspreise und erzeugt stündliche Daten

# Prüfen ob bereits verarbeitete Brennstoffpreise vorhanden sind
fuel_prices_path = os.path.join(DATA_DIR, "processed_fuel_prices.xlsx")
if os.path.exists(fuel_prices_path):
    print(f"Lade vorverarbeitete Brennstoffpreise von {fuel_prices_path}")
    fuel_prices_df = pd.read_excel(fuel_prices_path)
    fuel_prices_df["timestamp"] = pd.to_datetime(fuel_prices_df["timestamp"])
else:
    # Brennstoffpreise verarbeiten
    fuel_prices_df = fuel_price_loader.process_fuel_prices(years=[2017, 2018, 2019])

# Erste Zeilen anzeigen
if fuel_prices_df is not None:
    fuel_prices_df.head()
    
    # Überprüfen welche Spalten enthalten sind
    print(f"Spalten im Brennstoffpreisdatensatz: {fuel_prices_df.columns.tolist()}")
    
    # Prüfen ob 'natural_gas' enthalten ist
    if 'natural_gas' not in fuel_prices_df.columns:
        print("WARNUNG: 'natural_gas' Spalte fehlt im Brennstoffpreisdatensatz!")

#%%
# 2. Elektrizitätspreisdatensatz aufbereiten

loader = ExcelElectricityDataLoader(DATA_DIR)
dataset = loader.prepare_dataset(years=[2017, 2018, 2019], use_cached=True)

# Option 2: Immer neu erstellen ohne Cache
# dataset = loader.prepare_dataset(years=[2017, 2018, 2019], use_cached=False)

# Option 3: Mit benutzerdefinierten Brennstoffpreisen
# custom_fuel_prices = pd.read_csv("custom_fuel_prices.csv")
# dataset = loader.prepare_dataset(custom_fuel_prices_df=custom_fuel_prices)

# Datensatz anzeigen
print("\nDatensatzübersicht:")
print(dataset.head())

#%%
# 3. Valide Features für das Training identifizieren
valid_features = features.identify_valid_features(dataset)

#%%
# 4. Prüfen ob ein vortrainiertes Modell vorhanden ist
model = train.load_single_model(MODEL_SAVE_PATH)

# Falls kein vortrainiertes Modell vorhanden, ein neues trainieren
if model is None:
    model = train.train_single_model(dataset, valid_features, save_path=MODEL_SAVE_PATH)

#%%
# 5. Modell trainieren oder vortrainiertes Modell laden und konsistente Modelle erstellen (best out of 10)

# Prüfen ob vortrainierte konsistente Modelle vorhanden sind
all_models = train.load_consistent_models(MODELS_SUMMARY_PATH)

# Falls keine vortrainierten Modelle vorhanden, neue trainieren
if all_models is None:
    all_models = train.train_consistent_models(
        dataset, 
        valid_features, 
        base_save_path=os.path.join(MODELS_DIR, "model.pkl"),
        n_splits=10  # Genau wie im Paper: 10 verschiedene wöchentliche Aufteilungen
    )

# Wähle das beste Modell für einfache Vorhersagen aus
if all_models:
    # Sortiere nach Score und wähle das Beste
    best_model_info = sorted(all_models, key=lambda x: x['score'])[0]
    print(f"Bestes Modell hat einen Score von {best_model_info['score']:.4f} (Split {best_model_info['split_seed']})")
    best_model = best_model_info['model']
else:
    # Fallback zur alten Methode falls konsistente Modelle nicht erstellt werden konnten
    print("Fallback: Trainiere ein einzelnes Modell...")
    model = train.load_consistent_models(MODEL_SAVE_PATH)
    if model is None:
        model = train.train_single_model(dataset, valid_features, save_path=MODEL_SAVE_PATH)
    best_model = model

#%%
# 6. Modell mit korrigierten SHAP-Interaktionsmethoden erweitern

ElectricityPriceModel = figure_4.extend_model_with_fixed_interactions(ElectricityPriceModel)
print("Modell erfolgreich mit korrigierten SHAP-Interaktionsmethoden erweitert")

#%%
# 7. SHAP-Erklärungen generieren

# SHAP-Erklärungen generieren falls nötig
model = SHAP_modelling.generate_shap_explanations(model)

# Verfügbare Features im Modell ausgeben
if hasattr(model, 'X_test'):
    print("\nVerfügbare Features im Modell:")
    for i, feature in enumerate(model.X_test.columns):
        print(f"  {i}: {feature}")


#%%
# 8. Visualisierungen erstellen
# For a single model:
PLOT_DIR = "./plots/"
os.makedirs(PLOT_DIR, exist_ok=True)
interaction_grid_path = figure_4.create_unified_visualizations(model, plot_dir=PLOT_DIR)

#%%
shap.plots.violin(model.shap_values, feature_names=model.feature_names)

#%%
# OR for ensemble/consistent models:
PLOT_DIR = "./plots/ensemble/"
os.makedirs(PLOT_DIR, exist_ok=True)
interaction_grid_path = figure_4.create_unified_visualizations(all_models, dataset, plot_dir=PLOT_DIR)

#%%
import figure_5
figure_5.create_generation_ramp_dependency_plots(model)


# Option 2: Detailed plot with interaction values
figure_5.create_detailed_dependency_plots(model)

#%%
# figure_5.create_generation_ramp_dependency_plots(all_models)


# # Option 2: Detailed plot with interaction values
# figure_5.create_detailed_dependency_plots(all_models)

#%%
# 8.1 Figure 3 replizieren
slopes_data = figure_3.load_slopes_data("./checkpoints/slopes_data_20250414_130549.pkl")
figure_3.create_combined_dependency_violin_plot(model=model, slopes_data=slopes_data, output_path="./plots/figure3_combined.png")

#%%
# 8. Modellleistung auswerten anhand 1 Modell

# Metriken erstellen
metrics_path = create_metrics.create_metrics(model)

#%%
# 9. Modell-Leistung auswerten anhand Ensemble-Modells
# best_model = sorted(all_models, key=lambda x: x['score'])[0]

# Metriken erstellen
# metrics_path = create_metrics.create_metrics(best_model)

#%%
