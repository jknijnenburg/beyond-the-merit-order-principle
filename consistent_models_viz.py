import types
import shap
import matplotlib.pyplot as plt
import os

from electricity_price_model import ElectricityPriceModel

PLOT_DIR = "./plots/consistent_models/"

def create_consistent_visualizations(all_models, dataset, plot_dir=PLOT_DIR):
    """
    Erweiterte Visualisierungen erstellen basierend auf dem Ensemble von 10 Modellen,
    wie im wissenschaftlichen Paper beschrieben.
    """
    print("\n=== SCHRITT 7: ENSEMBLE-BASIERTE VISUALISIERUNGEN ERSTELLEN ===\n")
    
    if not all_models or len(all_models) == 0:
        print("Keine Modelle für konsistente Visualisierungen vorhanden!")
        return None
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Wähle bestes Modell für grundlegende Visualisierungen
    best_model_info = sorted(all_models, key=lambda x: x['score'])[0]
    best_model = best_model_info['model']
    print(f"Bestes Modell aus {len(all_models)} Modellen ausgewählt (Score: {best_model_info['score']:.4f})")
    
    # 2. Erstelle ein ElectricityPriceModel-Objekt mit diesem Modell
    model_obj = ElectricityPriceModel(random_state=42)
    model_obj.model = best_model
    
    # Setze X_test und y_test - wichtig für alle Visualisierungen
    features = list(best_model_info['shap_importance'].keys())
    model_obj.X_test = dataset[features].copy()
    model_obj.y_test = dataset['price'] if 'price' in dataset.columns else None
    
    # 3. Erkläre das Modell mit SHAP
    print("Berechne SHAP-Werte für Basisvisualisierungen...")
    model_obj.explain_with_shap()
    
    # 4. Erstelle Standard-Visualisierungen mit dem besten Modell
    # Feature-Importance-Plot
    print("Feature-Importance-Plot wird erstellt...")
    importance_df = model_obj.plot_global_feature_importance()
    plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
    plt.close()
    print("Feature-Importance-Plot gespeichert")
    
    # SHAP-Abhängigkeitsplots
    print("SHAP-Abhängigkeitsplots werden erstellt...")
    model_obj.plot_paper_style_shap_dependencies(top_n=3)
    plt.savefig(os.path.join(plot_dir, "paper_style_dependencies.png"), dpi=300)
    plt.close()
    print("SHAP-Abhängigkeitsplots gespeichert")
    
    # 5. Jetzt erweitern wir das Modell mit den korrigierten Interaktionsmethoden
    # Zuerst SHAP-Interaktionswerte berechnen
    print("\nBerechne SHAP-Interaktionswerte (dies kann einige Zeit dauern)...")
    if not hasattr(model_obj, 'explainer'):
        model_obj.explainer = shap.TreeExplainer(model_obj.model)
    model_obj.shap_interaction_values = model_obj.explainer.shap_interaction_values(model_obj.X_test)
    print("SHAP-Interaktionswerte berechnet.")
    
    # 6. Das ursprüngliche ElectricityPriceModel mit korrigierten Interaktionsmethoden erweitern
    # Statt die Klasse zu erweitern, fügen wir die Methoden direkt dem Objekt hinzu
    model_obj.calculate_interaction_values = types.MethodType(
        lambda self: self, model_obj
    )
    
    # Interaktionsgitter erstellen mit der importierten Funktion
    
    output_path = os.path.join(plot_dir, "consistent_interaction_grid.png")
    grid_path = create_fixed_interaction_grid(model_obj, output_path=output_path, dpi=300)
    
    print(f"Konsistentes Interaktionsgitter erstellt: {grid_path}")
    
    # 7. Einzelne wichtige Interaktionsplots erstellen
    print("\nErstelle einzelne Interaktionsplots...")
    key_features = ["load_forecast", "solar_forecast", "wind_forecast", "net_import_export"]
    
    for i, feature1 in enumerate(key_features):
        if feature1 not in model_obj.X_test.columns:
            continue
        for feature2 in key_features[i+1:]:
            if feature2 not in model_obj.X_test.columns:
                continue
                
            # Feature-Indizes ermitteln
            feature_indices = {feature: list(model_obj.X_test.columns).index(feature) 
                            for feature in model_obj.X_test.columns}
                            
            idx1 = feature_indices[feature1]
            idx2 = feature_indices[feature2]
            
            filename = f"interaction_{feature1}_{feature2}.png"
            output_path = os.path.join(plot_dir, filename)
            
            # Interaktionsplot erstellen
            plt.figure(figsize=(10, 8))
            shap.dependence_plot(
                (idx1, idx2),  # Tupel-Notation für Interaktion
                model_obj.shap_interaction_values,
                model_obj.X_test,
                display_features=model_obj.X_test,
                show=False
            )
            
            # Titel und Achsen anpassen
            plt.xlabel(f"{feature1}")
            plt.ylabel(f"SHAP-Interaktionswert für\n{feature1} und {feature2}")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Interaktionsplot erstellt: {os.path.basename(output_path)}")
    
    return grid_path