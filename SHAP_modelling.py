def generate_shap_explanations(model):
    """SHAP-Erklärungen für das trainierte Modell generieren"""
    
    print("\n=== SCHRITT 6: SHAP-ERKLÄRUNGEN GENERIEREN ===\n")
    
    if not hasattr(model, 'shap_values') or model.shap_values is None:
        print("Generiere SHAP-Werte...")
        model.explain_with_shap()
        print("SHAP-Werte berechnet")
    else:
        print("SHAP-Werte bereits berechnet")
        
    if not hasattr(model, 'shap_interaction_values') or model.shap_interaction_values is None:
        print("Berechne SHAP-Interaktionswerte (dies kann einige Zeit dauern)...")
        model.calculate_interaction_values()
        print("SHAP-Interaktionswerte berechnet")
    else:
        print("SHAP-Interaktionswerte bereits berechnet")
    
    return model