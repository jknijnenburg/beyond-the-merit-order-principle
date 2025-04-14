def identify_valid_features(dataset):
    """Valide Features für das Modelltraining identifizieren"""
    
    print("\n=== SCHRITT 3: VALIDE FEATURES IDENTIFIZIEREN ===\n")
    
    # Prüfen welche Features Daten haben
    print("Analysiere Feature-Datenverfügbarkeit:")
    numeric_features = dataset.select_dtypes(include=["number"]).columns.tolist()
    valid_features = []

    for feature in numeric_features:
        if feature != "price" and not feature.endswith("_ramp"):  # Preis und abgeleitete Ramp-Features überspringen
            non_null_count = dataset[feature].notna().sum()
            non_null_percent = non_null_count / len(dataset) * 100
            print(f"  {feature}: {non_null_count} nicht-leere Werte ({non_null_percent:.2f}%)")

            # Nur Features mit signifikanten Daten einbeziehen
            if non_null_percent > 50:
                valid_features.append(feature)
                # Auch das Ramp-Feature einbeziehen falls es existiert
                ramp_feature = f"{feature}_ramp"
                if ramp_feature in dataset.columns:
                    valid_features.append(ramp_feature)

    print(f"\nVerwende {len(valid_features)} valide Features für das Training: {valid_features}")
    
    # Fehlende Features für die Interaktionsplots prüfen und warnen
    required_features = ["load_forecast", "solar_forecast", "wind_forecast", "net_import_export", "oil", "natural_gas"]
    missing_features = [f for f in required_features if f not in valid_features]
    
    if missing_features:
        print(f"\nWARNUNG: Folgende für Interaktionsplots benötigte Features fehlen: {missing_features}")
        print("Dies wird später zu Fehlern bei der Erstellung der Interaktionsplots führen.")
    
    return valid_features