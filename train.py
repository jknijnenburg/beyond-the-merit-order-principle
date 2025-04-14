import os
import pickle

from electricity_price_model import ElectricityPriceModel


def load_single_model(model_path):
    """Vortrainiertes Modell von der Festplatte laden"""

    if os.path.exists(model_path):
        print(f"Lade vortrainiertes Modell von {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Kein vortrainiertes Modell gefunden unter {model_path}")
        return None


def load_consistent_models(base_path):
    """Lade vortrainierte konsistente Modelle"""
    summary_path = os.path.join(
        os.path.dirname(base_path), "models_summary.pkl")

    if os.path.exists(summary_path):
        print(f"Lade vortrainierte Modellzusammenfassung von {summary_path}")
        with open(summary_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Keine Modellzusammenfassung gefunden unter {summary_path}")
        return None


def train_single_model(dataset, valid_features, save_path=None):
    """Elektrizitätspreisvorhersagemodell trainieren und optional speichern"""

    print("\n=== SCHRITT 5: ELEKTRIZITÄTSPREISMODELL TRAINIEREN ===\n")

    # Prüfen ob genügend Daten vorhanden sind
    if len(dataset) < 100:
        print(
            f"Warnung: Datensatz hat nur {len(dataset)} Zeilen, was möglicherweise nicht für das Training ausreicht.")
        return None

    # Modell initialisieren und trainieren
    print("Elektrizitätspreismodell wird trainiert...")
    model = ElectricityPriceModel(random_state=42)
    model.train(dataset, features=valid_features)
    print("Modelltraining erfolgreich abgeschlossen")

    # Modell speichern falls gewünscht
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modell gespeichert unter {save_path}")

    return model


def train_consistent_models(dataset, valid_features, base_save_path=None, n_splits=10):
    """Mehrere Elektrizitätspreisvorhersagemodelle trainieren für Konsistenzanalyse"""

    print("\n=== SCHRITT 5: KONSISTENTE ELEKTRIZITÄTSPREISMODELLE TRAINIEREN ===\n")

    # Prüfen ob genügend Daten vorhanden sind
    if len(dataset) < 100:
        print(
            f"Warnung: Datensatz hat nur {len(dataset)} Zeilen, was möglicherweise nicht für das Training ausreicht.")
        return None

    # Modell für Konsistenzanalyse initialisieren
    print(
        f"Trainiere {n_splits} Modelle mit unterschiedlichen wöchentlichen Aufteilungen...")
    model = ElectricityPriceModel(random_state=42)

    # Konsistenzanalyse durchführen
    all_models = model.evaluate_model_consistency(
        data=dataset,
        features=valid_features,
        target="price",
        time_column="timestamp",
        n_splits=n_splits
    )

    print(
        f"Konsistenzanalyse mit {len(all_models)} Modellen erfolgreich abgeschlossen")

    # Speichere alle Modelle falls gewünscht
    if base_save_path:
        model_dir = os.path.dirname(base_save_path)
        os.makedirs(model_dir, exist_ok=True)

        # Speichere jedes Modell separat
        for i, model_info in enumerate(all_models):
            save_path = os.path.join(model_dir, f"model_split_{i}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(model_info, f)

        # Speichere auch eine Zusammenfassung der Modelle
        summary_path = os.path.join(model_dir, "models_summary.pkl")
        with open(summary_path, 'wb') as f:
            pickle.dump(all_models, f)

        print(f"Alle Modelle gespeichert unter {model_dir}")

    return all_models
