"""
Debug-Skript zur Identifizierung des Series-Ambiguity-Fehlers
"""
import os
import pandas as pd
import traceback
import sys

# Pfad zum Datenverzeichnis
DATA_DIR = "./data/xlsx"  # Anpassen an Ihren Pfad

# Importieren Sie die Klasse GermanElectricityDataLoader
# Wichtig: Dies sollte nach den obigen Imports erfolgen
from electricity_price_model import GermanElectricityDataLoader

def debug_method(method_name, *args, **kwargs):
    """
    Führt eine Methode aus und fängt Fehler ab, um sie mit nützlichem Kontext auszugeben.
    """
    try:
        print(f"\nTeste Methode: {method_name}")
        method = getattr(data_loader, method_name)
        result = method(*args, **kwargs)
        print(f"✓ {method_name} erfolgreich ausgeführt")
        return result
    except Exception as e:
        print(f"✗ Fehler in {method_name}: {str(e)}")
        print("\nStacktrace:")
        traceback.print_exc()
        return None

# Hilfsfunktion, um Pandas-Series in booleschen Kontexten zu identifizieren
def find_possible_series_bool_issues(code_file):
    """
    Analysiert eine Python-Datei auf potenziell problematische Verwendungen von Pandas Series.
    """
    with open(code_file, 'r') as f:
        lines = f.readlines()
    
    suspicious_patterns = [
        'if ', 
        'elif ', 
        'while ', 
        ' and ', 
        ' or ', 
        ' not ', 
        ' == ', 
        ' != ', 
        ' in '
    ]
    
    exclude_patterns = [
        '.empty', 
        '.any()', 
        '.all()', 
        '.item()', 
        '.bool()', 
        'is None', 
        'is not None',
        'isinstance'
    ]
    
    suspicious_lines = []
    for i, line in enumerate(lines):
        if any(pattern in line for pattern in suspicious_patterns) and not any(pattern in line for pattern in exclude_patterns):
            suspicious_lines.append((i+1, line.strip()))
    
    return suspicious_lines

# Initialisieren des Data Loaders
data_loader = GermanElectricityDataLoader(DATA_DIR)

# Überprüfen jeder Methode einzeln
print("\n=== Test jeder Methode einzeln ===")

# 1. Test load_electricity_prices
electricity_prices = debug_method("load_electricity_prices", [2017])

# 2. Test load_solar_wind_forecasts
solar_df, wind_df = debug_method("load_solar_wind_forecasts", [2017])

# 3. Test load_power_system_features
power_features = debug_method("load_power_system_features", [2017])

# 4. Test load_import_export
import_export_df = debug_method("load_import_export", [2017])

# 5. Test load_fuel_prices
fuel_prices_df = debug_method("load_fuel_prices", [2017])

# Suche nach potenziell problematischem Code
print("\n=== Mögliche Stellen mit Series-Ambiguity-Problemen ===")
if os.path.exists('electricity_price_model.py'):
    issues = find_possible_series_bool_issues('electricity_price_model.py')
    for line_num, line in issues:
        print(f"Zeile {line_num}: {line}")

print("\n=== Vollständiger Datensatz-Test ===")
try:
    # Versuche nun, den vollständigen Datensatz zu laden
    full_dataset = data_loader.prepare_full_dataset([2017])
    print("✓ Vollständiger Datensatz erfolgreich geladen!")
    print(f"Datensatz-Form: {full_dataset.shape}")
    print(f"Spalten: {full_dataset.columns.tolist()}")
except Exception as e:
    print(f"✗ Fehler beim Laden des vollständigen Datensatzes: {str(e)}")
    print("\nStacktrace:")
    traceback.print_exc()