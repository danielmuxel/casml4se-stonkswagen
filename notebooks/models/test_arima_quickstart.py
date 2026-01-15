#!/usr/bin/env python
"""
ARIMA Model Quick Start - Manuelles Testen & Performance Check

Dieses Skript zeigt, wie du das ARIMAModel testen und die Performance
auf echten GW2 Daten prüfen kannst.

Usage:
    # Aus dem Projekt-Root:
    uv run python notebooks/models/test_arima_quickstart.py

    # Oder mit einem anderen Item:
    TEST_ITEM_ID=19699 uv run python notebooks/models/test_arima_quickstart.py
"""

import os
from datetime import datetime

from darts.metrics import mape, rmse, mae, smape

# Unsere Module
from gw2ml.data import load_gw2_series
from gw2ml.modeling import ARIMAModel


def main():
    # ══════════════════════════════════════════════════════════════════════════
    # 1. KONFIGURATION
    # ══════════════════════════════════════════════════════════════════════════
    
    item_id = int(os.getenv("TEST_ITEM_ID", "19976"))  # Default: Mystic Coin
    days_back = int(os.getenv("DAYS_BACK", "14"))       # 14 Tage Daten
    train_ratio = 0.8                                    # 80% Training, 20% Test
    
    print("=" * 60)
    print("🔮 ARIMA Model - Quick Start Test")
    print("=" * 60)
    print(f"📅 Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"🎯 Item ID: {item_id}")
    print(f"📊 Tage: {days_back}")
    print()
    
    # ══════════════════════════════════════════════════════════════════════════
    # 2. DATEN LADEN
    # ══════════════════════════════════════════════════════════════════════════
    
    print("📥 Lade Daten...")
    data = load_gw2_series(item_id, days_back=days_back)
    print(data.info())
    print()
    
    # ══════════════════════════════════════════════════════════════════════════
    # 3. TRAIN/TEST SPLIT
    # ══════════════════════════════════════════════════════════════════════════
    
    print(f"✂️  Splitte Daten ({int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test)...")
    train, test = data.split(train=train_ratio)
    print(f"   Train: {len(train)} Datenpunkte")
    print(f"   Test:  {len(test)} Datenpunkte")
    print()
    
    # ══════════════════════════════════════════════════════════════════════════
    # 4. MODELL TRAINIEREN
    # ══════════════════════════════════════════════════════════════════════════
    
    print("🎓 Trainiere ARIMA Modell...")
    
    # Verschiedene ARIMA Konfigurationen zum Vergleich
    configs = [
        {"p": 1, "d": 1, "q": 0, "name": "ARIMA(1,1,0) - AR only"},
        {"p": 0, "d": 1, "q": 1, "name": "ARIMA(0,1,1) - MA only"},
        {"p": 1, "d": 1, "q": 1, "name": "ARIMA(1,1,1) - Standard"},
        {"p": 2, "d": 1, "q": 2, "name": "ARIMA(2,1,2) - Komplex"},
    ]
    
    results = []
    
    for config in configs:
        name = config.pop("name")
        print(f"\n   🔄 {name}...")
        
        try:
            model = ARIMAModel(**config)
            model.fit(train)
            forecast = model.predict(n=len(test))
            
            metrics = {
                "name": str(model),
                "mape": mape(test, forecast),
                "rmse": rmse(test, forecast),
                "mae": mae(test, forecast),
                "smape": smape(test, forecast),
            }
            results.append(metrics)
            print(f"      ✅ MAPE: {metrics['mape']:.2f}%")
            
        except Exception as e:
            print(f"      ❌ Fehler: {e}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 5. ERGEBNISSE
    # ══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("📊 ERGEBNISSE")
    print("=" * 60)
    
    # Sortiere nach MAPE
    results_sorted = sorted(results, key=lambda x: x["mape"])
    
    print(f"\n{'Modell':<20} {'MAPE':>10} {'RMSE':>12} {'MAE':>12} {'SMAPE':>10}")
    print("-" * 66)
    
    for r in results_sorted:
        print(f"{r['name']:<20} {r['mape']:>9.2f}% {r['rmse']:>12.2f} {r['mae']:>12.2f} {r['smape']:>9.2f}%")
    
    print()
    
    # Bestes Modell
    best = results_sorted[0]
    print(f"🏆 Bestes Modell: {best['name']} (MAPE: {best['mape']:.2f}%)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 6. INTERPRETATION
    # ══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("📖 INTERPRETATION")
    print("=" * 60)
    print("""
    MAPE (Mean Absolute Percentage Error):
    - < 10%  : Sehr gut
    - 10-20% : Gut
    - 20-50% : Akzeptabel
    - > 50%  : Schlecht
    
    RMSE (Root Mean Square Error):
    - In der gleichen Einheit wie die Daten (Kupfer = Kupfer-Preis)
    - Bestraft große Fehler stärker
    
    Nächste Schritte:
    1. Versuche verschiedene (p, d, q) Kombinationen
    2. Teste auf verschiedenen Items
    3. Probiere SARIMA für saisonale Daten
    """)


if __name__ == "__main__":
    main()

