# TODO: Fehlende Bilder und Metriken

## Screenshots erstellen

Die folgenden Screenshots müssen aus dem Streamlit-Dashboard erstellt und in `images/` gespeichert werden:

- [ ] **screenshot-forecast-vergleich.png**
  - Quelle: Streamlit Forecast App
  - Inhalt: Vergleich der Zukunftsprognosen aller 4 Modelle (ARIMA, Exponential Smoothing, XGBoost, Chronos2) für Mystic Coin
  - Zeigt: Nächste Stunde (12 Schritte à 5 Minuten)

- [ ] **screenshot-backtest-vergleich.png**
  - Quelle: Streamlit Forecast App
  - Inhalt: Historischer Backtest mit Walk-Forward Validation
  - Zeigt: Prognosen (farbig) vs. tatsächlicher Preisverlauf (blau) für letzte 20% der Daten

- [ ] **screenshot-event-fail.png**
  - Quelle: Streamlit oder Jupyter Notebook
  - Inhalt: Beispiel für Modellversagen bei einem Spiel-Update/Event
  - Zeigt: Plötzlicher Preisanstieg, dem die Prognosen nicht folgen können

## Metriken verifizieren

Die Werte in Tabelle 1 (Kapitel 6) sind als **vorläufig** markiert und müssen aus den finalen Experimenten extrahiert werden:

- [ ] **ARIMA Metriken verifizieren**
  - Aktuell: MAPE 5.8%, SMAPE 5.9%, MAE 125.4, RMSE 150.2

- [ ] **Exponential Smoothing Metriken verifizieren**
  - Aktuell: MAPE 6.3%, SMAPE 6.5%, MAE 135.1, RMSE 162.8

- [ ] **XGBoost Metriken verifizieren**
  - Aktuell: MAPE 7.1%, SMAPE 7.2%, MAE 150.8, RMSE 181.3

- [ ] **Chronos2 Metriken verifizieren** ⚠️
  - Aktuell: MAPE 8.5%, SMAPE 8.7%, MAE 182.3, RMSE 215.6
  - Hinweis: Performance nochmals prüfen (siehe TODO in 06-resultate.md)

## Wo die Ergebnisse eingetragen werden

Nach Erstellung/Verifizierung:
- Bilder in `images/` ablegen
- Metriken in `chapters/06-resultate.md` Tabelle 1 aktualisieren
- "vorläufig" Hinweise entfernen wenn verifiziert

## Dokument-Finalisierung (zusätzliche TODOs)

Diese Punkte sind aktuell **nicht** (oder nur indirekt) durch die obigen Punkte abgedeckt, blockieren aber eine „finale“ Abgabe.

### Platzhalter/Template-Text entfernen

- [ ] **Lukas-Fazit ergänzen**
  - Datei: `chapters/08-fazit.md`
  - Aktuell steht dort ein Platzhalter: `"[Hier dein persönliches Fazit einfügen ...]"`

### Literaturverzeichnis & Zitationen

- [ ] **Bibliography-Datei bereitstellen oder Config anpassen**
  - Datei: `master-document.md` referenziert `references.bib` und `ieee.csl`
  - Aktuell existieren diese Dateien im Repo nicht → entweder anlegen oder die YAML-Referenzen entfernen/anpassen

- [ ] **Zitate/Quellen sauber nachziehen**
  - Datei: `chapters/03-stand-der-technik.md` (Literatur/Dokumentationen werden erwähnt, aber ohne Zitierung)
  - Datei: `chapters/07-diskussion.md` (EMH-Quote aus einem Buch: Quelle/Seite/Edition korrekt zitieren)

### Abbildungen (Nummerierung & Pfade)

- [ ] **Abbildungsnummern konsistent machen**
  - `chapters/05-methodik.md` nutzt „*Abbildung 1*“
  - `chapters/06-resultate.md` nutzt ebenfalls „**Abbildung 1/2/3**“
  - Entscheidung treffen: durchgehend manuell nummerieren oder automatische Captions/Crossrefs nutzen

- [ ] **Bildpfade in Resultaten prüfen/fixen (Master-Build)**
  - Datei: `chapters/06-resultate.md`
  - Aktuell: `../images/...` (kann beim Master-Build ins Leere zeigen, da Kapitel in eine Root-Datei zusammengeführt werden)
  - Erwartung meist: `images/...` (oder Resource-Path entsprechend setzen)
