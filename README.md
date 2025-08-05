<div align="center">
  <img src="assets/logo.png" alt="logo" width="200"/>
  <h1>DubbAI</h1>
</div>
DubbAI ist ein Tool, das die Übersetzung und Synchronisation von Medien automatisiert. Es verwendet Sprachsynthese, um die Originalstimmen der Sprecher in der übersetzten Sprache nachzuahmen.  

## Funktionen  

### Vorverarbeitung  
- Verarbeitet Video-Dateien automatisch für die Übersetzung und Synchronisation.  
- Behält die ursprüngliche Qualität während der Vorverarbeitung bei.  

### Sprachübersetzung  
- Übersetzt die Originalsprache (zunächst nur Deutsch/Japanisch geplant) ins Englische.  
- Übersetzungen erfolgen entweder über Untertitel oder mithilfe von LLM, je nachdem, was besser funktioniert.  

### Sprachsynthese und Synchronisation  
- Rekonstruiert die Originalstimmen der Sprecher in der übersetzten Sprache mithilfe von Sprachsynthese.  

### Ausgabe  
- Erstellt eine finale Video-Datei mit der neu synchronisierten Tonspur.  

## Key Points  
- Erstellung einer Pipeline zur automatisierten Vorbereitung und Nachbereitung.  
- Feintuning, Training und Testen der Sprachsynthese und Übersetzung.  


## How to run the docker env

```bash
docker-compose build
```
This will build both the `fish-speech` and `dubbai` images. The `fish-speech` build will take longer due to downloading the models (~25GB).

#### 3. Run the Services:
Once the images are built, run:
```bash
docker-compose up
```
This will start both services:
- The `fish-speech` service will be accessible at `http://localhost:7860`.
- The `dubbai` (Jupyter) service will be accessible at `http://localhost:8888`.
