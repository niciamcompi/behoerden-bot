# Behörden-Copilot Schweiz 🇨🇭

RAG-basierter Chatbot für Schweizer Behördeninformationen.
**Komplett kostenlos** — Gemini 2.0 Flash + Qdrant Free Tier.

---

## Schnellstart

### 1. Voraussetzungen

```bash
python 3.11+
pip install -r requirements.txt
```

### 2. API Keys holen (kostenlos)

| Service | Link | Kosten |
|---------|------|--------|
| Gemini API Key | https://aistudio.google.com/apikey | Kostenlos |
| Qdrant Cloud | https://cloud.qdrant.io | 1GB gratis |

```bash
cp .env.example .env
# .env mit deinen Keys befüllen
```

### 3. Webseiten crawlen

```bash
python -m crawler.scraper
# Speichert JSON-Dateien in data/raw/
```

### 4. In Qdrant einlesen

```bash
python -m ingestion.ingest
# Chunked und uploaded alle Dokumente
```

### 5. API starten

```bash
uvicorn api.main:app --reload
# http://localhost:8000/docs
```

---

## API Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/chat` | POST | RAG-Antwort auf Behördenfragen |
| `/checklist/{thema}` | GET | Vordefinierte Checkliste |
| `/themen` | GET | Alle verfügbaren Themen |
| `/health` | GET | Status-Check |

### Chat Beispiel

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "frage": "Ich ziehe von Zürich nach Bern. Was muss ich beim Einwohneramt machen?",
    "kanton": "ZH",
    "thema": "umzug"
  }'
```

### Checkliste Beispiel

```bash
curl http://localhost:8000/checklist/umzug
```

---

## Deployment (Render Free Tier)

1. GitHub Repo erstellen und Code pushen
2. Auf https://render.com neuen Web Service anlegen
3. Environment Variables setzen (aus .env)
4. Deploy — fertig ✅

**Hinweis**: Free Tier schläft nach 15 Min Inaktivität ein.
Erster Request dauert ~30 Sekunden (Cold Start).

---

## Verfügbare Themen

- `umzug` — Umzug innerhalb der Schweiz
- `geburt` — Geburt eines Kindes
- `firmengründung` — Selbständigkeit / Firma gründen
- `heirat` — Eheschliessung
- `einwanderung` — Einreise und Aufenthalt

---

## Nächste Schritte

- [ ] Flutter App (Android/iOS)
- [ ] Mehr Kantone (AG, BS, GE...)
- [ ] Französisch/Italienisch Support
- [ ] Automatischer Crawler-Refresh (wöchentlich)
- [ ] Google Play Veröffentlichung
