"""
api/main.py - 100% lokal mit Ollama + Web-UI
Start: uvicorn main:app --reload  →  http://localhost:8000
(Swagger bleibt unter /docs erreichbar, die Chat-GUI liegt auf "/")
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "behoerden_ch")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3:1b"
TOP_K = 5

SYSTEM_PROMPT = """Du bist ein Schweizer Behörden-Assistent.
Beantworte nur auf Basis des gegebenen Kontexts.
Wenn die Antwort nicht im Kontext steht: "Dazu habe ich keine verlässliche Behördenquelle gefunden."
Gib immer die Quellen-URLs an. Antworte auf Deutsch."""

CHECKLISTEN = {
    "umzug": {
        "titel": "Umzug innerhalb der Schweiz",
        "schritte": [
            {"schritt": 1, "aufgabe": "Abmeldung beim alten Einwohneramt", "frist": "Vor dem Umzug", "url": "https://www.ch.ch/de/umzug/"},
            {"schritt": 2, "aufgabe": "Anmeldung beim neuen Einwohneramt", "frist": "Innert 14 Tagen", "url": "https://www.ch.ch/de/umzug/"},
            {"schritt": 3, "aufgabe": "Fahrzeug ummelden", "frist": "Innert 14 Tagen", "url": "https://www.ch.ch/de/fahrzeuge/"},
            {"schritt": 4, "aufgabe": "Steueramt neue Adresse melden", "frist": "So früh wie möglich", "url": "https://www.ch.ch/de/steuern-und-finanzen/"},
            {"schritt": 5, "aufgabe": "Krankenkasse Adresse ändern", "frist": "Innerhalb 1 Monat", "url": "https://www.ch.ch/de/krankenkasse/"},
            {"schritt": 6, "aufgabe": "Post Nachsendeauftrag stellen", "frist": "Vor dem Umzug", "url": "https://www.post.ch/"},
        ],
    },
    "geburt": {
        "titel": "Geburt eines Kindes",
        "schritte": [
            {"schritt": 1, "aufgabe": "Geburt beim Zivilstandsamt melden", "frist": "Innert 3 Tagen", "url": "https://www.ch.ch/de/familie/geburt/"},
            {"schritt": 2, "aufgabe": "Kind beim Einwohneramt anmelden", "frist": "Innert 14 Tagen", "url": "https://www.ch.ch/de/familie/geburt/"},
            {"schritt": 3, "aufgabe": "Kind bei Krankenkasse anmelden", "frist": "Innert 3 Monaten", "url": "https://www.ch.ch/de/krankenkasse/"},
            {"schritt": 4, "aufgabe": "Mutterschaftsentschädigung beantragen", "frist": "Frühzeitig", "url": "https://www.ahv-iv.ch/"},
        ],
    },
    "firmengründung": {
        "titel": "Firmengründung / Selbständigkeit",
        "schritte": [
            {"schritt": 1, "aufgabe": "Rechtsform wählen (GmbH, AG, Einzelfirma)", "frist": "Zuerst", "url": "https://www.ch.ch/de/arbeit/selbstaendigkeit/"},
            {"schritt": 2, "aufgabe": "Handelsregistereintrag (ab GmbH/AG)", "frist": "Vor Geschäftsaufnahme", "url": "https://www.zefix.ch/"},
            {"schritt": 3, "aufgabe": "AHV-Ausgleichskasse anmelden", "frist": "Sofort", "url": "https://www.ahv-iv.ch/"},
            {"schritt": 4, "aufgabe": "MWST prüfen (ab CHF 100'000 Umsatz)", "frist": "Vor Überschreitung", "url": "https://www.estv.admin.ch/"},
        ],
    },
}

# Passend zu den SEEDS in crawler/scraper.py
KANTONE = ["bund", "ZH", "BE"]
THEMEN_FILTER = ["umzug", "heirat", "geburt", "todesfall", "firmengründung",
                 "pensionierung", "steuern", "einwanderung", "ahv", "mwst", "familie"]

qdrant_client: Optional[QdrantClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        print("Qdrant verbunden ✓")
    except Exception as e:
        print(f"Qdrant Fehler: {e}")
    yield

app = FastAPI(title="Behörden-Copilot Schweiz", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    frage: str
    kanton: Optional[str] = None
    thema: Optional[str] = None

class ChatResponse(BaseModel):
    antwort: str
    quellen: list[dict]
    kein_ergebnis: bool


def embed_query(text: str) -> list[float]:
    resp = httpx.post(f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text[:1500]}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]

def search_qdrant(query_vector, kanton=None, thema=None):
    filters = []
    if kanton:
        filters.append(FieldCondition(key="kanton", match=MatchValue(value=kanton)))
    if thema:
        filters.append(FieldCondition(key="thema", match=MatchValue(value=thema)))
    search_filter = Filter(must=filters) if filters else None
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=TOP_K,
        with_payload=True,
    ).points
    return [{"text": r.payload.get("text",""), "url": r.payload.get("url",""),
             "titel": r.payload.get("titel",""), "quelle_name": r.payload.get("quelle_name",""),
             "score": r.score} for r in results]

def call_ollama(context: str, frage: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

Kontext:
{context}

Frage: {frage}
Antwort:"""
    resp = httpx.post(f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    return """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Behörden-Copilot Schweiz</title>
<style>
  :root {
    --rot: #DA291C;          /* Schweizer Bundesrot */
    --rot-dunkel: #B22217;
    --tinte: #1A1A1A;
    --grau: #6B6B6B;
    --linie: #E3E0DB;
    --papier: #FAF9F7;
    --weiss: #FFFFFF;
    --gelb: #FFF6DC;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    background: var(--papier); color: var(--tinte);
    display: flex; flex-direction: column;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Kopfzeile: Schweizer Typografie-Stil ── */
  header {
    background: var(--weiss);
    border-bottom: 3px solid var(--rot);
    padding: 18px clamp(16px, 4vw, 40px) 14px;
  }
  .kopf-grid { max-width: 880px; margin: 0 auto; display: flex; align-items: center; gap: 16px; }
  .kreuz {
    width: 40px; height: 40px; background: var(--rot); flex-shrink: 0;
    display: grid; place-items: center; position: relative;
  }
  .kreuz::before, .kreuz::after { content: ""; position: absolute; background: var(--weiss); }
  .kreuz::before { width: 24px; height: 7px; }
  .kreuz::after  { width: 7px; height: 24px; }
  header h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; }
  header .untertitel {
    font-size: 0.72rem; color: var(--grau); text-transform: uppercase;
    letter-spacing: 0.12em; margin-top: 2px;
  }
  .status { margin-left: auto; font-size: 0.72rem; color: var(--grau);
            display: flex; align-items: center; gap: 6px; }
  .status .punkt { width: 8px; height: 8px; border-radius: 50%; background: #C4C4C4; }
  .status .punkt.ok { background: #2E9E5B; }

  /* ── Werkzeugleiste: Filter + Checklisten ── */
  .leiste {
    background: var(--weiss); border-bottom: 1px solid var(--linie);
    padding: 10px clamp(16px, 4vw, 40px);
  }
  .leiste-inner {
    max-width: 880px; margin: 0 auto;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  }
  .feld { display: flex; align-items: center; gap: 6px; }
  .feld label {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--grau);
  }
  select {
    font: inherit; font-size: 0.85rem; padding: 6px 10px;
    border: 1px solid var(--linie); border-radius: 6px; background: var(--weiss);
    color: var(--tinte); cursor: pointer;
  }
  select:focus { outline: 2px solid var(--rot); outline-offset: 1px; }
  .leiste .trenner { width: 1px; height: 22px; background: var(--linie); }
  .check-btn {
    font-size: 0.8rem; padding: 6px 12px; border: 1px solid var(--linie);
    border-radius: 6px; background: var(--weiss); cursor: pointer; color: var(--tinte);
  }
  .check-btn:hover { border-color: var(--rot); color: var(--rot); }

  /* ── Chatbereich ── */
  main { flex: 1; overflow-y: auto; padding: 28px clamp(16px, 4vw, 40px); }
  .chat { max-width: 880px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }

  /* Begrüssung + Beispielfragen */
  .begruessung h2 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
  .begruessung p { color: var(--grau); margin-top: 6px; font-size: 0.95rem; }
  .beispiele { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .beispiel {
    background: var(--weiss); border: 1px solid var(--linie); border-radius: 999px;
    padding: 8px 16px; font-size: 0.85rem; cursor: pointer; color: var(--tinte);
    transition: border-color .15s, color .15s;
  }
  .beispiel:hover { border-color: var(--rot); color: var(--rot); }

  .msg { max-width: 75%; padding: 13px 17px; border-radius: 14px;
         line-height: 1.6; font-size: 0.95rem; }
  .msg.user { background: var(--rot); color: var(--weiss);
              align-self: flex-end; border-bottom-right-radius: 4px; }
  .msg.bot { background: var(--weiss); border: 1px solid var(--linie);
             align-self: flex-start; border-bottom-left-radius: 4px; max-width: 85%; }
  .msg.bot .antwort { white-space: pre-wrap; }
  .msg.bot.warnung { background: var(--gelb); border-color: #E8D9A0; }
  .filter-hinweis { font-size: 0.72rem; opacity: 0.85; margin-bottom: 6px;
                    text-transform: uppercase; letter-spacing: 0.08em; }

  .quellen { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--linie); }
  .quellen .label { font-size: 0.68rem; text-transform: uppercase;
                    letter-spacing: 0.1em; color: var(--grau); margin-bottom: 6px; }
  .quellen a {
    display: block; color: var(--rot-dunkel); text-decoration: none;
    font-size: 0.83rem; padding: 3px 0;
  }
  .quellen a:hover { text-decoration: underline; }
  .quellen a::before { content: "→ "; }

  /* Tipp-Indikator */
  .tippt { align-self: flex-start; display: flex; gap: 5px; padding: 14px 18px;
           background: var(--weiss); border: 1px solid var(--linie); border-radius: 14px; }
  .tippt span { width: 7px; height: 7px; border-radius: 50%; background: var(--grau);
                animation: blink 1.2s infinite; }
  .tippt span:nth-child(2) { animation-delay: 0.2s; }
  .tippt span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .tippt span { animation: none; opacity: .6; } }

  /* Checkliste als Karte im Chat */
  .checkliste { align-self: stretch; background: var(--weiss);
                border: 1px solid var(--linie); border-radius: 14px; overflow: hidden; }
  .checkliste h3 { padding: 12px 18px; font-size: 0.9rem;
                   border-bottom: 2px solid var(--rot); }
  .check-item { display: flex; gap: 14px; padding: 12px 18px;
                border-bottom: 1px solid var(--linie); align-items: flex-start; }
  .check-item:last-child { border-bottom: none; }
  .nr { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--rot);
        font-size: 0.95rem; min-width: 22px; }
  .aufgabe { font-size: 0.9rem; font-weight: 500; }
  .frist { font-size: 0.78rem; color: var(--grau); margin-top: 2px; }
  .check-item a { font-size: 0.78rem; color: var(--rot-dunkel); text-decoration: none; }
  .check-item a:hover { text-decoration: underline; }

  /* ── Eingabezeile ── */
  .eingabe-bereich {
    background: var(--weiss); border-top: 1px solid var(--linie);
    padding: 14px clamp(16px, 4vw, 40px) 18px;
  }
  .eingabe { max-width: 880px; margin: 0 auto; display: flex; gap: 10px; }
  .eingabe input {
    flex: 1; font: inherit; font-size: 1rem; padding: 13px 18px;
    border: 1px solid var(--linie); border-radius: 12px; background: var(--papier);
  }
  .eingabe input:focus { outline: 2px solid var(--rot); outline-offset: 1px; background: var(--weiss); }
  .eingabe button {
    font: inherit; font-weight: 600; font-size: 0.95rem; color: var(--weiss);
    background: var(--rot); border: none; border-radius: 12px;
    padding: 13px 26px; cursor: pointer; transition: background .15s;
  }
  .eingabe button:hover { background: var(--rot-dunkel); }
  .eingabe button:disabled { background: #CFCFCF; cursor: not-allowed; }
  .hinweis { max-width: 880px; margin: 8px auto 0; font-size: 0.7rem; color: var(--grau); }

  @media (max-width: 560px) {
    .msg { max-width: 90%; }
    .msg.bot { max-width: 95%; }
    .eingabe button { padding: 13px 18px; }
    .status { display: none; }
  }
</style>
</head>
<body>

<header>
  <div class="kopf-grid">
    <div class="kreuz" aria-hidden="true"></div>
    <div>
      <h1>Behörden-Copilot Schweiz</h1>
      <div class="untertitel">Offizielle Behördeninformationen · Lokal mit RAG</div>
    </div>
    <div class="status"><span class="punkt" id="status-punkt"></span><span id="status-text">Prüfe Verbindung…</span></div>
  </div>
</header>

<div class="leiste">
  <div class="leiste-inner">
    <div class="feld">
      <label for="kanton">Kanton</label>
      <select id="kanton">
        <option value="">Alle</option>
        <option value="bund">Bund</option>
        <option value="ZH">Zürich</option>
        <option value="BE">Bern</option>
      </select>
    </div>
    <div class="feld">
      <label for="thema">Thema</label>
      <select id="thema">
        <option value="">Alle</option>
        <option value="umzug">Umzug</option>
        <option value="heirat">Heirat</option>
        <option value="geburt">Geburt</option>
        <option value="todesfall">Todesfall</option>
        <option value="firmengründung">Firmengründung</option>
        <option value="pensionierung">Pensionierung</option>
        <option value="steuern">Steuern</option>
        <option value="einwanderung">Einwanderung</option>
        <option value="ahv">AHV</option>
        <option value="mwst">MWST</option>
        <option value="familie">Familie</option>
      </select>
    </div>
    <div class="trenner"></div>
    <button class="check-btn" onclick="ladeCheckliste('umzug')">Checkliste Umzug</button>
    <button class="check-btn" onclick="ladeCheckliste('geburt')">Checkliste Geburt</button>
    <button class="check-btn" onclick="ladeCheckliste('firmengründung')">Checkliste Firma</button>
  </div>
</div>

<main>
  <div class="chat" id="chat">
    <div class="begruessung" id="begruessung">
      <h2>Grüezi! Wie kann ich helfen?</h2>
      <p>Stelle eine Frage zu Schweizer Behördengängen — die Antworten basieren auf offiziellen Quellen wie ch.ch und admin.ch.</p>
      <div class="beispiele">
        <button class="beispiel" onclick="setzeFrage('Was muss ich beim Umzug in der Schweiz beachten?')">Umzug — was beachten?</button>
        <button class="beispiel" onclick="setzeFrage('Ich habe ein Kind bekommen. Was sind die nächsten Schritte?')">Geburt — nächste Schritte</button>
        <button class="beispiel" onclick="setzeFrage('Wie gründe ich eine GmbH in der Schweiz?')">GmbH gründen</button>
        <button class="beispiel" onclick="setzeFrage('Was muss ich bei der Heirat erledigen?')">Heirat</button>
        <button class="beispiel" onclick="setzeFrage('Ab wann muss ich Mehrwertsteuer zahlen?')">MWST-Pflicht</button>
        <button class="beispiel" onclick="setzeFrage('Wie melde ich mich bei der AHV an?')">AHV-Anmeldung</button>
      </div>
    </div>
  </div>
</main>

<div class="eingabe-bereich">
  <div class="eingabe">
    <input id="frage" type="text" placeholder="Frage eingeben…" autocomplete="off"
           onkeydown="if(event.key==='Enter') sendeFrage()" />
    <button id="senden" onclick="sendeFrage()">Senden</button>
  </div>
  <div class="hinweis">Antworten werden lokal generiert und können Fehler enthalten. Verbindliche Auskünfte gibt die zuständige Behörde.</div>
</div>

<script>
const chat = document.getElementById('chat');

function esc(t) {
  const d = document.createElement('div');
  d.textContent = t;
  return d.innerHTML;
}

function entferneBegruessung() {
  const b = document.getElementById('begruessung');
  if (b) b.remove();
}

function setzeFrage(text) {
  document.getElementById('frage').value = text;
  document.getElementById('frage').focus();
}

function addUser(text, kanton, thema) {
  entferneBegruessung();
  const div = document.createElement('div');
  div.className = 'msg user';
  let filter = [];
  if (kanton) filter.push('Kanton: ' + kanton);
  if (thema) filter.push('Thema: ' + thema);
  div.innerHTML = (filter.length ? '<div class="filter-hinweis">' + esc(filter.join(' · ')) + '</div>' : '')
                + esc(text);
  chat.appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
}

function addBot(text, quellen, warnung) {
  const div = document.createElement('div');
  div.className = 'msg bot' + (warnung ? ' warnung' : '');
  let html = '<div class="antwort">' + esc(text) + '</div>';
  if (quellen && quellen.length) {
    html += '<div class="quellen"><div class="label">Quellen</div>';
    quellen.forEach(q => {
      html += '<a href="' + esc(q.url) + '" target="_blank" rel="noopener">'
            + esc(q.titel || q.quelle || q.url) + '</a>';
    });
    html += '</div>';
  }
  div.innerHTML = html;
  chat.appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
}

function zeigeTippt() {
  const div = document.createElement('div');
  div.className = 'tippt';
  div.innerHTML = '<span></span><span></span><span></span>';
  chat.appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
  return div;
}

async function sendeFrage() {
  const input = document.getElementById('frage');
  const frage = input.value.trim();
  if (!frage) return;

  const kanton = document.getElementById('kanton').value || null;
  const thema  = document.getElementById('thema').value || null;

  input.value = '';
  document.getElementById('senden').disabled = true;
  addUser(frage, kanton, thema);
  const tippt = zeigeTippt();

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({frage, kanton, thema})
    });
    const data = await resp.json();
    tippt.remove();
    addBot(data.antwort, data.quellen, data.kein_ergebnis);
  } catch (e) {
    tippt.remove();
    addBot('Verbindung zum Server fehlgeschlagen. Läuft die API auf Port 8000? (' + e.message + ')', [], true);
  }
  document.getElementById('senden').disabled = false;
  input.focus();
}

async function ladeCheckliste(thema) {
  entferneBegruessung();
  try {
    const resp = await fetch('/checklist/' + encodeURIComponent(thema));
    const data = await resp.json();
    if (data.error) { addBot(data.error, [], true); return; }
    const div = document.createElement('div');
    div.className = 'checkliste';
    let html = '<h3>' + esc(data.titel) + '</h3>';
    data.schritte.forEach(s => {
      html += '<div class="check-item"><div class="nr">' + s.schritt + '</div>'
            + '<div><div class="aufgabe">' + esc(s.aufgabe) + '</div>'
            + '<div class="frist">Frist: ' + esc(s.frist) + '</div>'
            + (s.url ? '<a href="' + esc(s.url) + '" target="_blank" rel="noopener">Mehr Infos</a>' : '')
            + '</div></div>';
    });
    div.innerHTML = html;
    chat.appendChild(div);
    div.scrollIntoView({behavior: 'smooth', block: 'end'});
  } catch (e) {
    addBot('Checkliste konnte nicht geladen werden: ' + e.message, [], true);
  }
}

// Health-Check beim Laden
(async () => {
  const punkt = document.getElementById('status-punkt');
  const text = document.getElementById('status-text');
  try {
    const resp = await fetch('/health');
    const data = await resp.json();
    if (data.status === 'ok' && data.qdrant) {
      punkt.classList.add('ok');
      text.textContent = 'Verbunden';
    } else {
      text.textContent = 'Qdrant nicht verbunden';
    }
  } catch {
    text.textContent = 'API nicht erreichbar';
  }
})();

document.getElementById('frage').focus();
</script>
</body>
</html>"""


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "qdrant": qdrant_client is not None}

@app.get("/themen")
async def get_themen():
    return {"themen": list(CHECKLISTEN.keys())}

@app.get("/checklist/{thema}")
async def get_checklist(thema: str):
    if thema not in CHECKLISTEN:
        return {"error": f"Thema '{thema}' nicht gefunden"}
    return CHECKLISTEN[thema]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.frage.strip():
        return ChatResponse(antwort="Bitte stelle eine Frage.", quellen=[], kein_ergebnis=True)

    query_vec = embed_query(req.frage)
    chunks = search_qdrant(query_vec, kanton=req.kanton, thema=req.thema)

    if not chunks or chunks[0]["score"] < 0.3:
        return ChatResponse(
            antwort="Dazu habe ich keine verlässliche Behördenquelle gefunden. Bitte prüfe direkt auf ch.ch.",
            quellen=[], kein_ergebnis=True)

    context = "\n\n---\n\n".join([f"Quelle: {c['url']}\n{c['text']}" for c in chunks])

    try:
        antwort = call_ollama(context, req.frage)
    except Exception as e:
        antwort = f"Fehler beim Sprachmodell: {e}"

    seen, quellen = set(), []
    for c in chunks:
        if c["url"] not in seen:
            seen.add(c["url"])
            quellen.append({"url": c["url"], "titel": c["titel"], "quelle": c["quelle_name"]})

    return ChatResponse(antwort=antwort, quellen=quellen, kein_ergebnis=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
