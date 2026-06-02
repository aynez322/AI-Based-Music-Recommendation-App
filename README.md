# AI-Based-Music-Recommendation-App
# Aplicație de Recomandare Muzicală bazată pe AI

Un sistem explicabil de recomandare muzicală construit cu o **arhitectură multi-agent CrewAI**, care combină analiza caracteristicilor audio, clasificarea genului cu XGBoost și explicabilitate prin SHAP pentru a sugera melodii similare cu orice piesă din dataset.

---

## Screenshot

<img width="1920" height="1080" alt="Input" src="https://github.com/user-attachments/assets/1bf7afc0-f5bc-46ab-ac72-9bf95c808505" />


---

## Cum funcționează

Sistemul folosește **3 agenți AI specializați** care lucrează secvențial:

| Agent | Rol | Unelte |
|---|---|---|
| Song Analyzer | Extrage 9 trăsături audio ale piesei de intrare | `song_lookup` |
| Music Researcher | Identifică genul și găsește 5 melodii similare | `genre_fingerprinter`, `similar_song_search` |
| Explainability Agent | Explică DE CE melodiile sunt similare, folosind valori SHAP | `shap_explain_similarity` |

**Pipeline-ul de procesare:**
1. Utilizatorul introduce titlul piesei și artistul
2. Agentul 1 caută profilul audio al piesei (dansabilitate, energie, intensitate, tempo, valență etc.)
3. Agentul 2 potrivește piesa cu unul din cele 30 de profiluri de gen folosind similaritate cosinus, apoi găsește cele mai similare 5 melodii
4. Agentul 3 rulează SHAP pe un model XGBoost antrenat pentru a genera explicații în limbaj natural pentru fiecare recomandare
5. Rezultatele sunt afișate în terminal și salvate în `src/ai_based_music_recommendation/output/recommendations.md`

---

## Tehnologii folosite

| Componentă | Librărie / Versiune |
|---|---|
| Limbaj | Python 3.10 – 3.13 |
| Framework multi-agent | CrewAI >= 0.102 |
| Clasificator de gen | XGBoost >= 3.2 |
| Explicabilitate | SHAP >= 0.49 |
| Manipulare date | pandas >= 2.0, NumPy >= 1.26 |
| Utilități ML | scikit-learn >= 1.7 |
| LLM (local) | Ollama / LLaMA 3.2 |
| LLM (cloud, opțional) | Gemini 2.5 Flash |
| Manager pachete | uv |

---

## Structura proiectului

```
AI-Based-Music-Recommendation-App/
├── src/ai_based_music_recommendation/
│   ├── main.py               # Punct de intrare & CLI
│   ├── crew.py               # Crew CrewAI — 3 agenți, 3 taskuri
│   ├── config/
│   │   ├── agents.yaml       # Rol / scop / backstory pentru fiecare agent
│   │   └── tasks.yaml        # Descrierile taskurilor și output-urile așteptate
│   ├── tools/
│   │   ├── dataset_tools.py        # song_lookup, genre_fingerprinter, similar_song_search
│   │   └── explainability_tools.py # shap_explain_similarity
│   └── output/
│       └── recommendations.md
├── data/
│   ├── merged_dataset.parquet      # Dataset principal (~1M+ melodii)
│   ├── genre_fingerprints          # 30 profiluri de gen muzical
│   └── genre_classifier.pkl        # Model XGBoost antrenat (generat automat la primul rulaj)
├── scripts/
│   ├── prepare_zenodo.py
│   ├── build_final_dataset.py
│   ├── merge_kaggle.py
│   └── train_genre_classifier.py
├── pyproject.toml
└── .env
```

---

## Instalare

### 1. Cerințe prealabile

- Python 3.10+
- Managerul de pachete [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com/) rulând local cu modelul `llama3.2` descărcat

```bash
ollama pull llama3.2
```

### 2. Instalare dependențe

```bash
uv sync
```

### 3. Configurare mediu

Creează un fișier `.env` în rădăcina proiectului:

```env
# LLM — Ollama local (implicit)
MODEL=ollama/llama3.2
OPENAI_API_KEY=ollama

# Sau comutare la Gemini (cloud)
# MODEL=gemini/gemini-2.5-flash
# OPENAI_API_KEY=<cheia-ta-gemini>

# Kaggle (necesar pentru descărcarea dataseturilor)
KAGGLE_USERNAME=username_tau
KAGGLE_KEY=token_api_kaggle

# Spotify (îmbogățire opțională)
SPOTIFY_CLIENT_ID=spotify_client_id
SPOTIFY_CLIENT_SECRET=spotify_client_secret
```

### 4. Pregătirea datelor (doar prima dată)

```bash
python scripts/prepare_zenodo.py
python scripts/build_final_dataset.py
python scripts/train_genre_classifier.py
```

> Modelul XGBoost se antrenează automat la primul rulaj dacă fișierul `genre_classifier.pkl` lipsește.

---

## Rulare

```bash
# Mod interactiv - solicită titlul și artistul
python -m ai_based_music_recommendation.main sau python main.py

# Transmitere directă a melodiei și artistului
python -m ai_based_music_recommendation.main "Bohemian Rhapsody" "Queen"

# Prin script-ul instalat
run_recommendations
```

Rezultatul este afișat în terminal cu secțiuni colorate și salvat și în `output/recommendations.md`.

---

## Detalii model ML

**Clasificator de gen XGBoost**
- Intrare: 11 trăsături audio (9 trăsături Spotify + mod + tonalitate)
- Ieșire: unul din 17 genuri consolidate (Rock, Pop, Hip-hop, Jazz, Electronic, R&B, Clasic, Country, Folk, Latin, Metal, Reggae, K-pop, Blues, Ska, Ambient, World)
- Configurare: 100 arbori, max_depth=6, learning_rate=0.1

**Amprente de gen (Genre Fingerprints)**
- 30 profiluri bazate pe reguli cu intervale caracteristice de trăsături
- Potrivire prin similaritate cosinus față de cel mai apropiat centroid

**Explicabilitate SHAP**
- Folosește `TreeExplainer` pentru a calcula valorile Shapley per recomandare
- Evidențiază care trăsături audio (ex: energie ridicată, acousticness scăzut) fac două piese similare

---

---

## Comparație LLM: Ollama (LLaMA 3.2) vs Gemini 2.5 Flash

Sistemul suportă ambele opțiuni prin variabila `MODEL` din `.env`. Mai jos o comparație practică bazată pe comportamentul observat în acest pipeline.

| Criteriu | Ollama / LLaMA 3.2 | Gemini 2.5 Flash |
|---|---|---|
| **Cost** | Gratuit (resurse locale) | ~$0.075 / 1M tokeni input |
| **Confidențialitate** | Date 100% locale | Date trimise la Google |
| **Instalare** | `ollama pull llama3.2` + Ollama rulând local | Doar cheie API în `.env` |
| **Latență** | Depinde de GPU/CPU local (2–15 s/pas) | ~1–3 s/pas (rețea + inferență cloud) |
| **Context window** | 128 K tokeni | 1 M tokeni |
| **Calitate output Agent 3** | Uneori returnează JSON brut în loc de proză — există fallback dedicat în `main.py` | Produce consecvent explicații în limbaj natural |
| **Urmărire instrucțiuni** | Moderată - poate devia de la formatul cerut | Ridicată - respectă prompt-ul structurat |
| **Disponibilitate** | Necesită mașina locală pornită | SLA 99.9%, accesibil de oriunde |

### Care este mai bun pentru acest proiect?

**Gemini 2.5 Flash** este alegerea recomandată dacă ai acces la internet și o cheie API:
- Agentul 3 (Explainability) generează explicații SHAP în proză coerentă fără fallback-uri.
- Latența totală a pipeline-ului este mai mică decât pe un laptop fără GPU dedicat.
- Urmărirea instrucțiunilor complexe din `tasks.yaml` este mai fiabilă.

**Ollama / LLaMA 3.2** este alegerea potrivită dacă:
- Rulezi offline sau ai restricții de confidențialitate (datele muzicale nu părăsesc mașina).
- Ai un GPU local cu ≥ 8 GB VRAM (inferența devine comparabilă ca viteză).
- Vrei zero costuri recurente.

  

> **Notă tehnică:** Codul conține un fallback `_render_shap_json` în `main.py` activat automat când agentul returnează JSON brut - comportament observat frecvent cu LLaMA 3.2, rar cu Gemini.

## Licență

MIT


