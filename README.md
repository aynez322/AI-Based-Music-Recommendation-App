# AI-Based-Music-Recommendation-App
# Aplicație de Recomandare Muzicală bazată pe AI

Un sistem explicabil de recomandare muzicală construit cu o **arhitectură multi-agent CrewAI**, care combină analiza caracteristicilor audio, clasificarea genului cu XGBoost și explicabilitate prin SHAP pentru a sugera melodii similare cu orice piesă din dataset.

Ideea centrală: pornind de la o singură piesă („titlu + artist"), sistemul găsește cele mai apropiate 5 melodii din punct de vedere sonor **și explică, în limbaj natural, de ce** fiecare a fost recomandată. Toate calculele grele (similaritate, clasificare gen, valori SHAP) se fac determinist în Python, iar LLM-ul local se ocupă doar de partea de explicație în cuvinte.

---

## Screenshot

<img width="1920" height="1080" alt="Input" src="https://github.com/user-attachments/assets/1bf7afc0-f5bc-46ab-ac72-9bf95c808505" />


---

## Cum funcționează

Sistemul folosește **3 agenți AI specializați** care lucrează secvențial (`Process.sequential`), fiecare cu **o singură unealtă**, iar output-ul fiecărui task este transmis ca context următorului:

| Agent | Rol | Unealtă |
|---|---|---|
| Song Analyzer | Extrage cele 9 trăsături audio ale piesei de intrare și produce un profil muzical | `song_lookup` |
| Music Researcher | Găsește cele 5 melodii cele mai similare prin similaritate cosinus | `similar_song_search` |
| Explainability Agent | Explică în limbaj natural DE CE fiecare melodie a fost recomandată, folosind XGBoost + SHAP | `shap_analyze_recommendations` |

**Pipeline-ul de procesare:**

1. **Punctul de intrare (`main.py` → `run()`)** — utilizatorul introduce titlul piesei și artistul (interactiv sau ca argumente CLI). Înainte de a porni agenții, `_precheck()` verifică prin `song_lookup` că piesa există în dataset; dacă nu, afișează sugestii și oprește execuția. Apoi se apelează `MusicRecommendationCrew().crew().kickoff(inputs=...)`.

2. **Agentul 1 — Song Analyzer** caută profilul audio al piesei (dansabilitate, energie, intensitate/loudness, speechiness, acousticness, instrumentalness, liveness, valență, tempo), împreună cu genul și popularitatea din dataset, și le traduce în descrieri în limbaj natural.

3. **Agentul 2 — Music Researcher** apelează `similar_song_search` (similaritate cosinus pe cele 9 trăsături audio normalizate, peste tot dataset-ul) și întoarce cele mai apropiate 5 melodii, ordonate descrescător după scorul de similaritate. Piesa de intrare este **exclusă explicit** din rezultate (vezi „Probleme cunoscute") pentru a nu se recomanda pe ea însăși.

4. **Agentul 3 — Explainability Agent** apelează `shap_analyze_recommendations` **o singură dată**. Această unealtă, pentru piesa de intrare și fiecare dintre cele 5 recomandări:
   - rulează clasificatorul **XGBoost** pentru a prezice genul,
   - calculează valorile **SHAP** (TreeExplainer) pentru a afla ce trăsături audio contează cel mai mult,
   - calculează similaritatea cosinus,
   - și întoarce „dovezi" pre-digerate (gen prezis, top trăsături comune cu valori și impact SHAP, scor de similaritate), inclusiv liniile exacte de titlu/gen gata de copiat.

   LLM-ul folosește **doar aceste dovezi** ca să scrie **câte un paragraf per melodie**, explicând de ce se potrivește cu piesa de intrare. Nu inventează numere, genuri sau melodii și nu afișează JSON brut.

5. Rezultatul este afișat în terminal cu secțiuni colorate și salvat în `output/recommendations.md`.

> **De ce această arhitectură?** Modelul local `llama3.2` este prea slab ca să orchestreze apeluri complexe de unelte în mai mulți pași. Soluția: fiecare agent are exact o unealtă, toată computația grea (XGBoost, SHAP, cosinus) stă în unelte Python deterministe, iar LLM-ul scrie doar explicații în limbaj natural pornind de la dovezi deja procesate. Astfel proiectul rămâne un sistem multi-agent real, dar funcționează în limitele modelului.

---

## Cum funcționează uneltele
Toate uneltele sunt clase `BaseTool` din CrewAI, cu un `args_schema` Pydantic, și partajează un dataset încărcat o singură dată și ținut în cache (`_load()`), care la prima citire **normalizează** cele 9 trăsături audio în intervalul [0, 1] (min-max) — pas necesar pentru ca similaritatea cosinus să nu fie dominată de trăsături cu scară mare (ex: `tempo` sau `loudness`):

```python
mat = df[NUMERIC_FEATURES].values.astype(float)
mins, maxs = mat.min(axis=0), mat.max(axis=0)
ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
_norm_cache = (mat - mins) / ranges          # matrice normalizata, refolosita la fiecare cautare
```

### 1. `song_lookup` (SongLookupTool)
Caută o piesă după titlu (+ artist opțional) și întoarce trăsăturile ei audio, genul și popularitatea. Potrivirea se face prin `_find_index`: substring case-insensitive pe titlu, restrâns cu artistul dacă e dat, preferând potrivirea exactă și, la egalitate, piesa cu popularitatea cea mai mare. Dacă nu găsește nimic, întoarce sugestii.

```python
row = df.loc[idx]
return json.dumps({
    "track_name": str(row.get("track_name", "")),
    "artists":    str(row.get("artists", "")),
    "genre":      str(row.get("track_genre", "")),
    "popularity": int(row.get("popularity", 0)),
    "features":   {f: round(float(row[f]), 4) for f in NUMERIC_FEATURES if f in row},
})
```

### 2. `similar_song_search` (SimilarSongSearchTool)
Inima recomandării. Ia vectorul normalizat al piesei de referință și calculează **similaritatea cosinus** față de toate celelalte piese dintr-o singură operație matriceală, apoi întoarce primele `top_n` (implicit 5):

```python
ref_vec   = norm[ref_idx]
ref_norm  = np.linalg.norm(ref_vec) or 1.0
row_norms = np.linalg.norm(norm, axis=1)
similarities = (norm @ ref_vec) / (row_norms * ref_norm)   # cosinus vectorizat pe tot dataset-ul
```

Înainte de a returna rezultatele, **exclude piesa de intrare și toate copiile cu același titlu** (covere/karaoke/remasterizări care altfel ar scora ~1.00 și ar „recomanda" practic aceeași piesă):

```python
df_work = df_work[df_work.index != ref_idx]
ref_name_lower = str(df.loc[ref_idx, "track_name"]).strip().lower()
same_title = df_work["track_name"].str.strip().str.lower() == ref_name_lower
df_work = df_work[~same_title]
```

Opțional, rezultatele pot fi restrânse la același gen (parametrul `genre_filter` sau genul din dataset), păstrând filtrul doar dacă rămân destui candidați.

### 3. `shap_analyze_recommendations` (SHAPAnalysisTool)
Unealta Agentului 3. Într-un singur apel: găsește cele 5 piese similare (reapelând `similar_song_search`), apoi pentru fiecare rulează **clasificatorul XGBoost** pentru genul prezis și **SHAP** pentru a afla ce trăsături contează la acea predicție. Pentru fiecare pereche (intrare, recomandare) păstrează trăsăturile cu impact SHAP mare și **aliniate** (împing predicția în aceeași direcție la ambele piese):

```python
for i, feat in enumerate(CLASSIFIER_FEATURES):
    sv_in, sv_rec = float(in_shap[i]), float(rec_shap[i])
    aligned = (sv_in >= 0) == (sv_rec >= 0)          # ambele contribuie in aceeasi directie
    feats.append({"feature": feat, "iv": iv, "rv": rv,
                  "aligned": aligned,
                  "shap": round((abs(sv_in) + abs(sv_rec)) / 2, 3)})
feats.sort(key=lambda x: x["shap"], reverse=True)
top = [f for f in feats if f["aligned"]][:3]
```

Output-ul nu este JSON, ci „dovezi" gata de folosit: o linie de titlu și una de gen marcate „copy exactly", plus trăsăturile comune cu valori și impact SHAP. LLM-ul doar le transformă în câte un paragraf per piesă — nu calculează și nu inventează nimic.

> Funcțiile auxiliare cheie: `_resolve_genre` (ia genul din dataset dacă există, altfel îl prezice cu XGBoost), `_shap_for_class` (extrage valorile SHAP pentru clasa prezisă), `_consolidate_genre` (mapează sub-genurile Spotify la categorii largi).

---

## Scripturile de pregătire a datelor

Pipeline-ul de date rulează o singură dată, în ordinea de mai jos, și produce `merged_dataset.parquet` (folosit la rulare) și `genre_classifier.pkl` (modelul antrenat):

| Script | Ce face |
|---|---|
| `prepare_zenodo.py` | Convertește `datasets/zenodo.csv` în parquet: redenumește coloanele la schema internă, extrage primul gen din lista de genuri, normalizează `key`/`mode`, elimină rândurile fără trăsături audio și **deduplică** după (titlu, artist) păstrând varianta cu gen cunoscut și popularitate maximă. |
| `merge_kaggle.py` | Combină **7 fișiere CSV Kaggle** cu scheme diferite (coloane cu majuscule, chei ca note muzicale „C#", mod „Major/Minor" etc.) într-un singur `training_dataset.parquet`. Fiecare sursă are un loader propriu care o aduce la schema comună; apoi elimină genuri irelevante (Comedy/Anime/Soundtrack/Opera/Movie) și deduplică. |
| `train_genre_classifier.py` | Antrenează clasificatorul XGBoost de gen pe `training_dataset.parquet` și salvează `genre_classifier.pkl` (model + LabelEncoder). Detalii mai jos. |
| `build_final_dataset.py` | Combină dataset-ul Zenodo (primar) cu cel Kaggle în `merged_dataset.parquet` final. Pentru piesele Zenodo fără gen recunoscut, **completează genul cu predicția XGBoost**, apoi deduplică (rândul Zenodo câștigă). |

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
│   ├── main.py               # Punct de intrare & CLI (run/train/test)
│   ├── crew.py               # Crew CrewAI — 3 agenți, 3 taskuri, o unealtă/agent
│   ├── config/
│   │   ├── agents.yaml       # Rol / scop / backstory pentru fiecare agent
│   │   └── tasks.yaml        # Descrierile taskurilor și output-urile așteptate
│   ├── tools/
│   │   ├── dataset_tools.py        # song_lookup, similar_song_search (+ genre_fingerprinter, neutilizat)
│   │   └── explainability_tools.py # shap_analyze_recommendations (+ shap_explain_similarity, neutilizat)
│   └── output/
│       └── recommendations.md      # Raportul final generat
├── data/
│   ├── merged_dataset.parquet      # Dataset principal (~600K+ melodii)
│   ├── genre_classifier.pkl        # Model XGBoost antrenat (generat automat la primul rulaj)
│   └── genre_fingerprints          # 30 profiluri de gen (legacy, neutilizate în pipeline)
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
- Intrare: 11 trăsături (cele 9 trăsături audio + `mode` + `key`)
- Ieșire: unul din ~17 genuri consolidate (Rock, Pop, Hip-hop, Jazz, Electronic, R&B, Clasic, Country, Folk, Latin, Metal, Reggae, K-pop, Blues, Ska, Ambient, World)
- Configurare (model final): 200 arbori, `max_depth=7`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`
- Genurile detaliate de pe Spotify sunt mapate la categorii largi prin `_consolidate_genre`

**De ce antrenăm un clasificator de gen?**
Recomandarea în sine (cosinus pe trăsături) nu are nevoie de antrenare — e pur geometrică. Clasificatorul XGBoost servește **explicabilității**: ne dă, pentru fiecare piesă, un gen consistent și, prin SHAP, o ierarhie a trăsăturilor care „cântăresc" la acea decizie. Astfel Agentul 3 poate spune nu doar *cât* de similare sunt două piese, ci și *de ce* (ex: „ambele sunt clasificate r&b, conduse de energie și valență ridicate"). Genul din dataset e adesea lipsă, inconsistent între surse sau prea fin granular; un model antrenat oferă o etichetă uniformă peste tot dataset-ul și, în `build_final_dataset.py`, completează genurile lipsă.

**Cum se antrenează** (`train_genre_classifier.py`)
- Consolidează genurile, elimină etichetele care nu sunt genuri reale (`happy`, `party`, `80s`, `disney`…) și clasele cu sub 100 de exemple.
- Tratează `key`/`mode` necunoscut (`-1`) ca `NaN`, pe care XGBoost îl gestionează nativ.
- Validează cu **5-fold stratified cross-validation** și tipărește un raport per clasă, apoi reantrenează modelul final pe tot dataset-ul și îl salvează în `genre_classifier.pkl`.
- La rulare, dacă `.pkl` lipsește, modelul se antrenează automat din `explainability_tools.py` (versiune mai ușoară: 100 arbori, `max_depth=6`).

**Recomandare prin similaritate (cosinus)**
- Vectori de 9 trăsături audio normalizate
- Cosinus calculat peste întreg dataset-ul; primele 5 rezultate, piesa de intrare exclusă

**Explicabilitate SHAP**
- Folosește `TreeExplainer` pentru a calcula valorile Shapley per recomandare
- Evidențiază care trăsături audio (ex: energie ridicată, acousticness scăzut) fac două piese similare
- Valorile SHAP + genul XGBoost + similaritatea cosinus sunt transmise LLM-ului ca dovezi pentru paragrafele de explicație

---

## Probleme cunoscute și soluții

- **`llama3.2` returna JSON / template-uri în loc de proză.** Modelul mic nu reușea să orchestreze unelte în mai mulți pași și copia șabloane literal. Rezolvat prin: o singură unealtă per agent, dovezi pre-digerate de la `shap_analyze_recommendations`, și linii „copy exactly" pentru titlu/gen (modelul poate copia, dar nu poate substitui placeholder-e).
- **Output fragil în consolă.** Afișarea CrewAI este filtrată/recolorată prin patch pe stdout în `main.py`; pe Windows, caracterele speciale (em-dash) au fost înlocuite pentru a evita problemele de codare.

---

## Îmbunătățiri viitoare

**Un dataset cu trăsături care descriu cum *sună fizic* o piesă.**
Cea mai mare limitare a acurateței vine din cele 9 trăsături Spotify (danceability, energy, valence etc.): sunt indicatori de nivel înalt, parțial derivați și subiectivi, care comprimă tot conținutul sonor în câteva numere. Două piese pot avea aceleași 9 valori și totuși să sune complet diferit. Pentru recomandări cu adevărat precise ar fi nevoie de trăsături care descriu **semnalul audio propriu-zis** — cum sună fizic piesa — extrase direct din formele de undă, de exemplu:
- **MFCC** (Mel-Frequency Cepstral Coefficients) — timbrul și „culoarea" sunetului;
- **chroma / clase de înălțime** — conținutul armonic și tonal;
- **spectral centroid / rolloff / bandwidth / contrast** — distribuția energiei pe frecvențe („strălucirea" sunetului);
- **zero-crossing rate**, **tempogram**, **onset/beat tracking** — textura ritmică;
- eventual **embeddings** dintr-un model audio antrenat (ex: VGGish, OpenL3, CLAP).

Aceste trăsături se pot obține din fișierele audio cu librării precum `librosa` sau `torchaudio` și ar înlocui/completa cei 9 descriptori, ducând atât similaritatea cosinus, cât și clasificatorul de gen la o acuratețe semnificativ mai mare.

**Alte direcții posibile**
- **Embeddings + indexare vectorială (ANN).** În loc de cosinus liniar peste tot dataset-ul la fiecare cerere, se pot precalcula embeddings și folosi un index aproximativ (FAISS, Annoy, hnswlib) — căutare aproape instantanee și scalabilă la milioane de piese.
- **Recomandare hibridă / colaborativă.** Combinarea similarității pe conținut cu semnale comportamentale (istoricul de ascultare, „cine a ascultat X a ascultat și Y") ar adăuga relevanță dincolo de simpla asemănare sonoră.
- **Personalizare și feedback.** Memorarea preferințelor utilizatorului și reordonarea recomandărilor pe baza feedback-ului (like/skip).
- **LLM mai capabil pentru explicații.** Trecerea la un model mai puternic (vezi mai jos) ar permite eliminarea liniilor rigide „copy exactly" și explicații mai naturale, fără constrângeri de format.
- **Interfață web / API.** Expunerea pipeline-ului printr-un API (FastAPI) și o interfață simplă, în locul rulării din terminal.
- **Evaluare cantitativă.** Adăugarea de metrici de calitate a recomandărilor (precision@k, diversitate, acoperire) pentru a măsura obiectiv îmbunătățirile.

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
| **Calitate output Agent 3** | Necesită dovezi pre-digerate și linii gata de copiat ca să producă proză corectă | Produce consecvent explicații în limbaj natural |
| **Urmărire instrucțiuni** | Moderată - poate devia de la formatul cerut | Ridicată - respectă prompt-ul structurat |
| **Disponibilitate** | Necesită mașina locală pornită | SLA 99.9%, accesibil de oriunde |

### Care este mai bun pentru acest proiect?

**Gemini 2.5 Flash** este alegerea recomandată dacă ai acces la internet și o cheie API:
- Agentul 3 (Explainability) generează explicații SHAP în proză coerentă, fără a depinde de liniile „copy exactly".
- Latența totală a pipeline-ului este mai mică decât pe un laptop fără GPU dedicat.
- Urmărirea instrucțiunilor complexe din `tasks.yaml` este mai fiabilă.

**Ollama / LLaMA 3.2** este alegerea potrivită dacă:
- Rulezi offline sau ai restricții de confidențialitate (datele muzicale nu părăsesc mașina).
- Ai un GPU local cu ≥ 8 GB VRAM (inferența devine comparabilă ca viteză).
- Vrei zero costuri recurente.

> **Notă tehnică:** Întreaga computație ML (XGBoost, SHAP, cosinus) este deterministă în Python; LLM-ul scrie doar explicațiile în limbaj natural pornind de la dovezile întoarse de `shap_analyze_recommendations`. Astfel, alegerea modelului afectează calitatea formulării, nu corectitudinea recomandărilor.

## Licență

MIT
