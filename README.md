# PaanaAI — Deployment Guide

## Project Structure
```
paanaai/
├── frontend/
│   └── index.html          ← Complete frontend (no build needed)
└── backend/
    ├── main.py             ← FastAPI server
    └── requirements.txt
```

---

## Backend Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Place your trained model
Copy `RoleDetectionModel.pth` from your Google Drive into the `backend/` folder:
```bash
# Download from Google Drive, then:
cp /path/to/RoleDetectionModel.pth backend/RoleDetectionModel.pth
```

Or set the `MODEL_PATH` environment variable to point elsewhere:
```bash
export MODEL_PATH=/path/to/RoleDetectionModel.pth
```

### 3. Start the server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- Health check: http://localhost:8000/api/health
- API docs: http://localhost:8000/docs

---

## Frontend Setup

No build step needed — it's a single HTML file.

**Option A — Open directly in browser:**
```
Open frontend/index.html in your browser
```

**Option B — Serve with Python (recommended for file uploads):**
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

## Features

### Player Role Detection
- Upload any player heatmap image (PNG/JPG)
- CNN model classifies into 9 roles:
  - Attacking Midfielder, Central Midfielder, Centre Back
  - Defensive Midfielder, Left Back, Left Winger
  - Right Back, Right Winger, Striker
- Returns Leader/Learner classification + confidence scores for all roles
- **Demo mode** works even without model file (shows random results)

### Best Pass System
- Interactive 2D football pitch (bird's eye view)
- Drag any player to reposition
- Click a blue (Team A) player to set as ball carrier
- 5 Tactical Modes:
  - 🛡️ Ultra Defensive (mode 0)
  - 🔒 Defensive (mode 1)
  - ⚖️ Balanced (mode 2)
  - ⚔️ Attacking (mode 3)
  - 🔥 Ultra Attacking (mode 4)
- Ground pass = solid green arrow
- Lofted pass = curved gold dashed arrow
- xT (Expected Threat) based scoring engine

---

## Model Note

If `RoleDetectionModel.pth` is not found, the backend runs in **demo mode**
and returns realistic-looking random predictions. Everything else works normally.

To load from Google Drive in a server environment, use `gdown`:
```bash
pip install gdown
gdown --id YOUR_GDRIVE_FILE_ID -O backend/RoleDetectionModel.pth
```

---

## Deployment (Production)

For public hosting:

**Backend → Railway / Render / HuggingFace Spaces:**
```bash
# Add to requirements.txt if deploying on cloud:
gunicorn
# Start command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Frontend → Netlify / Vercel / GitHub Pages:**
- Just upload `frontend/index.html`
- Update `const API = 'https://your-backend-url.com'` in the HTML

**Update CORS in main.py** for production:
```python
allow_origins=["https://your-frontend-domain.com"]
```
