# AI-Based Student Success System

Dashboard + risk predictor built from `student success.ipynb`.

```
student-success/
├── docs/                  <- FRONTEND (this folder is what GitHub Pages serves)
│   ├── index.html, style.css, app.js
│   └── data/              <- stats, students, metrics, importance (JSON made by train_export.py)
└── backend/               <- PYTHON API (needed only for the live "Check a student" form)
    ├── pipeline.py        <- your notebook's preprocessing, shared by training and the API
    ├── train_export.py    <- trains RF + XGBoost like the notebook, writes model.joblib and docs/data/*.json
    ├── main.py            <- FastAPI: GET /health, POST /predict
    ├── requirements.txt
    └── data/student-mat.csv
```

Dashboard, tables and charts read static JSON, so they work on GitHub Pages alone.
Only the prediction form needs the backend.

## 1. Run it on your computer
```bash
cd backend
pip install -r requirements.txt
python train_export.py          # only needed if you change the data or model
uvicorn main:app --reload       # API at http://localhost:8000  (docs at /docs)
```
In a second terminal: `cd docs && python -m http.server 5500`, then open http://localhost:5500

## 2. Deploy the frontend (GitHub Pages)
1. Create a GitHub repository and upload this whole project (or `git push` it).
2. Repo **Settings → Pages**. Under *Build and deployment*, choose **Deploy from a branch**, branch `main`, folder **/docs**, then Save.
3. After about a minute your site is live at `https://YOUR-USERNAME.github.io/YOUR-REPO/`.

## 3. Deploy the backend (Render, free tier)
1. Sign in at render.com with GitHub, then **New → Web Service**, and pick your repo.
2. Fill in:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt && python train_export.py`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `PYTHON_VERSION` = `3.12.3`
   - Instance type: Free
3. Deploy. You'll get a URL like `https://student-api.onrender.com`. Check `/health` in your browser.
4. Open `docs/app.js` and change line 2 to `const API_URL = "https://student-api.onrender.com";`, then commit and push. GitHub Pages updates itself.

Free Render services sleep when idle, so the first prediction can take 30-60 seconds. Open the `/health` URL a minute before a demo.
Alternatives: Hugging Face Spaces, Railway or PythonAnywhere. Any host that runs `uvicorn main:app` works.

## Things to know for viewers
- The form takes **raw dataset columns** (G1, G2, study time, failures, absences, support). The backend applies the notebook's mapping into proxy features such as attendance and topic scores. These are proxies, not real measurements, as your notebook notes.
- Risk is **G3 < 10**. G2 is a very strong signal of G3, which is why accuracy is high (~91%, ROC-AUC ~0.98). Say this in your demo.
- Student risk scores include students the model trained on, so treat them as illustrative, not as a fair test. The performance metrics use only the held-out 25%.
