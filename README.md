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

\n## 4. Configure Helping AI (demo version)
\nThe Helping AI chat uses the OpenAI API from the FastAPI backend. The API key must never be placed in `docs/app.js` or committed to GitHub.
\n1. Install the updated backend requirements: `pip install -r requirements.txt`.
\n2. Copy `.env.example` to `.env` inside `backend/` and set `OPENAI_API_KEY` to your own key. Keep `.env` private.
\n3. Optionally set `OPENAI_MODEL` to a model available to your API project.
\n4. Start the API with `uvicorn main:app --reload` from `backend/`. Open `http://localhost:8000/docs` and test `POST /assistant/chat`.
\n5. Configure `OPENAI_API_KEY` and `OPENAI_MODEL` as environment variables on your backend host as well. Redeploy the backend after changing requirements.
\n6. The frontend sends the most recent prediction and its SHAP factors as optional context. Without a prediction, Helping AI answers general study questions.
\n\n**Security limitation:** the student/teacher/admin selector is only a demo mode. It does not authenticate anyone. Do not expose actual student records or cohort information until login, server-side role authorization, and per-student access controls are implemented. Current teacher/admin mode can only answer general questions based on the request context; it does not yet retrieve a real cohort from the database.
\n
## 4. Configure Helping AI (demo version)

The Helping AI chat calls the OpenAI API from the FastAPI backend. Never place an API key in `docs/app.js` or commit it to GitHub.

1. Install the updated backend dependencies: `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` inside `backend/` and set `OPENAI_API_KEY` to your API key. Keep `.env` private.
3. Optionally set `OPENAI_MODEL` to a model available to your API project.
4. Start the API with `uvicorn main:app --reload` from `backend/`. Open `http://localhost:8000/docs` and test `POST /assistant/chat`.
5. Set `OPENAI_API_KEY` and `OPENAI_MODEL` as environment variables on your backend host too. Redeploy the backend after updating requirements.
6. The frontend sends the most recent prediction and SHAP factors as optional context. Without a prediction, Helping AI answers general study questions.

**Security limitation:** the student/teacher/admin selector is a demo convenience, not authentication. Do not use real student records until login, server-side role authorization, and per-student access controls are implemented. Teacher/admin mode currently answers from the submitted request only; it does not retrieve a real student cohort.
