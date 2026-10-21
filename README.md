# AI-Based Student Success System

Dashboard and student-risk predictor built from the Student Performance dataset.

## Project structure

```
docs/                  # Static frontend served by GitHub Pages
  index.html
  style.css
  app.js
  data/                # Dashboard JSON data
backend/               # FastAPI prediction and Helping AI APIs
  main.py
  ai_assistant.py      # Gemini-powered study assistant
  pipeline.py
  train_export.py
  requirements.txt
  .env.example
  data/student-mat.csv
```

The dashboard's tables and charts read static JSON and can run on GitHub Pages. The live prediction form and Helping AI chat require the backend.

## 1. Run locally

From the repository root:

```bash
cd backend
pip install -r requirements.txt
# Optional: retrain and regenerate the model/data if needed
python train_export.py
uvicorn main:app --reload
```

In a second terminal, serve the frontend:

```bash
cd docs
python -m http.server 5500
```

Open `http://localhost:5500`. The default frontend API URL is configured in `docs/app.js`; for local use, set it to `http://127.0.0.1:8000`.

## 2. Configure Helping AI with Gemini

Helping AI uses Google's Gemini API through the FastAPI backend. **Never put the Gemini API key in frontend JavaScript or commit it to GitHub.**

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. In the `backend` folder, copy `.env.example` to `.env`.
3. Set your key in `backend/.env`:
   ```dotenv
   GEMINI_API_KEY=your_actual_api_key
   GEMINI_MODEL=gemini-2.5-flash
   ```
4. Install dependencies with `pip install -r requirements.txt`.
5. Start the API from `backend` using `uvicorn main:app --reload`.
6. Visit `http://127.0.0.1:8000/docs` and test `POST /assistant/chat`.

For deployment, set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` in the backend host's environment-variable settings, then redeploy. Do not upload your local `.env` file.

The chat can answer general study questions. After a prediction is made, the frontend also sends its risk score and SHAP factors as optional context to help personalize study suggestions. Without a prediction, the assistant has no student-specific context.

## 3. Deploy the frontend on GitHub Pages

1. Push the project to GitHub.
2. Open **Settings → Pages**.
3. Choose **Deploy from a branch**, select `main` and the `/docs` folder, then save.
4. Set `API_URL` in `docs/app.js` to your deployed backend URL.

## 4. Deploy the backend on Render

1. Create a new Web Service and select this repository.
2. Set **Root Directory** to `backend`.
3. Use build command: `pip install -r requirements.txt && python train_export.py`.
4. Use start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Add `GEMINI_API_KEY` and optionally `GEMINI_MODEL` in the service's environment settings.
6. Deploy and check the `/health` endpoint. Make sure `API_URL` in `docs/app.js` points to this service.

Free hosting may sleep while idle, so the first request can take longer while the backend starts.

## Important limitations

- Risk is defined as a predicted final grade `G3 < 10` on a 0–20 scale. A prediction is an estimate, not a certainty.
- Some dashboard features are derived proxies, not direct measurements.
- Cohort risk scores may include students used during model training; treat them as illustrative. Reported model metrics are computed on the held-out test split.
- Student/teacher/admin selection in Helping AI is a demo mode, **not authentication**. Teacher/admin responses use only the submitted question and optional context; they do not query a real student database. Do not use sensitive real student records until login, server-side authorization, and access controls are implemented.
- AI responses may be inaccurate. SHAP factors describe model associations, not proven causes.
