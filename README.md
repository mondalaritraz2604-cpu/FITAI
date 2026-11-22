# FITAI — Deployable AIML Project

FITAI is a Streamlit application for a B.Tech AIML project. It includes:
1. A K-Nearest Neighbors (KNN) workout-plan recommender.
2. MediaPipe Pose landmark detection and knee-angle estimation from an uploaded image.
3. SQLite workout logging and progress charts.

## Run locally

Recommended: Python 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Deploy publicly with Streamlit Community Cloud

1. Create a GitHub account/repository.
2. Upload `app.py`, `requirements.txt`, and this README to the repository root.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository, branch, and `app.py`.
5. Deploy. Streamlit will provide a public `https://....streamlit.app` link to share.

## Important deployment note about data

The included SQLite database is local to the running app. On free cloud hosting, local files may be reset when the app restarts or redeploys. For a production or multi-user version, replace SQLite with a managed database such as Supabase or PostgreSQL and add authentication.

## Team responsibilities suggestion

- Member 1: research, requirements, exercise catalogue
- Member 2: KNN recommender and evaluation
- Member 3: MediaPipe pose module
- Member 4: Streamlit UI and SQLite
- Member 5: integration, testing, documentation and deployment

## Responsible-use note

FITAI is an educational prototype, not medical advice or a substitute for a qualified fitness professional. A single image and knee-angle estimate cannot establish correct exercise form or prevent injury. Camera access is not used in this build; pose analysis uses an image the user chooses to upload.
