import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import date
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io

st.set_page_config(page_title="FITAI | Smart Fitness Guide", page_icon="💪", layout="wide")

DB_PATH = Path("fitai.db")

# Curated exercise-session catalogue. KNN matches user preferences to sessions.
SESSIONS = pd.DataFrame([
    {"name":"Beginner Full Body","goal":"General fitness","level":"Beginner","minutes":25,"days":3,"equipment":"No equipment","focus":"Squats, wall push-ups, glute bridges, marching"},
    {"name":"Strength Foundations","goal":"Build strength","level":"Beginner","minutes":30,"days":3,"equipment":"No equipment","focus":"Chair squats, incline push-ups, glute bridges, plank"},
    {"name":"Intermediate Strength","goal":"Build strength","level":"Intermediate","minutes":45,"days":4,"equipment":"Optional dumbbells","focus":"Squats, push-ups, lunges, rows, plank"},
    {"name":"Mobility Reset","goal":"Improve mobility","level":"Beginner","minutes":20,"days":3,"equipment":"No equipment","focus":"Cat-cow, hip hinges, thoracic rotations, shoulder circles"},
    {"name":"Daily Mobility Flow","goal":"Improve mobility","level":"Intermediate","minutes":30,"days":5,"equipment":"No equipment","focus":"Dynamic stretches, hip mobility, balance and controlled flows"},
    {"name":"Low-impact Endurance","goal":"Improve endurance","level":"Beginner","minutes":25,"days":4,"equipment":"No equipment","focus":"Marching, step touch, low-impact jacks, walking intervals"},
    {"name":"Endurance Builder","goal":"Improve endurance","level":"Intermediate","minutes":40,"days":5,"equipment":"No equipment","focus":"Brisk walking intervals, step-ups, controlled bodyweight circuit"},
    {"name":"Quick Strength Circuit","goal":"Build strength","level":"Intermediate","minutes":20,"days":3,"equipment":"No equipment","focus":"Squats, incline push-ups, reverse lunges, dead bug"},
    {"name":"Gentle Fitness Starter","goal":"General fitness","level":"Beginner","minutes":15,"days":3,"equipment":"No equipment","focus":"Easy marching, sit-to-stand, wall push-ups, gentle stretching"},
    {"name":"Balanced Fitness","goal":"General fitness","level":"Intermediate","minutes":35,"days":4,"equipment":"No equipment","focus":"Squats, incline push-ups, step-backs, core stability"},
])

GOALS = ["Build strength", "General fitness", "Improve mobility", "Improve endurance"]
LEVELS = ["Beginner", "Intermediate"]

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_date TEXT NOT NULL,
        activity TEXT NOT NULL,
        duration INTEGER NOT NULL,
        notes TEXT
    )""")
    conn.commit()
    return conn

def recommend(goal, level, minutes, days):
    # Encode categorical inputs and scale numeric inputs; KNN finds closest curated plans.
    catalog = SESSIONS.copy()
    X = pd.get_dummies(catalog[["goal", "level"]])
    X["minutes"] = catalog["minutes"]
    X["days"] = catalog["days"]
    query = pd.DataFrame([{"goal": goal, "level": level}])
    q = pd.get_dummies(query)
    X, q = X.align(q, join="outer", axis=1, fill_value=0)
    X["minutes"] = catalog["minutes"]
    X["days"] = catalog["days"]
    q["minutes"] = minutes
    q["days"] = days
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    qs = scaler.transform(q)
    model = NearestNeighbors(n_neighbors=3, metric="euclidean")
    model.fit(Xs)
    distances, indices = model.kneighbors(qs)
    result = catalog.iloc[indices[0]].copy()
    result["match_distance"] = distances[0]
    return result.sort_values("match_distance")

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def analyze_pose(uploaded):
    try:
        import mediapipe as mp
        import cv2
    except Exception as exc:
        return None, f"Pose-analysis dependencies could not load: {exc}"
    image = Image.open(uploaded).convert("RGB")
    arr = np.array(image)
    pose = mp.solutions.pose
    with pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as estimator:
        result = estimator.process(arr)
    if not result.pose_landmarks:
        return None, "No clear full-body pose was detected. Try a well-lit, full-body side-view image."
    lm = result.pose_landmarks.landmark
    # Average left/right hip-knee-ankle angles when visibility is adequate.
    angles = []
    for hip, knee, ankle_idx in [
        (pose.PoseLandmark.LEFT_HIP.value, pose.PoseLandmark.LEFT_KNEE.value, pose.PoseLandmark.LEFT_ANKLE.value),
        (pose.PoseLandmark.RIGHT_HIP.value, pose.PoseLandmark.RIGHT_KNEE.value, pose.PoseLandmark.RIGHT_ANKLE.value)
    ]:
        pts = [lm[hip], lm[knee], lm[ankle_idx]]
        if min(p.visibility for p in pts) >= 0.45:
            angles.append(angle(
                (pts[0].x, pts[0].y),
                (pts[1].x, pts[1].y),
                (pts[2].x, pts[2].y)
            ))
    if not angles:
        return None, "Knee landmarks were not visible enough. Retake the image with your full body in frame."
    knee_angle = float(np.mean(angles))
    annotated = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(
        annotated, result.pose_landmarks, pose.POSE_CONNECTIONS
    )
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return (knee_angle, annotated), None

def load_logs():
    conn = get_db()
    df = pd.read_sql_query("SELECT workout_date, activity, duration, notes FROM workouts ORDER BY workout_date DESC, id DESC", conn)
    conn.close()
    return df

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 85% 0%, #1b3150 0, #0b1020 42%); color:#edf2ff}
section[data-testid="stSidebar"] {background:#111a2e}
div[data-testid="stMetric"] {background:#16233a;padding:14px;border-radius:14px;border:1px solid #293653}
h1,h2,h3 {color:#edf2ff}
div.stButton>button {background:#7cdbb2;color:#07150f;border:0;border-radius:10px;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("💪 FITAI")
st.caption("AI-powered personalized fitness and workout guidance · Academic prototype")

with st.sidebar:
    st.header("Your fitness profile")
    goal = st.selectbox("Fitness goal", GOALS)
    level = st.selectbox("Experience level", LEVELS)
    minutes = st.slider("Available minutes per session", 10, 90, 30, 5)
    days = st.select_slider("Days per week", options=[2,3,4,5,6], value=3)
    st.caption("Recommendations are general educational suggestions, not medical advice.")

tab1, tab2, tab3, tab4 = st.tabs(["✨ Workout planner", "📷 Pose analysis", "📈 Progress tracker", "ℹ️ About"])

with tab1:
    st.subheader("Personalized workout recommendations")
    st.write("FITAI uses a content-based **K-Nearest Neighbors (KNN)** recommender to match your goal, experience, session duration and weekly availability to a curated workout catalogue.")
    if st.button("Generate my plan", key="recommend"):
        recs = recommend(goal, level, minutes, days)
        best = recs.iloc[0]
        st.success(f"Best match: {best['name']}")
        st.markdown(f"**Focus:** {best['focus']}")
        st.markdown(f"**Duration:** {int(best['minutes'])} minutes · **Schedule:** {int(best['days'])} days/week · **Equipment:** {best['equipment']}")
        st.markdown("**Other close matches**")
        st.dataframe(recs[["name","goal","level","minutes","days","equipment"]].reset_index(drop=True), use_container_width=True, hide_index=True)
        st.info("Start with a comfortable intensity, warm up and rest as needed. Stop if you experience pain, dizziness or unusual discomfort.")
    st.divider()
    st.subheader("Example session structure")
    st.write("Warm-up (5 min) → Main exercises (15–30 min) → Cool-down (5 min). Adjust the duration to your current ability.")

with tab2:
    st.subheader("Camera-image pose analysis")
    st.write("Upload a full-body image. MediaPipe Pose detects body landmarks; FITAI estimates a visible knee angle as a computer-vision demonstration.")
    uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg","jpeg","png"])
    if uploaded and st.button("Analyze pose"):
        with st.spinner("Detecting pose landmarks..."):
            output, error = analyze_pose(uploaded)
        if error:
            st.warning(error)
        else:
            knee_angle, annotated = output
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(annotated, caption="Detected pose landmarks", use_container_width=True)
            with col2:
                st.metric("Estimated knee angle", f"{knee_angle:.1f}°")
                st.write("This angle is a geometric measurement from one image, not a complete squat-form score.")
                st.caption("For reliable exercise counting, process a sequence of video frames and use movement-state logic. This prototype does not diagnose posture or prevent injuries.")

with tab3:
    st.subheader("Log and review your workouts")
    with st.form("log_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            workout_date = st.date_input("Date", value=date.today())
        with c2:
            activity = st.selectbox("Activity", ["Strength training","Walking","Yoga / mobility","Cycling","Other"])
        with c3:
            duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=25)
        notes = st.text_input("Notes (optional)", max_chars=150)
        submitted = st.form_submit_button("Save workout")
    if submitted:
        conn = get_db()
        conn.execute("INSERT INTO workouts(workout_date, activity, duration, notes) VALUES(?,?,?,?)",
                     (workout_date.isoformat(), activity, int(duration), notes))
        conn.commit()
        conn.close()
        st.success("Workout saved to the local SQLite database.")
    logs = load_logs()
    if not logs.empty:
        total_minutes = int(logs["duration"].sum())
        sessions = len(logs)
        a, b = st.columns(2)
        a.metric("Logged sessions", sessions)
        b.metric("Total minutes", total_minutes)
        st.dataframe(logs, use_container_width=True, hide_index=True)
        chart = logs.copy()
        chart["workout_date"] = pd.to_datetime(chart["workout_date"])
        daily = chart.groupby("workout_date")["duration"].sum().sort_index()
        st.line_chart(daily)
    else:
        st.info("No workouts logged yet. Add your first session above.")

with tab4:
    st.subheader("Project overview")
    st.markdown("""
    **FITAI** is a student-built AIML prototype with three main modules:
    - **Workout recommendation:** KNN-based matching over a curated workout catalogue.
    - **Pose analysis:** MediaPipe Pose landmark detection and geometric knee-angle estimation from an uploaded image.
    - **Progress tracking:** SQLite-based workout logging and visual summaries.

    **Limitations:** This is not a medical, dietetic or physiotherapy product. Recommendations are general, the exercise catalogue is small, and a single image cannot assess complete movement quality. Do not use it to make health decisions.
    """)
    st.markdown("**Suggested next improvements:** expand and validate the workout catalogue, add a consent-based live camera flow, process video frames for repetition counting, and test usability with volunteers.")

st.caption("FITAI · B.Tech AIML project prototype · Keep exercise safe and appropriate to your abilities.")
