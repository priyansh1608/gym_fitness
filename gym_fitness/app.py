from flask import Flask, render_template, request, jsonify

from database.db_handler import (
    init_db,
    save_user,
    save_plan
)

from models.ml_recommender import FitnessRecommender


app = Flask(__name__)

app.config["SECRET_KEY"] = "gym-ai-secret-key"


# Initialize database
init_db()


# Initialize ML model
recommender = FitnessRecommender()


# Shoulder-width / hip-width ratio thresholds (from pose joints).
# These are approximate - tune them after testing with a few people.
RATIO_BROAD = 1.5
RATIO_NARROW = 1.2


def classify_scan(body_scan):
    """Validate scan numbers sent by the browser and describe them."""

    if not isinstance(body_scan, dict):
        return None

    try:
        ratio = float(body_scan["shoulder_hip_ratio"])
        torso_leg = float(body_scan["torso_leg_ratio"])
    except (KeyError, ValueError, TypeError):
        return None

    if not (0.5 <= ratio <= 3.0 and 0.2 <= torso_leg <= 3.0):
        return None

    if ratio >= RATIO_BROAD:
        shape = "Broad shoulders"
        focus = ["Glute bridge - 3 x 12", "Step-ups - 3 x 10"]
        note = "Lower body ko balance karne par focus."

    elif ratio >= RATIO_NARROW:
        shape = "Balanced proportions"
        focus = []
        note = "Upper aur lower body ka balance achha hai."

    else:
        shape = "Wider hips"
        focus = ["Overhead press (light) - 3 x 10",
                 "Lateral raises - 3 x 12"]
        note = "Shoulders aur upper body par extra focus."

    return {
        "shape": shape,
        "shoulder_hip_ratio": round(ratio, 2),
        "torso_leg_ratio": round(torso_leg, 2),
        "focus": focus,
        "note": note
    }



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan")
def scan():
    return render_template("scan.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():

    data = request.get_json()

    try:

        name = data.get("name", "User")
        age = int(data["age"])
        gender = data.get("gender", "Not specified")

        height = float(data["height"])
        weight = float(data["weight"])

        goal = data["goal"]
        activity = data["activity"]
        food = data["food"]

    except (KeyError, ValueError, TypeError):

        return jsonify({
            "error": "Invalid user data"
        }), 400


    # -------------------------
    # BMI CALCULATION
    # -------------------------

    height_meter = height / 100

    bmi = weight / (height_meter ** 2)

    bmi = round(bmi, 2)


    # -------------------------
    # ML RECOMMENDATION
    # -------------------------

    result = recommender.recommend(

        age=age,
        height=height,
        weight=weight,
        goal=goal,
        activity=activity,
        food=food
    )


    # -------------------------
    # BODY SCAN (optional)
    # -------------------------

    scan = classify_scan(data.get("body_scan"))

    if scan:

        # skip exercises that are already in the plan
        have = {w.split(" - ")[0].lower() for w in result["workout"]}

        scan["focus"] = [
            e for e in scan["focus"]
            if e.split(" - ")[0].lower() not in have
        ]

        for exercise in scan["focus"]:
            result["workout"].append("Focus: " + exercise)

        result["workout_text"] = "\n".join(result["workout"])

        result["scan"] = scan


    # -------------------------
    # SAVE USER
    # -------------------------

    user_id = save_user(

        name=name,
        age=age,
        gender=gender,
        height=height,
        weight=weight,
        goal=goal,
        activity=activity,
        food=food,
        bmi=bmi
    )


    # -------------------------
    # SAVE PLAN
    # -------------------------

    save_plan(

        user_id=user_id,

        workout_plan=result["workout_text"],

        diet_plan=result["diet_text"],

        calories=result["calories"],

        protein=result["protein"]
    )


    # -------------------------
    # RESPONSE
    # -------------------------

    result["user_id"] = user_id

    result["bmi"] = bmi

    result["name"] = name

    return jsonify(result)


# -------------------------
# START APPLICATION
# -------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )