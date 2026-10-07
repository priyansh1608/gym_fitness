import os

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

import joblib

from config import MODEL_PATH


class FitnessRecommender:

    def __init__(self):

        self.goal_encoder = LabelEncoder()

        self.activity_encoder = LabelEncoder()


        # -------------------------
        # ENCODERS
        # -------------------------

        self.goal_encoder.fit([
            "Weight Loss",
            "Muscle Gain",
            "General Fitness"
        ])


        self.activity_encoder.fit([
            "Beginner",
            "Intermediate",
            "Advanced"
        ])


        # -------------------------
        # LOAD OR TRAIN MODEL
        # -------------------------

        if os.path.exists(MODEL_PATH):

            self.model = joblib.load(
                MODEL_PATH
            )

        else:

            self.model = self.train_model()


    def create_training_data(self):

        X = []

        y = []


        goals = [
            "Weight Loss",
            "Muscle Gain",
            "General Fitness"
        ]


        activities = [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]


        for age in range(18, 61, 5):

            for bmi in [
                18,
                21,
                24,
                27,
                30,
                33
            ]:

                for goal in goals:

                    for activity in activities:

                        goal_id = self.goal_encoder.transform(
                            [goal]
                        )[0]


                        activity_id = self.activity_encoder.transform(
                            [activity]
                        )[0]


                        X.append([
                            age,
                            bmi,
                            goal_id,
                            activity_id
                        ])


                        # Training labels
                        if activity == "Beginner":

                            label = "Beginner Full Body"

                        elif goal == "Weight Loss":

                            label = "Cardio + Strength"

                        elif goal == "Muscle Gain":

                            label = "Strength Training"

                        else:

                            label = "Balanced Fitness"


                        y.append(label)


        return np.array(X), np.array(y)


    def train_model(self):

        X, y = self.create_training_data()


        model = RandomForestClassifier(

            n_estimators=100,

            random_state=42,

            max_depth=6
        )


        model.fit(X, y)


        # Save trained model

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        joblib.dump(
            model,
            MODEL_PATH
        )


        return model


    def calculate_calories(
        self,
        age,
        height,
        weight,
        activity
    ):

        # Educational estimate
        # Mifflin-style calculation

        bmr = (
            10 * weight
            + 6.25 * height
            - 5 * age
            + 5
        )


        activity_multiplier = {

            "Beginner": 1.35,

            "Intermediate": 1.50,

            "Advanced": 1.65

        }


        multiplier = activity_multiplier.get(
            activity,
            1.40
        )


        calories = bmr * multiplier


        return int(round(calories))


    def recommend(
        self,
        age,
        height,
        weight,
        goal,
        activity,
        food
    ):

        # -------------------------
        # BMI
        # -------------------------

        height_meter = height / 100

        bmi = weight / (
            height_meter ** 2
        )


        # -------------------------
        # ENCODE FEATURES
        # -------------------------

        goal_id = self.goal_encoder.transform(
            [goal]
        )[0]


        activity_id = self.activity_encoder.transform(
            [activity]
        )[0]


        features = np.array([[
            age,
            bmi,
            goal_id,
            activity_id
        ]])


        # -------------------------
        # ML PREDICTION
        # -------------------------

        prediction = self.model.predict(
            features
        )[0]


        # -------------------------
        # CALORIES
        # -------------------------

        calories = self.calculate_calories(

            age,
            height,
            weight,
            activity
        )


        # -------------------------
        # WORKOUT
        # -------------------------

        if goal == "Weight Loss":

            calories -= 300


            workout = [

                "Brisk walking - 20 to 30 minutes",

                "Bodyweight squats - 3 x 10",

                "Push-ups - 3 x 8",

                "Plank - 3 x 20 seconds",

                "Light stretching - 5 to 10 minutes"

            ]


        elif goal == "Muscle Gain":

            calories += 250


            workout = [

                "Squats - 3 x 10",

                "Push-ups / Chest Press - 3 x 10",

                "Rows - 3 x 10",

                "Lunges - 3 x 10",

                "Plank - 3 x 30 seconds"

            ]


        else:

            workout = [

                "Walking/Cycling - 20 minutes",

                "Squats - 3 x 10",

                "Push-ups - 3 x 8",

                "Glute bridge - 3 x 12",

                "Plank - 3 x 20 seconds"

            ]


        # -------------------------
        # PROTEIN
        # -------------------------

        if goal == "Muscle Gain":

            protein = weight * 1.6

        else:

            protein = weight * 1.2


        protein = round(
            protein,
            1
        )


        # -------------------------
        # DIET
        # -------------------------

        diet = self.generate_diet(
            food,
            goal
        )


        return {

            "model_class": prediction,

            "calories": calories,

            "protein": protein,

            "workout": workout,

            "diet": diet,

            "workout_text": "\n".join(
                workout
            ),

            "diet_text": "\n".join(
                diet
            )
        }


    def generate_diet(
        self,
        food,
        goal
    ):

        if food == "Veg":

            meals = [

                "Breakfast: Oats + milk/curd + fruit",

                "Lunch: Dal + rice/roti + vegetables",

                "Snack: Fruit + roasted chana/nuts",

                "Dinner: Paneer/tofu + vegetables"

            ]


        elif food == "Vegan":

            meals = [

                "Breakfast: Oats + soy milk + fruit",

                "Lunch: Dal/chickpeas + rice/roti",

                "Snack: Fruit + nuts",

                "Dinner: Tofu/soy chunks + vegetables"

            ]


        else:

            meals = [

                "Breakfast: Eggs + oats + fruit",

                "Lunch: Chicken/fish + rice/roti + vegetables",

                "Snack: Fruit + curd/nuts",

                "Dinner: Eggs/chicken + salad"

            ]


        if goal == "Weight Loss":

            meals.append(
                "Focus on sensible portions and minimally processed foods."
            )


        elif goal == "Muscle Gain":

            meals.append(
                "Include protein across meals and adequate calories."
            )


        else:

            meals.append(
                "Keep meals balanced with protein, vegetables and whole grains."
            )


        return meals