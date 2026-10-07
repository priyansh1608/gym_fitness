import sqlite3

from config import DATABASE


SCHEMA_PATH = "database/schema.sql"


def get_connection():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    connection = get_connection()

    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        schema = file.read()

    connection.executescript(schema)

    connection.commit()

    connection.close()


def save_user(
    name,
    age,
    gender,
    height,
    weight,
    goal,
    activity,
    food,
    bmi
):

    connection = get_connection()

    cursor = connection.execute(
        """
        INSERT INTO users
        (
            name,
            age,
            gender,
            height,
            weight,
            goal,
            activity_level,
            food_preference,
            bmi
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            age,
            gender,
            height,
            weight,
            goal,
            activity,
            food,
            bmi
        )
    )

    user_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return user_id


def save_plan(
    user_id,
    workout_plan,
    diet_plan,
    calories,
    protein
):

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO plans
        (
            user_id,
            workout_plan,
            diet_plan,
            calories,
            protein
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            workout_plan,
            diet_plan,
            calories,
            protein
        )
    )

    connection.commit()

    connection.close()