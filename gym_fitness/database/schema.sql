CREATE TABLE IF NOT EXISTS users (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT NOT NULL,

    age INTEGER NOT NULL,

    gender TEXT,

    height REAL NOT NULL,

    weight REAL NOT NULL,

    goal TEXT NOT NULL,

    activity_level TEXT NOT NULL,

    food_preference TEXT NOT NULL,

    bmi REAL NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS plans (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER NOT NULL,

    workout_plan TEXT NOT NULL,

    diet_plan TEXT NOT NULL,

    calories INTEGER,

    protein REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
);