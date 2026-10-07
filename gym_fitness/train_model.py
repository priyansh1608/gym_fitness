from models.ml_recommender import FitnessRecommender


print("Training fitness ML model...")


model = FitnessRecommender()


print("Model training completed.")

print(
    "Saved model at:"
)

print(
    "models/trained_model.pkl"
)