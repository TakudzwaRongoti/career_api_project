# main.py
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
with open("career_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define expected columns from training dataset
# Replace this list with all the actual columns used during training
FEATURE_COLUMNS = [
    "Drawing","Dancing","Singing","Sports"	,"Video Game",	"Acting",	"Travelling",	"Gardening"	,"Animals"	,"Photography",	"Teaching",	"Exercise",	"Coding",	"Electricity Components",	"Mechanic Parts"	,"Computer Parts",	"Researching",	"Architecture",	"Historic Collection",	"Botany",	"Zoology",	"Physics",	"Accounting",	"Economics"	,"Sociology"	,"Geography",	"Psycology",	"History",	"Science",	"Bussiness Education"	,"Chemistry"	,"Mathematics",	"Biology",	"Makeup",	"Designing",	"Content writing",	"Crafting",	"Literature",	"Reading",	"Cartooning",	"Debating",	"Asrtology",	"Hindi",	"French"	,"English",	"Urdu",	"Other Language",	"Solving Puzzles",	"Gymnastics",	"Yoga"	,"Engeeniering",	"Doctor",	"Pharmisist",	"Cycling",	"Knitting",	"Director",	"Journalism",	"Bussiness",	"Listening Music"
    # add all remaining one-hot columns from your training dataset
]

app = FastAPI()

# Input model
class StudentProfile(BaseModel):
    interests: list[str]
    hobbies: list[str]
    subject_strengths: list[str]

@app.post("/predict")
def predict(profile: StudentProfile):
    # Initialize feature vector with all 0s
    input_dict = {col: 0 for col in FEATURE_COLUMNS}

    # Set 1 for matching fields
    for field in profile.interests + profile.hobbies + profile.subject_strengths:
        if field in input_dict:
            input_dict[field] = 1  # set feature to active

    # Construct DataFrame
    input_df = pd.DataFrame([input_dict])

    # Make prediction
    prediction = model.predict(input_df)

    return {"predicted_course": prediction[0]}
