from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import numpy as np

from models import get_db, HealthData

# Initialize FastAPI
app = FastAPI(title="Mental Health Risk Predictor")

# Input schema
class HealthDataInput(BaseModel):
    sleep_hours: float
    exercise_hours: float
    stress_level: int
    social_activity: int
    work_hours: float
    screen_time: float

# POST endpoint for prediction
@app.post("/predict")
def predict_health_risk(data: HealthDataInput, db: Session = Depends(get_db)):
    try:
        # Rule-based logic
        risk_scores = {"High": 0, "Medium": 0, "Low": 0}

        # sleep_hours
        if data.sleep_hours < 3:
            risk_scores["High"] += 1
        elif data.sleep_hours < 5:
            risk_scores["Medium"] += 1
        else:
            risk_scores["Low"] += 1

        # exercise_hours
        if data.exercise_hours < 0.5:
            risk_scores["High"] += 1
        elif data.exercise_hours < 1.5:
            risk_scores["Medium"] += 1
        else:
            risk_scores["Low"] += 1

        # stress_level
        if data.stress_level > 8:
            risk_scores["High"] += 1
        elif data.stress_level > 5:
            risk_scores["Medium"] += 1
        else:
            risk_scores["Low"] += 1

        # social_activity
        if data.social_activity < 1:
            risk_scores["High"] += 1
        elif data.social_activity < 3:
            risk_scores["Medium"] += 1
        else:
            risk_scores["Low"] += 1

        # work_hours
        if data.work_hours > 12:
            risk_scores["High"] += 1
        elif data.work_hours > 8:
            risk_scores["Medium"] += 1
        else:
            risk_scores["Low"] += 1

        # screen_time
        if data.screen_time > 8:
            risk_scores["High"] += 1
        elif data.screen_time > 5:
            risk_scores["Medium"] += 1
        else:
            risk_scores["Low"] += 1

        # Final prediction based on majority vote
        prediction = max(risk_scores, key=risk_scores.get)

        # Save to database
        db_record = HealthData(
            sleep_hours=data.sleep_hours,
            exercise_hours=data.exercise_hours,
            stress_level=data.stress_level,
            social_activity=data.social_activity,
            work_hours=data.work_hours,
            screen_time=data.screen_time,
            prediction=prediction
        )

        db.add(db_record)
        db.commit()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET endpoint to view history
@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    records = db.query(HealthData).order_by(HealthData.TimeStamp.desc()).limit(12).all()
    return records
