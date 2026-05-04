from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from feature_engineering import FeatureEngineer

app = FastAPI()


MODEL_PATH = "house_price_model_rf.pkl"

model = joblib.load(MODEL_PATH)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={
        "form_data": {}
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form = await request.form()
    data = dict(form)

    # 🔹 Minimal conversion ONLY
    data['area_sqft'] = float(data['area_sqft'])
    data['bedrooms'] = int(data['bedrooms'])
    data['bathrooms'] = int(data['bathrooms'])
    data['floors'] = int(data['floors'])
    data['year_built'] = int(data['year_built'])
    data['distance_to_city_km'] = float(data['distance_to_city_km'])
    data['school_rating'] = float(data['school_rating'])

    # Yes/No → 0/1
    data['garage'] = 1 if data['garage'] == "Yes" else 0
    data['renovated'] = 1 if data['renovated'] == "Yes" else 0

    df = pd.DataFrame([data])

    #  No feature engineering here
    prediction = model.predict(df)[0]

    return templates.TemplateResponse(request=request, name="index.html", context={
        "prediction": round(prediction, 2),
        "form_data": data
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)