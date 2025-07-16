# fastapi_app.py
from fastapi import FastAPI, Query
from deepfake_detection import evaluate_video  # your script
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/analyze")
def analyze(url: str):
    model_paths = {
        "cnn": "D:/Deepfake_Detection/models/best_model.pth",
        "gnn": "D:/Deepfake_Detection/models/best_heterognn_model1.pt",
        "sentiment": "D:/Deepfake_Detection/models/sentiment_model_gb1.pkl",
        "bot": "D:/Deepfake_Detection/models/bot_model_tuned1.pkl",
        "feature_list_path": "D:/Deepfake_Detection/models/sentiment_model_features.pkl",
        "bot_feature_list_path": "D:/Deepfake_Detection/models/bot_model_features.pkl"
    }

    results, *_ = evaluate_video(url, model_paths)
    return results

@app.get("/")
def root():
    return {"message": "Deepfake detection API is running."}

@app.get("/favicon.ico")
def favicon():
    return {}

