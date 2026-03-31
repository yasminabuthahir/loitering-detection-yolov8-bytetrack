from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Loitering Detection API running"}