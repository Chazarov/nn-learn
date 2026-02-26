import uvicorn
from fastapi import FastAPI

from api.handlers import router

app = FastAPI(title="Multilayer Perceptron API")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
