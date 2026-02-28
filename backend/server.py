from fastapi import FastAPI, WebSocket
from database import SessionLocal
from models import YogaSession
from datetime import datetime
import json

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    db = SessionLocal()

    while True:
        data = await websocket.receive_text()
        yoga_data = json.loads(data)

        session = YogaSession(
            user_id=yoga_data["user_id"],
            pose_name=yoga_data["pose_name"],
            duration=yoga_data["duration"],
            accuracy=yoga_data["accuracy"],
            calories=yoga_data["calories"],
            timestamp=datetime.now()
        )

        db.add(session)
        db.commit()

        await websocket.send_text("Saved to DB")