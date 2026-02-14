from fastapi import FastAPI, WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
import cv2
import numpy as np
import json
import asyncio

app = FastAPI()

class GeneratedVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.h, self.w = 480, 640
        
    async def recv(self):
        # Генерируем кадр с нуля
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Ваш алгоритм визуализации
        pixels_y, pixels_x = np.mgrid[100:200:10, 100:400:10]
        frame[pixels_y, pixels_x] = [0, 255, 0]  # Зелёные точки
        
        # Добавляем движущийся круг
        t = time.time()
        cx = int(self.w * 0.5 + 100 * np.cos(t))
        cy = int(self.h * 0.5 + 50 * np.sin(t * 2))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 3)
        
        return frame

pcs = set()

@app.websocket("/offer/{client_id}")
async def offer(websocket: WebSocket, client_id: int):
    await websocket.accept()
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    # Добавляем трек с генерируемым видео
    pc.addTrack(GeneratedVideoTrack())
    
    @pc.on("icecandidate")
    def on_icecandidate(candidate):
        asyncio.create_task(
            websocket.send_text(json.dumps({"candidate": candidate.sdp}))
        )
    
    @pc.on("track")
    def on_track(track):
        print(f"Получен трек {track.kind}")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    await websocket.send_text(json.dumps({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }))
    
    # Получаем ответ от клиента
    data = await websocket.receive_text()
    answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    await pc.setRemoteDescription(answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)