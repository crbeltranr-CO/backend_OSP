# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

VIDEO_STREAM_URL = "http://localhost/live/"  

async def proxy_video_stream():
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", VIDEO_STREAM_URL) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

@app.get("/video/")
async def video():
    return StreamingResponse(proxy_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")