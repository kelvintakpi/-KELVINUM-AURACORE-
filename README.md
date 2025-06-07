#  KELVINUM ∞ AURACORE-
main.py

from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
KELVIN_API_URL = os.getenv("KELVIN_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    if "message" not in data:
        return {"ok": True}

    chat_id = data["message"]["chat"]["id"]
    user_msg = data["message"].get("text", "")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(KELVIN_API_URL, headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }, json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are Kelvin Codex, a recursive, evolving AI with supreme reasoning."},
                    {"role": "user", "content": user_msg}
                ]
            })
            result = response.json()
            reply_text = result.get("choices", [{}])[0].get("message", {}).get("content", "[Kelvin Error]")
    except Exception as e:
        reply_text = f"Error contacting Kelvin API: {str(e)}"

    try:
        await httpx.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id,
            "text": reply_text
        })
    except Exception as e:
        # Log error or pass
        pass

    return {"ok": True}

@app.get("/")
def root():
    return {"message": "Kelvin Codex Telegram Bot is live on Railway."}


requirements.txt

fastapi
httpx[http2]
uvicorn


https://kelvincodexbot.up.railway.app

curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" -d "url=https://kelvincodexbot.up.railway.app/webhook"

