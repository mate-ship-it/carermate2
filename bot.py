import os
import tempfile
import time
import aiohttp
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTAPI_URL = os.getenv("FASTAPI_URL")  # e.g. https://your-domain.com

# Parse authorized chat IDs
raw_ids = os.getenv("AUTHORIZED_CHAT_ID", "")
AUTHORIZED_CHAT_IDS = [int(x) for x in raw_ids.split(",") if x.strip().isdigit()]

# Initialize OpenAI Client (for fallback translation)
client = OpenAI(api_key=OPENAI_API_KEY)

# Authorization Check
def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_CHAT_IDS

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text(
            "❌ You are not authorised to use this bot."
        )
    keyboard = [["Help", "Write", "Record"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Hi! Please choose an option below:", reply_markup=reply_markup
    )

# Handle text messages (direct GPT fallback)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    text = update.message.text.strip()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant. Only return the direct English translation, no explanation."},
                {"role": "user", "content": f"Translate this Somali text into English:\n\n{text}"}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"⚠️ Text Error: {e}")
        await update.message.reply_text("⚠️ Error while processing your message.")

# Handle voice messages with polling
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    voice = update.message.voice or update.message.voice_note if hasattr(update.message, 'voice_note') else None
    if not voice:
        return await update.message.reply_text("⚠️ No voice message detected.")

    # Download voice file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as f:
        audio_path = f.name
        file = await context.bot.get_file(voice.file_id)
        await file.download_to_drive(custom_path=audio_path)

    try:
        # Enqueue transcription
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            with open(audio_path, 'rb') as audio_file:
                form = aiohttp.FormData()
                form.add_field('file', audio_file, filename=os.path.basename(audio_path), content_type='audio/ogg')
                async with session.post(f"{FASTAPI_URL}/transcribe", data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(f"FastAPI error {resp.status}: {text}")
                    data = await resp.json()
        task_id = data.get('task_id')
        if not task_id:
            raise Exception(f"No task_id in response: {data}")

        # Poll for status
        status = None
        result = None
        for _ in range(30):  # retry up to ~30s
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{FASTAPI_URL}/status/{task_id}") as sresp:
                    if sresp.status != 200:
                        text = await sresp.text()
                        raise Exception(f"Status error {sresp.status}: {text}")
                    status_data = await sresp.json()
            status = status_data.get('status')
            if status == 'done':
                result = status_data.get('result', {})
                break
            if status == 'failure':
                error_msg = status_data.get('error', 'Unknown')
                raise Exception(f"Task failed: {error_msg}")
            await asyncio.sleep(1)

        if status != 'done':
            return await update.message.reply_text("⚠️ Transcription timed out.")

        english = result.get('english') or result.get('translation')
        if not english:
            raise Exception(f"No translation in result: {result}")

        await update.message.reply_text(english)

    except Exception as e:
        print(f"⚠️ Voice Error: {e}")
        await update.message.reply_text(f"⚠️ Error: {e}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Bot startup
def main():
    if not all([BOT_TOKEN, OPENAI_API_KEY, FASTAPI_URL, AUTHORIZED_CHAT_IDS]):
        print("🚨 Missing required environment variables or authorized IDs.")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
