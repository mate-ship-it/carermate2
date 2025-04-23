import os
import tempfile
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
FASTAPI_URL = "https://j89w80nvhw0tyy-8000.proxy.runpod.net/transcribe"

# Parse authorized chat IDs
raw_ids = os.getenv("AUTHORIZED_CHAT_ID", "")
AUTHORIZED_CHAT_IDS = [int(x) for x in raw_ids.split(",") if x.strip().isdigit()]

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Authorization Check
def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_CHAT_IDS

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    keyboard = [["Help", "Write", "Record"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Hi! Please choose an option below:", reply_markup=reply_markup)

# Handle text messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    text = update.message.text.strip().lower()

    if text == "help":
        return await update.message.reply_text("🆘 What do you need help with?")
    elif text == "write":
        return await update.message.reply_text("✍️ Please type what you'd like me to help you write.")
    elif text == "record":
        return await update.message.reply_text("🎙️ Please send a voice message.")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Text Error: {e}")
        await update.message.reply_text("⚠️ Error while processing your message.")

# Handle voice messages
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    voice = update.message.voice
    if not voice:
        return await update.message.reply_text("⚠️ No voice message detected.")

    file = await context.bot.get_file(voice.file_id)

    audio_path = ""
    try:
        # Download to temp .ogg
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as f:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(file.file_path) as resp:
                    f.write(await resp.read())
            audio_path = f.name

        # Send to FastAPI for transcription
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            with open(audio_path, "rb") as audio_file:
                form = aiohttp.FormData()
                form.add_field('file', audio_file, filename="voice.ogg", content_type='audio/ogg')
                async with session.post(FASTAPI_URL, data=form) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"FastAPI Error {resp.status}: {error_text}")
                    fastapi_result = await resp.json()
                    somali_text = fastapi_result.get("transcription", "")

        if not somali_text:
            return await update.message.reply_text("⚠️ Could not transcribe the voice message.")

        # Translate via GPT-4
        translation = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": f"Translate this Somali text into English:\n\n{somali_text}"}
            ]
        )
        await update.message.reply_text(translation.choices[0].message.content)

    except Exception as e:
        print(f"⚠️ Voice Error: {e}")
        await update.message.reply_text(f"⚠️ Error: {e}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Main app runner
if __name__ == "__main__":
    if not BOT_TOKEN or not OPENAI_API_KEY or not AUTHORIZED_CHAT_IDS:
        print("🚨 Missing env vars or no authorized chat IDs set.")
        exit(1)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("🤖 Bot is running... Make sure only one instance is active.")
    app.run_polling(close_loop=False)
