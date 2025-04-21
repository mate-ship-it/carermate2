import os
import tempfile
import subprocess
import logging
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
from transformers import pipeline

# ————— CONFIGURATION —————

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Read and strip environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set or is empty")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set or is empty")

# parse comma‑separated chat IDs (they won’t have newlines, but strip just in case)
raw_ids = os.getenv("AUTHORIZED_CHAT_ID", "").strip()
AUTHORIZED_CHAT_IDS = [
    int(x) for x in raw_ids.split(",") if x.strip().isdigit()
]
if not AUTHORIZED_CHAT_IDS:
    raise RuntimeError("No valid AUTHORIZED_CHAT_IDs provided")

# instantiate OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# initialize the Somali ASR pipeline
# if your model is private, use use_auth_token=HUGGINGFACE_API_TOKEN (also stripped)
somali_asr = pipeline(
    "automatic-speech-recognition",
    model="Mustafaa4a/ASR-Somali",
    # use_auth_token=HUGGINGFACE_API_TOKEN or leave commented if public
)

def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_CHAT_IDS

# ————— HANDLERS —————

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    keyboard = [["Help", "Write", "Record"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Hi! Choose an option:", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")

    text = update.message.text.strip()
    low  = text.lower()

    if low == "help":
        return await update.message.reply_text("🆘 What do you need help with?")
    if low == "write":
        return await update.message.reply_text("✍️ Please type what you'd like me to help you write.")
    if low == "record":
        return await update.message.reply_text("🎙️ Please send a voice message.")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": text}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content)
    except Exception as e:
        logging.exception("Text handler failed")
        await update.message.reply_text(f"⚠️ Error: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")

    # Download the voice note as .ogg
    voice = update.message.voice
    tg_file = await voice.get_file()
    ogg_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    await tg_file.download_to_drive(ogg_tmp.name)

    # Prepare a WAV temp file
    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    try:
        # Convert OGG → WAV via ffmpeg (ensure ffmpeg is installed)
        subprocess.run(
            ["ffmpeg", "-i", ogg_tmp.name, wav_tmp.name, "-y", "-loglevel", "panic"],
            check=True
        )

        # 1) Transcribe with the Somali ASR model
        asr_result = somali_asr(wav_tmp.name)
        somali_text = asr_result.get("text", "").strip()

        # Clean up audio files immediately
        ogg_tmp.close()
        wav_tmp.close()
        os.remove(ogg_tmp.name)
        os.remove(wav_tmp.name)

        # 2) Translate Somali → English using GPT-4o
        translation = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {
                    "role": "user",
                    "content": f"Translate this Somali text into English:\n\n{somali_text}"
                }
            ]
        )

        await update.message.reply_text(translation.choices[0].message.content)

    except Exception as e:
        logging.exception("ASR/translation pipeline failed")
        # remove any leftover files
        for tmp in (ogg_tmp, wav_tmp):
            try:
                tmp.close()
                os.remove(tmp.name)
            except OSError:
                pass
        await update.message.reply_text(f"⚠️ Processing error: {e}")

# ————— BOT SETUP —————

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("🤖 Bot is running…")
    app.run_polling()
