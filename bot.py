import os
import tempfile
import subprocess
import logging
import asyncio

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

# Strip and validate env vars to avoid header/newline bugs
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
raw_ids = os.getenv("AUTHORIZED_CHAT_ID", "").strip()

if not BOT_TOKEN or not OPENAI_API_KEY or not raw_ids:
    raise RuntimeError("BOT_TOKEN, OPENAI_API_KEY, and AUTHORIZED_CHAT_ID must all be set")

AUTHORIZED_CHAT_IDS = [int(x) for x in raw_ids.split(",") if x.isdigit()]
if not AUTHORIZED_CHAT_IDS:
    raise RuntimeError("No valid AUTHORIZED_CHAT_IDs provided")

# Clients
client = OpenAI(api_key=OPENAI_API_KEY)
# ASR pipeline (weights are cached in the image)
somali_asr = pipeline(
    "automatic-speech-recognition",
    model="Mustafaa4a/ASR-Somali"
)

def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_CHAT_IDS

# ————— HANDLERS —————
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised.")
    kb = [["Help", "Write", "Record"]]
    await update.message.reply_text("Hi! Pick an option:", 
                                    reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised.")
    txt = update.message.text.strip()
    low = txt.lower()
    if low == "help":
        return await update.message.reply_text("🆘 How can I help?")
    if low == "write":
        return await update.message.reply_text("✍️ What would you like me to write?")
    if low == "record":
        return await update.message.reply_text("🎙️ Send me a voice message.")
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": txt}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content)
    except Exception as e:
        logging.exception("Text handling failed")
        await update.message.reply_text(f"⚠️ Error: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised.")

    # Download OGG
    voice = update.message.voice
    tg_file = await voice.get_file()
    ogg_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    await tg_file.download_to_drive(ogg_tmp.name)

    # Convert to WAV
    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", ogg_tmp.name, wav_tmp.name, "-y", "-loglevel", "panic"],
        check=True
    )

    try:
        # 1) ASR (offload to thread so event loop stays responsive)
        asr = await asyncio.to_thread(somali_asr, wav_tmp.name)
        somali_text = asr.get("text", "").strip()

        # Clean up
        for f in (ogg_tmp, wav_tmp):
            f.close()
            os.remove(f.name)

        # 2) Translate via GPT-4o
        trans = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": f"Translate this Somali text into English:\n\n{somali_text}"}
            ]
        )
        await update.message.reply_text(trans.choices[0].message.content)

    except Exception as e:
        logging.exception("Voice handling failed")
        # cleanup on error
        for f in (ogg_tmp, wav_tmp):
            try:
                f.close()
                os.remove(f.name)
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
