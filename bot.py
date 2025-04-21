import os
import tempfile
import subprocess
import logging
import asyncio

from telegram import Update, ReplyKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline  # Ensure transformers and sentencepiece are installed

# ————— CONFIGURATION & LOGGING —————
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Validate environment
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AUTH_RAW = os.getenv("AUTHORIZED_CHAT_ID", "").strip()
if not BOT_TOKEN or not OPENAI_KEY or not AUTH_RAW:
    logging.error("Missing BOT_TOKEN, OPENAI_API_KEY, or AUTHORIZED_CHAT_ID")
    raise SystemExit("Configuration error")
AUTHORIZED_IDS = {int(x) for x in AUTH_RAW.split(",") if x.isdigit()}
if not AUTHORIZED_IDS:
    logging.error("No valid AUTHORIZED_CHAT_IDs")
    raise SystemExit("Configuration error")

# Initialize clients and pipelines
openai = OpenAI(api_key=OPENAI_KEY)
# ASR pipeline (pre-cached in Docker build)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="Mustafaa4a/ASR-Somali"
)
# Translation pipeline requires sentencepiece>=0.1.97
ttranslator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-cus-en"
)

# ————— HELPERS —————
async def run_in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_IDS

# ————— HANDLERS —————
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")
    kb = ReplyKeyboardMarkup([["Help","Write","Record"]], resize_keyboard=True)
    await update.message.reply_text("Choose an option:", reply_markup=kb)
    logging.info(f"Authorized start by {update.effective_chat.id}")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")
    text = update.message.text.strip()
    low = text.lower()
    if low in ("help","write","record"):
        prompts = {"help":"🆘 How can I assist?","write":"✍️ What should I draft?","record":"🎙️ Please send a voice note."}
        return await update.message.reply_text(prompts[low])

    # Send typing action
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        resp = await run_in_thread(
            openai.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":text}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content)
    except Exception as e:
        logging.exception("Error in text handler")
        await update.message.reply_text(f"⚠️ Error: {e}")

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")

    ogg = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        # Download .ogg
        tg_file = await update.message.voice.get_file()
        await tg_file.download_to_drive(ogg.name)
        # Convert to WAV
        await run_in_thread(
            subprocess.run,
            ["ffmpeg","-i",ogg.name,wav.name,"-y","-loglevel","panic"],
            check=True
        )
        # ASR
        asr_res = await run_in_thread(asr_pipeline, wav.name)
        somali_text = asr_res.get("text","<no transcription>")
        # Translate
        trans_res = await run_in_thread(tttranslator, somali_text)
        english = trans_res[0].get("translation_text","<no translation>")
        # Reply
        await update.message.reply_text(f"🇸🇴 {somali_text}\n\n🇬🇧 {english}")
    except Exception as e:
        logging.exception("Error in voice handler")
        await update.message.reply_text(f"⚠️ Error: {e}")
    finally:
        for f in (ogg,wav):
            try: os.remove(f.name)
            except: pass

# ————— RUN —————
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logging.info("Bot starting…")
    app.run_polling()
