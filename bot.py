import os
import tempfile
import subprocess
import logging
import aiohttp

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from openai import OpenAI

# ————— CONFIGURATION & LOGGING —————
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Environment variables
BOT_TOKEN   = os.getenv("BOT_TOKEN", "").strip()
OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
HF_TOKEN    = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
AUTH_RAW    = os.getenv("AUTHORIZED_CHAT_ID", "").strip()

if not BOT_TOKEN or not OPENAI_KEY or not HF_TOKEN or not AUTH_RAW:
    logging.error("Missing required environment variables")
    raise SystemExit("Configuration error")

AUTHORIZED_IDS = {int(x) for x in AUTH_RAW.split(",") if x.isdigit()}
if not AUTHORIZED_IDS:
    logging.error("No valid AUTHORIZED_CHAT_IDs provided")
    raise SystemExit("Configuration error")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

# Hugging Face ASR Inference API settings
HF_ASR_URL = "https://api-inference.huggingface.co/models/Mustafaa4a/ASR-Somali"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ————— HELPERS —————
def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_IDS

async def hf_asr(audio_path: str) -> str:
    """Call HF Inference API for ASR and return the transcribed text."""
    async with aiohttp.ClientSession(headers=HF_HEADERS) as session:
        with open(audio_path, "rb") as f:
            data = f.read()
        resp = await session.post(HF_ASR_URL, data=data)
        resp.raise_for_status()
        result = await resp.json()
    return result.get("text", "")

# ————— HANDLERS —————
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")
    kb = ReplyKeyboardMarkup([["Help","Write","Record"]], resize_keyboard=True)
    await update.message.reply_text("Hi! Choose an option:", reply_markup=kb)
    logging.info(f"User {update.effective_chat.id} started bot")

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")
    text = update.message.text.strip()
    low = text.lower()
    if low == "help":
        return await update.message.reply_text("🆘 How can I assist?")
    if low == "write":
        return await update.message.reply_text("✍️ What shall I draft?")
    if low == "record":
        return await update.message.reply_text("🎙️ Please send a voice note.")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user",  "content":text}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content)
    except Exception as e:
        logging.exception("Error processing text message")
        await update.message.reply_text(f"⚠️ Error: {e}")

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")

    tmp_ogg = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        # Download OGG from Telegram
        tg_file = await update.message.voice.get_file()
        await tg_file.download_to_drive(tmp_ogg.name)

        # Convert to WAV
        subprocess.run(
            ["ffmpeg","-i",tmp_ogg.name,tmp_wav.name,"-y","-loglevel","panic"],
            check=True
        )

        # 1) ASR via HF Inference API
        somali_text = await hf_asr(tmp_wav.name)

        # 2) Translate via OpenAI GPT-4o
        translation = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"Translate Somali to English."},
                {"role":"user",  "content":somali_text}
            ]
        )
        eng_text = translation.choices[0].message.content

        # Reply
        await update.message.reply_text(f"🇸🇴 {somali_text}\n\n🇬🇧 {eng_text}")
    except Exception as e:
        logging.exception("Error during voice handling")
        await update.message.reply_text(f"⚠️ Error: {e}")
    finally:
        for f in (tmp_ogg, tmp_wav):
            try:
                os.remove(f.name)
            except OSError:
                pass

# ————— MAIN —————
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logging.info("Bot starting…")
    app.run_polling()
