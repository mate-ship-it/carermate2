import os
import tempfile
import subprocess
import logging
import asyncio

from telegram import Update, ChatAction
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

# ————— CONFIG —————
load_dotenv()
logging.basicConfig(level=logging.INFO)

BOT_TOKEN    = os.getenv("BOT_TOKEN", "").strip()
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
AUTH_RAW     = os.getenv("AUTHORIZED_CHAT_ID", "").strip()
if not BOT_TOKEN or not OPENAI_KEY or not AUTH_RAW:
    raise SystemExit("Missing BOT_TOKEN, OPENAI_API_KEY, or AUTHORIZED_CHAT_ID")

AUTHORIZED_IDS = {int(x) for x in AUTH_RAW.split(",") if x.isdigit()}
if not AUTHORIZED_IDS:
    raise SystemExit("No valid AUTHORIZED_CHAT_IDs")

# API clients & pipelines
openai = OpenAI(api_key=OPENAI_KEY)
somali_asr = pipeline("automatic-speech-recognition", model="Mustafaa4a/ASR-Somali")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-cus-en")

def authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_IDS

# ————— HELPERS —————
async def run_blocking(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

async def download_voice(voice, target_path):
    tg_file = await voice.get_file()
    await tg_file.download_to_drive(target_path)

def convert_ogg_to_wav(ogg_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-i", ogg_path, wav_path, "-y", "-loglevel", "panic"],
        check=True
    )

# ————— HANDLERS —————
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")
    kb = [["Help", "Write", "Record"]]
    await update.message.reply_text("Hi, choose:", reply_markup=kb, resize_keyboard=True)

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")
    text = update.message.text.strip().lower()
    if text == "help":
        return await update.message.reply_text("🆘 How can I assist?")
    if text == "write":
        return await update.message.reply_text("✍️ What would you like me to draft?")
    if text == "record":
        return await update.message.reply_text("🎙️ Please send a voice note.")

    # STREAM GPT RESPONSE
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        stream = openai.chat.completions.create(
            model="gpt-4o", stream=True,
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user",  "content":update.message.text}
            ]
        )
        msg = await update.message.reply_text("")  # placeholder
        buffer = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.get("content", "")
            if delta:
                buffer += delta
                await ctx.bot.edit_message_text(buffer,
                    chat_id=msg.chat_id, message_id=msg.message_id)
    except Exception as e:
        logging.exception("GPT stream error")
        await update.message.reply_text(f"⚠️ Error: {e}")

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ Unauthorized.")

    ogg_f = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    wav_f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        # download & convert
        await download_voice(update.message.voice, ogg_f.name)
        await run_blocking(convert_ogg_to_wav, ogg_f.name, wav_f.name)

        # ASR & translation off‑loaded
        asr_out = await run_blocking(somali_asr, wav_f.name)
        somali_text = asr_out.get("text", "").strip()
        trans_out = await run_blocking(translator, somali_text)
        english = trans_out[0]["translation_text"]

        await update.message.reply_text(
            f"🇸🇴 {somali_text}\n\n🇬🇧 {english}"
        )
    except Exception as e:
        logging.exception("Voice processing error")
        await update.message.reply_text(f"⚠️ Error: {e}")
    finally:
        for f in (ogg_f, wav_f):
            try:
                f.close()
                os.remove(f.name)
            except:
                pass

# ————— BOT SETUP —————
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("🤖 Bot live")
    app.run_polling()
