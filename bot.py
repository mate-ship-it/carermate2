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

BOT_TOKEN        = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
AUTH_IDS_RAW     = os.getenv("AUTHORIZED_CHAT_ID", "").strip()
if not BOT_TOKEN or not OPENAI_API_KEY or not AUTH_IDS_RAW:
    raise RuntimeError("BOT_TOKEN, OPENAI_API_KEY, and AUTHORIZED_CHAT_ID must be set")

AUTHORIZED_CHAT_IDS = [int(x) for x in AUTH_IDS_RAW.split(",") if x.isdigit()]
if not AUTHORIZED_CHAT_IDS:
    raise RuntimeError("No valid AUTHORIZED_CHAT_IDs provided")

# Clients
client = OpenAI(api_key=OPENAI_API_KEY)
# ASR model is baked into the image
somali_asr = pipeline("automatic-speech-recognition", model="Mustafaa4a/ASR-Somali")
# Local translation model—small and fast
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-som-en")

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
        return await update.message.reply_text("✍️ What should I draft for you?")
    if low == "record":
        return await update.message.reply_text("🎙️ Send a voice note.")

    # STREAMING GPT response
    try:
        await update.message.chat.send_action("typing")
        stream = client.chat.completions.create(
            model="gpt-4o",
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": txt}
            ]
        )
        buffer = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.get("content", "")
            buffer += delta
            # update the same message so it grows over time
            if len(buffer) > 20:
                try:
                    await context.bot.edit_message_text(
                        chat_id=update.effective_chat.id,
                        message_id=context.user_data.get("stream_msg_id"),
                        text=buffer
                    )
                except:
                    pass
        # store the first time send
        if "stream_msg_id" not in context.user_data:
            msg = await update.message.reply_text(buffer)
            context.user_data["stream_msg_id"] = msg.message_id
    except Exception as e:
        logging.exception("Write command failed")
        await update.message.reply_text(f"⚠️ Error: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised.")

    voice = update.message.voice
    tg_file = await voice.get_file()
    ogg_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    await tg_file.download_to_drive(ogg_tmp.name)

    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", ogg_tmp.name, wav_tmp.name, "-y", "-loglevel", "panic"],
        check=True
    )

    try:
        # 1) ASR (in thread)
        asr = await asyncio.to_thread(somali_asr, wav_tmp.name)
        somali_text = asr.get("text", "").strip()

        # 2) LOCAL translation
        translation = await asyncio.to_thread(translator, somali_text)
        eng = translation[0]["translation_text"]

        # clean up
        for f in (ogg_tmp, wav_tmp):
            f.close(); os.remove(f.name)

        await update.message.reply_text(f"✅ Somali: {somali_text}\n\n🇬🇧 English: {eng}")
    except Exception as e:
        logging.exception("Voice handling failed")
        for f in (ogg_tmp, wav_tmp):
            try: f.close(); os.remove(f.name)
            except: pass
        await update.message.reply_text(f"⚠️ Error: {e}")

# ————— BOOTSTRAP —————
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("🤖 Bot is running…")
    app.run_polling()
