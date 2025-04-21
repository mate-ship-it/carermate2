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
from transformers import pipeline

load_dotenv()

BOT_TOKEN             = os.getenv("BOT_TOKEN")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # if needed for private models

# parse comma-separated IDs
raw_ids = os.getenv("AUTHORIZED_CHAT_ID", "")
AUTHORIZED_CHAT_IDS = [int(x) for x in raw_ids.split(",") if x.strip().isdigit()]

client = OpenAI(api_key=OPENAI_API_KEY)

# initialize the Somali ASR pipeline
# if your Hugging Face model is public, you don't need HF token;
# otherwise set env var HUGGINGFACE_API_TOKEN and uncomment the `use_auth_token` line below:
somali_asr = pipeline(
    "automatic-speech-recognition",
    model="Mustafaa4a/ASR-Somali",
    # use_auth_token=HUGGINGFACE_API_TOKEN
)

def is_authorized(chat_id: int) -> bool:
    return chat_id in AUTHORIZED_CHAT_IDS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")
    keyboard = [["Help", "Write", "Record"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Hi! Please choose an option below:", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")

    text = update.message.text.strip()
    low = text.lower()

    if low == "help":
        return await update.message.reply_text("🆘 What do you need help with?")
    elif low == "write":
        return await update.message.reply_text("✍️ Please type what you'd like me to help you write.")
    elif low == "record":
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
    except Exception:
        await update.message.reply_text("⚠️ Error while processing your message.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_chat.id):
        return await update.message.reply_text("❌ You are not authorised to use this bot.")

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    # download to temp .ogg
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as f:
        async with aiohttp.ClientSession() as session:
            async with session.get(file.file_path) as resp:
                f.write(await resp.read())
        audio_path = f.name

    try:
        # 1) transcribe Somali audio
        asr_result = somali_asr(audio_path)
        os.remove(audio_path)

        somali_text = asr_result["text"]
        # 2) translate Somali text into English via GPT-4
        translation = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user",   "content": f"Translate this Somali text into English:\n\n{somali_text}"}
            ]
        )

        await update.message.reply_text(translation.choices[0].message.content)
    except Exception:
        # cleanup if something went wrong
        if os.path.exists(audio_path):
            os.remove(audio_path)
        await update.message.reply_text("⚠️ Error while transcribing or translating your voice message.")

if __name__ == "__main__":
    if not BOT_TOKEN or not OPENAI_API_KEY or not AUTHORIZED_CHAT_IDS:
        print("🚨 Missing env vars or no authorized chat IDs set.")
        exit(1)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("🤖 Bot is running...")
    app.run_polling()
