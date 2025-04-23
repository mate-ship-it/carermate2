import os
import tempfile
import aiohttp
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTAPI_URL = os.getenv("FASTAPI_URL")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me text or a voice message to translate.")

# Handle text messages
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Translate this Somali text into English. Only return the English translation."},
                {"role": "user", "content": text}
            ]
        )
        await update.message.reply_text(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"Text Error: {e}")
        await update.message.reply_text("Error processing your message.")

# Handle voice messages
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    audio_path = ""

    try:
        # Download voice file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as f:
            async with aiohttp.ClientSession() as session:
                async with session.get(file.file_path) as resp:
                    f.write(await resp.read())
            audio_path = f.name

        # Send to FastAPI for transcription + translation
        async with aiohttp.ClientSession() as session:
            with open(audio_path, "rb") as audio_file:
                form = aiohttp.FormData()
                form.add_field('file', audio_file, filename="voice.ogg", content_type='audio/ogg')
                async with session.post(FASTAPI_URL, data=form) as resp:
                    fastapi_result = await resp.json()

        english_text = fastapi_result.get("translation", "Translation failed.")
        await update.message.reply_text(english_text)

    except Exception as e:
        print(f"Voice Error: {e}")
        await update.message.reply_text("Error processing your voice message.")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Main app runner
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("Bot is running...")
    app.run_polling()
