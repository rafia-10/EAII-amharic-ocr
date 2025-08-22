from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import os
from OCR import pipeline   # <-- import your pipeline function

TOKEN = os.environ["TELEGRAM_TOKEN"]

# ---- Bot Handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me an image, and I'll return the recognized text as a file.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get the photo object (largest size)
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    # Download image to local storage
    file_path = f"downloads/{photo.file_id}.jpg"
    os.makedirs("downloads", exist_ok=True)
    await file.download_to_drive(file_path)

    # Run your OCR pipeline with bw=True
    detected_text = pipeline(file_path, bw=True)

    # Save detected text to a file
    txt_file_path = f"downloads/{photo.file_id}.txt"
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(detected_text.strip())

    # Send back the txt file
    with open(txt_file_path, "rb") as f:
        await update.message.reply_document(document=InputFile(f, filename="result.txt"))


# ---- Main ----
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
