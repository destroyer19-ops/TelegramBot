import logging
import requests
from telegram import Update, InputMediaPhoto, InputMediaDocument
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters as Filters
)
from PIL import Image, UnidentifiedImageError
import io

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# API Endpoint of your web app for image classification
# API_ENDPOINT = 'https://lung-diseases-classification-n7om.onrender.com//classify'
API_ENDPOINT = 'https://lung-diseases-classification-2.onrender.com/'

# Define the start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm your doctor assistant bot, please talk to me!\nSend a valid Chest X-ray")

# Prediction function
async def predict_and_reply(context: ContextTypes.DEFAULT_TYPE, image, chat_id, message_id):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    # Prepare data to send to API
    files = {'xray_image': ('image.jpeg', image_bytes, 'image/jpeg')}
    try:
        response = requests.post(API_ENDPOINT, files=files)
        if response.ok:
            data = response.json()
            prediction_text = f"I detect potential {data['classification']} in the X-ray.\n"
            await context.bot.send_message(chat_id=chat_id, text=prediction_text, reply_to_message_id=message_id)
        else:
            await context.bot.send_message(chat_id=chat_id, text="Error in classification. Please try again later.", reply_to_message_id=message_id)
    except Exception as e:
        logger.error(f"Error in API request: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Error in classification. Please try again later.", reply_to_message_id=message_id)

# Handle compressed photos
async def handle_photos(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Handling compressed photos")
    largest_photo = update.message.photo[-1]
    file_id = largest_photo.file_id
    new_file = await context.bot.get_file(file_id)
    photo_bytes = await new_file.download_as_bytearray()
    image = Image.open(io.BytesIO(photo_bytes))
    await predict_and_reply(context, image, update.effective_chat.id, update.message.message_id)

# Handle uncompressed photos
async def handle_documents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Handling uncompressed photos")
    file_id = update.message.document.file_id
    new_file = await context.bot.get_file(file_id)
    document_bytes = await new_file.download_as_bytearray()

    try:
        image = Image.open(io.BytesIO(document_bytes))
        await predict_and_reply(context, image, update.effective_chat.id, update.message.message_id)
    except UnidentifiedImageError:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please send a valid image in one of the supported formats.", reply_to_message_id=update.message.message_id)

# Main handler
async def handle_media_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Handling media group")
    media_group = update.message.media_group_id
    if media_group:
        if update.message.photo:
            await handle_photos(update, context)
        elif update.message.document:
            await handle_documents(update, context)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please send a valid image.", reply_to_message_id=update.message.message_id)

def main() -> None:
    application = ApplicationBuilder().token('7411606724:AAFhc0TGI22sSSP258Jh640HdDkw0bqCcFc').build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    
    # Handlers for compressed and uncompressed photos
    photo_handler = MessageHandler(Filters.PHOTO, handle_photos)
    document_handler = MessageHandler(Filters.Document.ALL, handle_documents)
    media_group_handler = MessageHandler(Filters.ALL, handle_media_group)
    
    application.add_handler(photo_handler)
    application.add_handler(document_handler)
    application.add_handler(media_group_handler)

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
