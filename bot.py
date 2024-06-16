import logging
from telegram.ext import (filters as Filters, CommandHandler, ContextTypes, ApplicationBuilder, MessageHandler)
from telegram import Update

import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from tensorflow.keras.preprocessing.image import img_to_array

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load model 
model = tf.keras.models.load_model('MobileNet_lung_disease_diagnosis-5_Class.keras')

# class labels 
classification_classes = {
    0: 'Pneumonia',
    1: 'Covid',
    2: 'Lung_Opacity',
    3: 'Normal',
    4: 'Tuberculosis'
}

def preprocess_image(image) -> np.array:
    """
    Here the input image is preprocessed for classification

    Parameters: 
    image (PIL.image): This is the input image to be preprocessed

    It returns the preprocessed images as a Numpy array - np.array
    """
    image = Image.open(image).convert("RGB").resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # print
    return image

async def classify_image(image: np.array) -> dict:
    
    classification = model.predict(image, verbose=0)[0]
    classified_label = classification_classes[np.argmax(classification)]
    print(classification)
    print(classification.shape)
    return {
        "classification": classified_label,
        "Pneumonia": round(float(classification[0]), 6),
        "Covid": round(float(classification[1]), 6),
        "Lung_Opacity": round(float(classification[2]), 6),
        "Normal": round(float(classification[3]), 6),
        "Tuberculosis": round(float(classification[4]), 6)
    }

# Handle compressed photos
async def handle_photos(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Handling  compressed photos")
    largest_photo = update.message.photo[-1]
    file_id = largest_photo.file_id
    new_file = await context.bot.get_file(file_id)
    photo_bytes = await new_file.download_as_bytearray()
    image = Image.open(io.BytesIO(photo_bytes))
    await classify_image(context, image, update.effective_chat.id, update.message.message_id)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome to Classification bot, please talk to me!")
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Send me your Xray image")
async def handle_documents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Handling uncompressed photos")
    file_id = update.message.document.file_id
    new_file = await context.bot.get_file(file_id)
    document_bytes = await new_file.download_as_bytearray()

    try:
        image = Image.open(io.BytesIO(document_bytes))
        await classify_image(context, image, update.effective_chat.id, update.message.message_id)
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

    help_handler = CommandHandler('help', help)

    # Handlers for compressed and uncompressed photos
    photo_handler = MessageHandler(Filters.PHOTO, handle_photos)

    document_handler = MessageHandler(Filters.Document.ALL, handle_documents) # telegram handles uncompressed photos sent thru desktop app as documents
    media_group_handler = MessageHandler(Filters.ALL, handle_media_group)
    application.add_handler(start_handler)
    application.add_handler(help_handler)
    application.add_handler(photo_handler)
    application.add_handler(document_handler)
    application.add_handler(media_group_handler)


    application.run_polling()

if __name__ == '__main__':
    main()
