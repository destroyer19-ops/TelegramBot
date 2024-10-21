
```markdown
# Doctor Assistant Bot

This is a Telegram bot that can assist in diagnosing lung diseases based on user-provided X-ray images. The bot uses a pre-trained deep learning model (`MobileNet_lung_disease_diagnosis-5_Class.keras`) to predict the likelihood of five different lung diseases, including Pneumonia, Covid, Lung Opacity, Normal, and Tuberculosis.

## Features

- Accepts compressed images (as photos) and uncompressed images (as documents).
- Processes the images and returns a prediction of lung diseases based on the provided image.
- Supports interaction through text and media messages.
- Built with TensorFlow for machine learning, and `python-telegram-bot` library for Telegram integration.

## Model Information

The deep learning model used is a fine-tuned **MobileNet** trained for classifying five types of lung conditions:
1. **Pneumonia**
2. **Covid**
3. **Lung Opacity**
4. **Normal**
5. **Tuberculosis**

## Requirements

Before running the bot, ensure you have the following installed:

- Python 3.7+
- TensorFlow
- Pillow
- python-telegram-bot

You can install the dependencies with:

```bash
pip install tensorflow Pillow python-telegram-bot
```

## Usage

### Running the Bot

1. Clone the repository and navigate to the project directory.

2. Download and place the trained model file (`MobileNet_lung_disease_diagnosis-5_Class.keras`) in the project directory.

3. Open the `main.py` file and replace the Telegram bot token in the line below with your own bot token:
    ```python
    application = ApplicationBuilder().token('<YOUR_BOT_TOKEN>').build()
    ```

4. Start the bot by running:
    ```bash
    python main.py
    ```

### Commands

- **/start**: Starts the conversation with the bot and welcomes the user.

### How to Use

- **Send an Image**: To diagnose an image, send a lung X-ray image to the bot. The bot will accept both compressed and uncompressed images. Compressed images (e.g., from mobile devices) can be sent as photos, while uncompressed images (e.g., from desktop) should be sent as documents.

- **Receive a Prediction**: After receiving the image, the bot will preprocess it, run it through the model, and return the predicted probabilities for each lung condition.

## Code Overview

- **`start()`**: A simple command handler that sends a welcome message when the `/start` command is issued.

- **`preprocess_image(image)`**: A function to resize, normalize, and prepare the image for prediction.

- **`predict_and_reply(context, image, chat_id, message_id)`**: Runs the preprocessed image through the model and sends the top predictions as a message.

- **`handle_photos(update, context)`**: Handles compressed images (photos) sent via Telegram.

- **`handle_documents(update, context)`**: Handles uncompressed images sent as documents via Telegram.

- **`main()`**: The main function where the bot is initialized, handlers are registered, and the bot is started.

## Logging

The bot logs all activities to the console using Python's `logging` module. This helps in debugging and tracking bot interactions.

## License

This project is open-source and available under the [MIT License](LICENSE).
```

### Key Points:
1. Replace `<YOUR_BOT_TOKEN>` with your actual Telegram bot token.
2. Ensure you have your trained model in the same directory.
3. Instructions are provided on how to install dependencies, run the bot, and interact with it.