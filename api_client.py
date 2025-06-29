import os
from google.cloud import vision
# We import GOOGLE_APPLICATION_CREDENTIALS_PATH from config.py,
# but the actual setting of the environment variable happens in app.py
# to ensure it's loaded before the client is used.
from config import GOOGLE_APPLICATION_CREDENTIALS_PATH 

class GoogleVisionClient:
    def __init__(self):
        """
        Initializes the Google Cloud Vision API client.
        The GOOGLE_APPLICATION_CREDENTIALS environment variable should be set
        (done in app.py) before this client is created.
        """
        try:
            self.client = vision.ImageAnnotatorClient()
            print("Google Vision Client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Vision Client: {e}")
            print(f"Please ensure the path in GOOGLE_APPLICATION_CREDENTIALS_PATH in config.py is correct and the JSON file exists.")
            self.client = None # Set client to None if initialization fails

    def detect_text_from_image_bytes(self, image_bytes):
        """
        Performs text detection on the given image bytes using Google Vision API.

        Args:
            image_bytes (bytes): The content of the image file (e.g., PNG, JPEG) as bytes.

        Returns:
            list: A list of TextAnnotation objects from the Google Vision API response,
                  or None if the client wasn't initialized or an error occurred.
        """
        if not self.client:
            print("Google Vision client not initialized. Cannot detect text.")
            return None
        try:
            image = vision.Image(content=image_bytes)
            response = self.client.text_detection(image=image)
            # texts[0] is typically the entire text detected in the image,
            # subsequent entries are individual words/lines.
            return response.text_annotations
        except Exception as e:
            print(f"Error during Google Vision text detection: {e}")
            print("Possible issues: network connection, invalid image format, or API usage limits.")
            return None