from backend.app.recipes.vision.object_detection import ObjectDetectionRecipe
from backend.app.recipes.vision.image_classifier import ImageClassifierRecipe
from backend.app.recipes.vision.ocr import OCRRecipe
from backend.app.recipes.vision.image_embedding import ImageEmbeddingRecipe
from backend.app.recipes.vision.video_tracking import VideoObjectTrackingRecipe
from backend.app.recipes.vision.speech_to_text import SpeechToTextRecipe
from backend.app.recipes.vision.text_to_speech import TextToSpeechRecipe

__all__ = [
    "ObjectDetectionRecipe", "ImageClassifierRecipe", "OCRRecipe",
    "ImageEmbeddingRecipe", "VideoObjectTrackingRecipe",
    "SpeechToTextRecipe", "TextToSpeechRecipe",
]