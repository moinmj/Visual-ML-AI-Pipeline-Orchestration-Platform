from backend.app.recipes.vision.object_detection import ObjectDetectionRecipe
from backend.app.recipes.vision.image_classifier import ImageClassifierRecipe
from backend.app.recipes.vision.ocr import OCRRecipe
from backend.app.recipes.vision.image_embedding import ImageEmbeddingRecipe
from backend.app.recipes.vision.video_tracking import VideoObjectTrackingRecipe

__all__ = [
    "ObjectDetectionRecipe", "ImageClassifierRecipe", "OCRRecipe",
    "ImageEmbeddingRecipe", "VideoObjectTrackingRecipe",
]