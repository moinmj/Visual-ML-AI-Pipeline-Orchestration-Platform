"""
Image Embedding Generator recipe (Modeling -> Computer Vision & Audio).

Turns each image into a numeric vector (embedding) with a pretrained torchvision
network (ImageNet weights, classification head removed). Use the vectors for
similarity search, clustering, duplicate detection, or as features for a trainer.

Images come from 'image_files' (picked from unstructured/; same rules and security
checks as object_detection, see _common.py). There is no table / column input.

Place this file at:  backend/app/recipes/vision/image_embedding.py
Needs:               torch + torchvision (already installed). No new packages.
Note:                torchvision downloads the weights (45-100 MB) the first time a model is used.

Security: only the built-in model names below are accepted (no model path from the node config).
"""
import json
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c

# model name -> torchvision weights enum name
SUPPORTED_MODELS = {
    "resnet18": "ResNet18_Weights",                    # 512-d, fastest
    "resnet50": "ResNet50_Weights",                    # 2048-d, good default
    "efficientnet_b0": "EfficientNet_B0_Weights",      # 1280-d
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",  # 576-d, tiny
}
DEFAULT_MODEL = "resnet50"
BATCH_SIZE = 16
OUTPUT_FORMATS = ("json_vector", "columns")

_EMBEDDER_CACHE: Dict[str, Any] = {}
_EMBEDDER_LOCK = threading.Lock()


class _Embedder:
    """A pretrained network with its classification head removed."""

    def __init__(self, name: str):
        import torch
        from torchvision import models

        weights = getattr(models, SUPPORTED_MODELS[name]).DEFAULT
        net = getattr(models, name)(weights=weights)
        if name.startswith("resnet"):
            net.fc = torch.nn.Identity()
        else:
            net.classifier = torch.nn.Identity()
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net.eval().to(self.device)
        self.transform = weights.transforms()

    def encode(self, images: List[Any]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        with torch.no_grad():
            out = self.net(batch)
        return out.cpu().numpy().astype("float32")


def _load_embedder(name: str) -> _Embedder:
    """Loads a model once per process and reuses it."""
    with _EMBEDDER_LOCK:
        if name not in _EMBEDDER_CACHE:
            try:
                _EMBEDDER_CACHE[name] = _Embedder(name)
            except ImportError as e:
                raise RuntimeError("Image Embedding needs 'torch' and 'torchvision'.") from e
        return _EMBEDDER_CACHE[name]


class ImageEmbeddingRecipe(BaseRecipe):
    recipe_id = "image_embedding_generator"
    name = "Image Embedding Generator"
    version = "1.0.0"
    category = "vision_audio"
    media_type = "image"
    description = "Converts images into numeric embedding vectors for search or ML."
    input_types = []  # source node: reads its own files, takes no input
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                **c.source_schema("image"),
                "model": {
                    "type": "select",
                    "title": "Embedding Model",
                    "input_type": "dropdown",
                    "enum": list(SUPPORTED_MODELS),
                    "default": DEFAULT_MODEL,
                    "description": "Pretrained network. resnet18 = fastest (512 values), resnet50 = better quality (2048 values).",
                },
                "normalize": {
                    "type": "boolean",
                    "title": "Normalize Vectors",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Scale each vector to length 1. Recommended for similarity search (cosine).",
                },
                "output_format": {
                    "type": "select",
                    "title": "Output Format",
                    "input_type": "dropdown",
                    "enum": list(OUTPUT_FORMATS),
                    "default": "json_vector",
                    "description": "json_vector = one 'embedding' column (JSON list). columns = one numeric column per value (emb_0, emb_1, ...) so a trainer can use them directly.",
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = c.validate_source(config, "image")
        if config.get("model", DEFAULT_MODEL) not in SUPPORTED_MODELS:
            errors.append(f"Unknown model '{config.get('model')}'. Choose one of: {', '.join(SUPPORTED_MODELS)}.")
        if config.get("output_format", "json_vector") not in OUTPUT_FORMATS:
            errors.append("output_format must be 'json_vector' or 'columns'.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name)
        df, image_col, from_files = c.resolve_source(inputs, config, self.name, "image")

        model_name = config.get("model", DEFAULT_MODEL)
        normalize = bool(config.get("normalize", True))
        out_format = config.get("output_format", "json_vector")
        embedder = _load_embedder(model_name)

        values = df[image_col].tolist()
        vectors: List[Optional[np.ndarray]] = [None] * len(values)
        row_errors: List[Optional[str]] = [None] * len(values)
        sizes: List[Optional[tuple]] = [None] * len(values)

        for start in range(0, len(values), BATCH_SIZE):
            idxs, images = [], []
            for i in range(start, min(start + BATCH_SIZE, len(values))):
                try:
                    image = c.open_image(values[i], tenant_id, from_files)
                    sizes[i] = image.size
                    idxs.append(i)
                    images.append(image)
                except Exception as e:  # one bad image must not fail the whole run
                    row_errors[i] = c.safe_error(e)
            if images:
                batch = np.asarray(embedder.encode(images), dtype="float32")
                if normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    batch = batch / np.where(norms == 0, 1.0, norms)
                for i, vec in zip(idxs, batch):
                    vectors[i] = vec

        dim = next((int(v.shape[0]) for v in vectors if v is not None), None)
        out = df.copy().reset_index(drop=True)
        out["image_width"] = [s[0] if s else None for s in sizes]
        out["image_height"] = [s[1] if s else None for s in sizes]
        out["embedding_dim"] = [dim if v is not None else None for v in vectors]
        if out_format == "columns" and dim:
            emb = pd.DataFrame(
                [v if v is not None else np.full(dim, np.nan, dtype="float32") for v in vectors],
                columns=[f"emb_{k}" for k in range(dim)],
            )
            out = pd.concat([out, emb], axis=1)
        else:
            out["embedding"] = [json.dumps([round(float(x), 6) for x in v]) if v is not None else None for v in vectors]
        out["embedding_error"] = row_errors
        out["image_preview_url"] = [c.preview_url(v) for v in values]

        failed = [e for e in row_errors if e]
        previews = [
            c.preview_item(
                original=v,
                error=err,
                embedding_dim=dim if vec is not None else None,
                embedding_head=[round(float(x), 4) for x in vec[:5]] if vec is not None else None,
            )
            for v, vec, err in zip(values, vectors, row_errors)
        ]
        metrics = {
            "task_type": "image_embedding",
            "model": model_name,
            "input_source": "image_files",
            "embedding_dim": dim,
            "normalized": normalize,
            "items_processed": len(values) - len(failed),
            "items_failed": len(failed),
            "errors_sample": failed[:5],
            "preview_items": previews,
        }
        return {"dataframe": out, "metrics": metrics, "output_summary": metrics, "task_type": "image_embedding"}

    def to_code(self, config: Dict[str, Any]) -> str:
        name = config.get("model", DEFAULT_MODEL)
        head = "fc" if str(name).startswith("resnet") else "classifier"
        weights = SUPPORTED_MODELS.get(name, "ResNet50_Weights")
        return (
            "import torch\nfrom torchvision import models\nfrom PIL import Image\n\n"
            f"weights = models.{weights}.DEFAULT\n"
            f"net = models.{name}(weights=weights)\n"
            f"net.{head} = torch.nn.Identity()\n"
            "net.eval()\n\n"
            "def embed(path):\n"
            "    x = weights.transforms()(Image.open(path).convert('RGB')).unsqueeze(0)\n"
            "    with torch.no_grad():\n"
            "        return net(x)[0].numpy()"
        )