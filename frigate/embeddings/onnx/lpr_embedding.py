import logging
import os
import warnings

import cv2
import numpy as np

from frigate.comms.inter_process import InterProcessRequestor
from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_runners import BaseModelRunner, get_optimized_runner
from frigate.embeddings.types import EnrichmentModelTypeEnum
from frigate.types import ModelStatusTypesEnum
from frigate.util.downloader import ModelDownloader

from .base_embedding import BaseEmbedding

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The class CLIPFeatureExtractor is deprecated",
)

logger = logging.getLogger(__name__)

LPR_EMBEDDING_SIZE = 256

# Official PP-OCRv6 medium text recognizer, pinned to the exact HuggingFace
# revision the edge image bakes. The revision hash is part of the URL: a branch
# name would let the upstream artifact change under a cached file name.
PPOCRV6_MEDIUM_REPOSITORY = "PaddlePaddle/PP-OCRv6_medium_rec_onnx"
PPOCRV6_MEDIUM_REVISION = "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"

# Local cache names. They deliberately differ from the upstream file names so a
# model_cache directory can never serve a different recognizer's inference.onnx,
# and from any earlier recognizer so a stale cache entry cannot be reused.
PPOCRV6_MEDIUM_MODEL_FILE = "recognition_ppocrv6_medium.onnx"
PPOCRV6_MEDIUM_CONFIG_FILE = "recognition_ppocrv6_medium.yml"

# Output classes the recognizer emits: the CTC blank, the 18708 ordered entries
# of PostProcess.character_dict, and the trailing space class.
PPOCRV6_MEDIUM_CLASS_COUNT = 18710


class PaddleOCRDetection(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        model_file = (
            "detection_v3-large.onnx"
            if model_size == "large"
            else "detection_v5-small.onnx"
        )
        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")
        super().__init__(
            model_name="paddleocr-onnx",
            model_file=model_file,
            download_urls={
                model_file: f"{GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/{'v3' if model_size == 'large' else 'v5'}/{model_file}"
            },
        )
        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: BaseModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = get_optimized_runner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                model_type=EnrichmentModelTypeEnum.paddleocr.value,
            )

    def _preprocess_inputs(self, raw_inputs):
        preprocessed = []
        for x in raw_inputs:
            preprocessed.append(x)
        return [{"x": preprocessed[0]}]


class PaddleOCRClassification(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")
        super().__init__(
            model_name="paddleocr-onnx",
            model_file="classification.onnx",
            download_urls={
                "classification.onnx": f"{GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx"
            },
        )
        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: BaseModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = get_optimized_runner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                model_type=EnrichmentModelTypeEnum.paddleocr.value,
            )

    def _preprocess_inputs(self, raw_inputs):
        processed = []
        for img in raw_inputs:
            processed.append({"x": img})
        return processed


class PaddleOCRRecognition(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        revision_url = (
            f"{HF_ENDPOINT}/{PPOCRV6_MEDIUM_REPOSITORY}/resolve/"
            f"{PPOCRV6_MEDIUM_REVISION}"
        )
        super().__init__(
            model_name="paddleocr-onnx",
            model_file=PPOCRV6_MEDIUM_MODEL_FILE,
            download_urls={
                PPOCRV6_MEDIUM_MODEL_FILE: f"{revision_url}/inference.onnx",
                # The recognizer's own inference config carries the ordered
                # label map the network was trained against. Decoding without
                # it is guesswork, so it is a required artifact, not an extra.
                PPOCRV6_MEDIUM_CONFIG_FILE: f"{revision_url}/inference.yml",
            },
        )
        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: BaseModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = get_optimized_runner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                model_type=EnrichmentModelTypeEnum.paddleocr.value,
            )

    def _preprocess_inputs(self, raw_inputs):
        processed = []
        for img in raw_inputs:
            processed.append({"x": img})
        return processed


class LicensePlateDetector(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")
        super().__init__(
            model_name="yolov9_license_plate",
            model_file="yolov9-256-license-plates.onnx",
            download_urls={
                "yolov9-256-license-plates.onnx": f"{GITHUB_ENDPOINT}/hawkeye217/yolov9-license-plates/raw/refs/heads/master/models/yolov9-256-license-plates.onnx"
            },
        )

        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: BaseModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = get_optimized_runner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                model_type=EnrichmentModelTypeEnum.yolov9_license_plate.value,
            )

    def _preprocess_inputs(self, raw_inputs):
        if isinstance(raw_inputs, list):
            raise ValueError("License plate embedding does not support batch inputs.")

        img = raw_inputs
        height, width, channels = img.shape

        # Resize maintaining aspect ratio
        if width > height:
            new_height = int(((height / width) * LPR_EMBEDDING_SIZE) // 4 * 4)
            img = cv2.resize(img, (LPR_EMBEDDING_SIZE, new_height))
        else:
            new_width = int(((width / height) * LPR_EMBEDDING_SIZE) // 4 * 4)
            img = cv2.resize(img, (new_width, LPR_EMBEDDING_SIZE))

        # Get new dimensions after resize
        og_h, og_w, channels = img.shape

        # Create black square frame
        frame = np.full(
            (LPR_EMBEDDING_SIZE, LPR_EMBEDDING_SIZE, channels),
            (0, 0, 0),
            dtype=np.float32,
        )

        # Center the resized image in the square frame
        x_center = (LPR_EMBEDDING_SIZE - og_w) // 2
        y_center = (LPR_EMBEDDING_SIZE - og_h) // 2
        frame[y_center : y_center + og_h, x_center : x_center + og_w] = img

        # Normalize to 0-1
        frame = frame / 255.0

        # Convert from HWC to CHW format and add batch dimension
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        return [{"images": frame}]
