#!/bin/sh
# Download the model files Frigate would otherwise fetch on first use so the
# image carries them. Runs in the python-less `wget` build stage, so the list
# mirrors frigate/embeddings/onnx/{lpr,face}_embedding.py and
# frigate/data_processing/real_time/{face,bird}.py: same files, same relative
# paths under /config/model_cache. The runtime presence check is a bare
# os.path.exists (frigate/util/downloader.py), so only the destination name has
# to match.
#
# Frigate itself fetches the PaddleOCR, plate-detector and bird files from
# branch tips. A baked image must not drift with them, so every URL here is
# pinned to an immutable revision (or a tagged release asset) and every file is
# verified against the sha256 recorded below; a mismatch fails the build.
#
# Usage: bake_model_cache.sh <seed-root>
set -eu

SEED_ROOT="${1:?seed root directory required}"
GITHUB_ENDPOINT="${GITHUB_ENDPOINT:-https://github.com}"
GITHUB_RAW_ENDPOINT="${GITHUB_RAW_ENDPOINT:-https://raw.githubusercontent.com}"

mkdir -p "${SEED_ROOT}"
: > "${SEED_ROOT}/SHA256SUMS"

# <sha256> <relative destination> <pinned source url>
while read -r sha dest url; do
    [ -n "${dest}" ] || continue
    mkdir -p "${SEED_ROOT}/$(dirname "${dest}")"
    wget -q --tries=3 --timeout=60 -O "${SEED_ROOT}/${dest}.part" "${url}"
    printf '%s  %s\n' "${sha}" "${SEED_ROOT}/${dest}.part" | sha256sum -c --quiet -
    mv "${SEED_ROOT}/${dest}.part" "${SEED_ROOT}/${dest}"
    printf '%s  ./%s\n' "${sha}" "${dest}" >> "${SEED_ROOT}/SHA256SUMS"
done <<MANIFEST
321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294 facedet/facedet.onnx ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facedet.onnx
70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b facedet/landmarkdet.yaml ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/landmarkdet.yaml
54660297ebad23b7106a8ccd05f2f2f616b3d39bf7d7e82e148cf8ed7e27ae1a facedet/facenet.tflite ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facenet.tflite
ec639a0429b4819130d1405a2d3b38beaa4cc4a6c5bd9cf48b94fdf65461de83 facedet/arcface.onnx ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/arcface.onnx
d7fe3ea74652890722c0f4d02458b7261d9f5ae6c92904d05707c9eb155c7924 paddleocr-onnx/detection_v5-small.onnx ${GITHUB_RAW_ENDPOINT}/hawkeye217/paddleocr-onnx/dd143338456fc8f68367f6257565287ba070dc10/models/v5/detection_v5-small.onnx
ffe9717ac1270ca5301a79276123efe4392a26b27185888954ce7747227ded9d paddleocr-onnx/detection_v3-large.onnx ${GITHUB_RAW_ENDPOINT}/hawkeye217/paddleocr-onnx/dd143338456fc8f68367f6257565287ba070dc10/models/v3/detection_v3-large.onnx
8b3a1675eabd312234b46b64cf257f0e685d07005117fb209e87e34fe5aed90f paddleocr-onnx/classification.onnx ${GITHUB_RAW_ENDPOINT}/hawkeye217/paddleocr-onnx/dd143338456fc8f68367f6257565287ba070dc10/models/classification.onnx
ad7dd55f6759fa02333bff6eb179a4f51be5b89cbe6f710249c95f47d0211350 paddleocr-onnx/recognition_v4.onnx ${GITHUB_RAW_ENDPOINT}/hawkeye217/paddleocr-onnx/dd143338456fc8f68367f6257565287ba070dc10/models/v4/recognition_v4.onnx
b100155114d198cdf3a2a5143b1535f597eb7839fd7d858f3d621146b4e603dd paddleocr-onnx/ppocr_keys_v1.txt ${GITHUB_RAW_ENDPOINT}/hawkeye217/paddleocr-onnx/dd143338456fc8f68367f6257565287ba070dc10/models/v4/ppocr_keys_v1.txt
938a34ce868da69432fd5a872f24337b86e3dc8753426e16ae18163f615c2a2c yolov9_license_plate/yolov9-256-license-plates.onnx ${GITHUB_RAW_ENDPOINT}/hawkeye217/yolov9-license-plates/e3bc34b068d558430fa893638494bef4772b8a18/models/yolov9-256-license-plates.onnx
350fcd8cf1df1560060d464595dfed8b174b05792788052896004848d9ad04f9 bird/bird.tflite ${GITHUB_RAW_ENDPOINT}/google-coral/test_data/104342d2d3480b3e66203073dac24f4e2dbb4c41/mobilenet_v2_1.0_224_inat_bird_quant.tflite
a16108dfe3f8daff015b87a97ab6a17e717b9b1bccd719f6d8f747746d7b9277 bird/birdmap.txt ${GITHUB_RAW_ENDPOINT}/google-coral/test_data/104342d2d3480b3e66203073dac24f4e2dbb4c41/inat_bird_labels.txt
MANIFEST

# The manifest ships with the seed so the image can re-verify what it carries.
( cd "${SEED_ROOT}" && sha256sum -c --quiet SHA256SUMS )
du -sh "${SEED_ROOT}"
cat "${SEED_ROOT}/SHA256SUMS"
