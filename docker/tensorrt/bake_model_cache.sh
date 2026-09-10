#!/bin/sh
# Download the model files Frigate would otherwise fetch on first use so the
# image carries them. Runs in the python-less `wget` build stage, so the list
# mirrors the URLs in frigate/embeddings/onnx/{lpr,face}_embedding.py and
# frigate/data_processing/real_time/{face,bird}.py exactly. The runtime presence
# check is a bare os.path.exists on these relative paths under /config/model_cache
# (frigate/util/downloader.py), so only the destination name has to match.
#
# Usage: bake_model_cache.sh <seed-root>
set -eu

SEED_ROOT="${1:?seed root directory required}"
GITHUB_ENDPOINT="${GITHUB_ENDPOINT:-https://github.com}"
GITHUB_RAW_ENDPOINT="${GITHUB_RAW_ENDPOINT:-https://raw.githubusercontent.com}"

mkdir -p "${SEED_ROOT}"

# <relative destination> <source url>
while read -r dest url; do
    [ -n "${dest}" ] || continue
    mkdir -p "${SEED_ROOT}/$(dirname "${dest}")"
    wget -q --tries=3 --timeout=60 -O "${SEED_ROOT}/${dest}.part" "${url}"
    if [ ! -s "${SEED_ROOT}/${dest}.part" ]; then
        echo "empty download: ${url}" >&2
        exit 1
    fi
    mv "${SEED_ROOT}/${dest}.part" "${SEED_ROOT}/${dest}"
done <<MANIFEST
facedet/facedet.onnx ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facedet.onnx
facedet/landmarkdet.yaml ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/landmarkdet.yaml
facedet/facenet.tflite ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facenet.tflite
facedet/arcface.onnx ${GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/arcface.onnx
paddleocr-onnx/detection_v5-small.onnx ${GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v5/detection_v5-small.onnx
paddleocr-onnx/detection_v3-large.onnx ${GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v3/detection_v3-large.onnx
paddleocr-onnx/classification.onnx ${GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx
paddleocr-onnx/recognition_v4.onnx ${GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/recognition_v4.onnx
paddleocr-onnx/ppocr_keys_v1.txt ${GITHUB_ENDPOINT}/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/v4/ppocr_keys_v1.txt
yolov9_license_plate/yolov9-256-license-plates.onnx ${GITHUB_ENDPOINT}/hawkeye217/yolov9-license-plates/raw/refs/heads/master/models/yolov9-256-license-plates.onnx
bird/bird.tflite ${GITHUB_RAW_ENDPOINT}/google-coral/test_data/master/mobilenet_v2_1.0_224_inat_bird_quant.tflite
bird/birdmap.txt ${GITHUB_RAW_ENDPOINT}/google-coral/test_data/master/inat_bird_labels.txt
MANIFEST

# The paddleocr URLs are branch tips rather than tagged releases, so the digest
# manifest is the only stable identity of what a given image actually carries.
(
    cd "${SEED_ROOT}"
    find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS
)
du -sh "${SEED_ROOT}"
cat "${SEED_ROOT}/SHA256SUMS"
