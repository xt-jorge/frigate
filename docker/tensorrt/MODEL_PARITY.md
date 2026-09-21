# Baked model cache parity (amd64 GPU and Jetpack 7)

Frigate downloads the license plate recognition, face recognition and bird
classification enrichment files into `/config/model_cache` the first time a
feature that needs them runs. On a host with no public egress that download
never completes, and `/config` is a durable mount that masks whatever the image
put at that path, so the files have to be baked somewhere else in the image and
copied in before Frigate starts.

Two images do that today:

| Image           | Dockerfile                     | Built with           |
| --------------- | ------------------------------ | -------------------- |
| Jetpack 7 arm64 | `docker/tensorrt/Dockerfile.arm64` | `BAKE_MODEL_CACHE=1` |
| amd64 GPU       | `docker/tensorrt/Dockerfile.amd64` | `BAKE_MODEL_CACHE=1` |

Both run the same `docker/tensorrt/bake_model_cache.sh` in a `model-cache`
stage, ship the result at `/model_cache_seed`, and start the same
`model-cache-seed` s6 oneshot
(`docker/tensorrt/detector/rootfs/etc/s6-overlay/s6-rc.d/model-cache-seed`)
ahead of the `frigate` service. **The two images therefore seed the identical
set of files at the identical relative paths.** The seed only ever creates a
file that is not already there, so an operator-supplied or previously downloaded
file always wins.

Every other image — `tensorrt` (amd64) built without the argument, `jp5`, `jp6`,
and the plain builds — defaults to `BAKE_MODEL_CACHE=0`, carries an empty
`/model_cache_seed`, and behaves exactly as upstream.

## Pinned inputs

Frigate itself fetches most of these from a branch tip. A baked image must not
drift with that tip, so every URL below is pinned to an immutable revision or a
tagged release asset, and each file is verified against its sha256 during the
build; a mismatch fails the build. The same manifest is written to
`/model_cache_seed/SHA256SUMS` so the image can re-verify what it carries.

Source revisions:

| Source                                | Pinned revision                            |
| ------------------------------------- | ------------------------------------------ |
| `NickM-27/facenet-onnx`               | release tag `v1.0` (release assets)        |
| `hawkeye217/paddleocr-onnx`           | `dd143338456fc8f68367f6257565287ba070dc10` |
| `hawkeye217/yolov9-license-plates`    | `e3bc34b068d558430fa893638494bef4772b8a18` |
| `google-coral/test_data`              | `104342d2d3480b3e66203073dac24f4e2dbb4c41` |

Files, relative to `/config/model_cache`:

| Destination                                            | sha256 |
| ------------------------------------------------------ | ------ |
| `facedet/facedet.onnx`                                  | `321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294` |
| `facedet/landmarkdet.yaml`                              | `70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b` |
| `facedet/facenet.tflite`                                | `54660297ebad23b7106a8ccd05f2f2f616b3d39bf7d7e82e148cf8ed7e27ae1a` |
| `facedet/arcface.onnx`                                  | `ec639a0429b4819130d1405a2d3b38beaa4cc4a6c5bd9cf48b94fdf65461de83` |
| `paddleocr-onnx/detection_v5-small.onnx`                | `d7fe3ea74652890722c0f4d02458b7261d9f5ae6c92904d05707c9eb155c7924` |
| `paddleocr-onnx/detection_v3-large.onnx`                | `ffe9717ac1270ca5301a79276123efe4392a26b27185888954ce7747227ded9d` |
| `paddleocr-onnx/classification.onnx`                    | `8b3a1675eabd312234b46b64cf257f0e685d07005117fb209e87e34fe5aed90f` |
| `paddleocr-onnx/recognition_v4.onnx`                    | `ad7dd55f6759fa02333bff6eb179a4f51be5b89cbe6f710249c95f47d0211350` |
| `paddleocr-onnx/ppocr_keys_v1.txt`                      | `b100155114d198cdf3a2a5143b1535f597eb7839fd7d858f3d621146b4e603dd` |
| `yolov9_license_plate/yolov9-256-license-plates.onnx`   | `938a34ce868da69432fd5a872f24337b86e3dc8753426e16ae18163f615c2a2c` |
| `bird/bird.tflite`                                      | `350fcd8cf1df1560060d464595dfed8b174b05792788052896004848d9ad04f9` |
| `bird/birdmap.txt`                                      | `a16108dfe3f8daff015b87a97ab6a17e717b9b1bccd719f6d8f747746d7b9277` |

`docker/tensorrt/bake_model_cache.sh` is the source of truth for this list; the
table above is a transcription of its manifest.

## Why one script serves both architectures

Every baked file is ONNX, TFLite, YAML or text — none of it is compiled for a
target. The loaders resolve them by relative path under `MODEL_CACHE_DIR` and
hand the ONNX files to ONNX Runtime
(`frigate/embeddings/onnx/{lpr,face}_embedding.py`,
`frigate/data_processing/real_time/{face,bird}.py`), whichever execution
provider that runtime selected. No TensorRT engine or other device-specific
artifact is baked: ONNX Runtime's TensorRT provider builds its engines at run
time into `/config/model_cache/tensorrt/ort/trt-engines`
(`frigate/util/model.py`), which is deliberately outside the seeded set, and the
Jetson-only `trt-model-prepare` oneshot is not part of the amd64 image.

## What is not baked

The object detection model itself is not part of this set. A Frigate+ model
(`plus://<id>`) is account-scoped and is fetched with the user's API key, so it
cannot be baked here; an air-gapped deployment has to supply it out of band.
