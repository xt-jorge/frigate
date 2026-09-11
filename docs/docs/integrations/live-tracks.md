---
id: live-tracks
title: Live track metadata
---

`GET /api/tracks` provides compact metadata for integrations that sample current
detector frames. It reads the live tracker, so a true-positive object can appear
without a persisted Event row, saved snapshot, recording, or review item.
Stationary objects remain eligible; motion activity does not determine whether a
track is still present.

Each result contains exactly `id`, `camera`, `label`, and `end_time`. It contains
no image, path history, recognition result, or identity proof. The metadata is
frozen when the camera publishes its detector frame. Later mutations of the
tracker cannot change that published snapshot.

## Discovery

For example:

```text
GET /api/tracks?cameras=front,side&labels=person&in_progress=1&limit=257
```

`cameras` is a required comma-separated list of explicit names; `all` is rejected.
Only cameras allowed by the authenticated user's role are read. `labels` defaults
to `all`;
`in_progress=1` selects tracks without an explicit end and `in_progress=0`
selects retained ended tracks. Omitting `in_progress` includes both.
`limit` defaults to 100 and must be between 1 and 257. Results are ordered by
camera and track ID and contain at most that many rows. An integration with a
256-track budget can request 257 to detect overflow without increasing its
budget.

Discovery returns partial hints. A selected camera whose detector is disabled,
whose frame is missing, or whose frame timestamp is invalid, in the future, or
more than five seconds old is skipped. Other selected cameras can still supply
metadata. An omitted track is always unknown: the list may be limited, a camera
may be unavailable, or its next frame may not have been published yet. Never
retire a cached track or report a camera healthy merely because of this list.

## Exact lookup

```text
GET /api/tracks?cameras=front&labels=person&event_id=1789140000.123456-abcdef&limit=1
```

`event_id` requires exactly one camera. For an authorized camera, an unavailable
or stale detector-frame snapshot returns HTTP 503. A fresh snapshot with no
matching track returns `[]`, which also means unknown. MQTT callbacks can run
before the corresponding frame snapshot is published, so a new MQTT track can
temporarily be absent from this endpoint. The integration can retry while its
own track lifetime remains valid.

Only a returned non-null `end_time` reports the tracker's explicit end. Ended
tracks are retained only until normal tracker cleanup; this is not an event
history API. Responses use `Cache-Control: private, no-store`.

## Atomic current frame

```text
GET /api/front/tracks/1789140000.123456-abcdef/frame.jpg
```

For a true-positive `person`, `car`, `truck`, `bus`, or `motorcycle`, this returns
one full, unannotated detector JPEG. It does not crop to the track. The track
identity and pixels are copied from the same published frame under the camera's
frame lock. The detector and camera first verify the original capture stamp of
the shared-memory ring slot; an overwritten or busy slot is discarded.

`X-Frigate-Track` is a JSON object of at most 1024 bytes containing exactly `id`,
`camera`, `label`, and `end_time`. `X-Frame-Time` is the original capture time in
seconds. `X-Calibration-Frame` contains the same image dimensions, capture time
in milliseconds, and frozen vehicle boxes used by the calibration endpoint.
The response is private and must not be cached.

The route requires access to that camera. A fresh snapshot with a missing or
unsupported track returns HTTP 404 (unknown, retryable), and only an explicit
track end returns HTTP 410. Missing, disabled, stale (over five seconds old),
future-dated, or invalid frame state returns HTTP 503. JPEG/metadata failures also
return 503, never a substituted frame. There is no Event lookup or history
fallback. A frame proves only a current tracked object, not recognized identity.

## Historical events and paired updates

The existing `/api/events` endpoint continues to return persisted event history.
Its temporary `view=track` projection has been removed; live-frame integrations
must use `/api/tracks` and must not fall back to historical events. Recording or
snapshot settings do not need to be enabled to make live tracks visible.

Deploy the producer and its consumers together, and wait for all corrected
instances before acceptance. During a rolling update, old consumers or a
producer without `/api/tracks` can fail closed or return no usable metadata.
Do not turn that temporary lack of metadata into an end event or accepted
recognition evidence.
