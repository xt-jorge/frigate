"""Manages camera object detection processes."""

import logging
import queue
import time
from datetime import datetime, timezone
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

import cv2

from frigate.camera import CameraMetrics, PTZMetrics
from frigate.camera.state import FrozenFace
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import CameraConfig, DetectConfig, LoggerConfig, ModelConfig
from frigate.config.camera.camera import CameraTypeEnum
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.const import (
    PROCESS_PRIORITY_HIGH,
    REQUEST_REGION_GRID,
)
from frigate.motion import MotionDetector
from frigate.motion.improved_motion import ImprovedMotionDetector
from frigate.object_detection.base import RemoteObjectDetector
from frigate.ptz.autotrack import ptz_moving_at_frame_time
from frigate.track import ObjectTracker
from frigate.track.norfair_tracker import NorfairTracker
from frigate.track.tracked_object import TrackedObjectAttribute
from frigate.util.builtin import EventsPerSecond
from frigate.util.image import (
    FrameManager,
    SharedMemoryFrameManager,
    draw_box_with_label,
    publication_frame_name,
)
from frigate.util.model import OCCUPANCY_CANDIDATE_MIN_SCORE
from frigate.util.object import (
    create_tensor_input,
    get_cluster_candidates,
    get_cluster_region,
    get_cluster_region_from_grid,
    get_min_region_size,
    get_startup_regions,
    inside_any,
    intersects_any,
    is_object_filtered,
    reduce_detections,
)
from frigate.util.process import FrigateProcess
from frigate.util.time import get_tomorrow_at_time
from frigate.video.occupancy import (
    OccupancyContinuity,
    contains,
    occupancy_frame,
    zone_bounds,
)

logger = logging.getLogger(__name__)


class CameraTracker(FrigateProcess):
    def __init__(
        self,
        config: CameraConfig,
        model_config: ModelConfig,
        labelmap: dict[int, str],
        detection_queue: Queue,
        detected_objects_queue,
        camera_metrics: CameraMetrics,
        ptz_metrics: PTZMetrics,
        region_grid: list[list[dict[str, Any]]],
        publication_frame_count: int,
        stop_event: MpEvent,
        log_config: LoggerConfig | None = None,
    ) -> None:
        super().__init__(
            stop_event,
            PROCESS_PRIORITY_HIGH,
            name=f"frigate.process:{config.name}",
            daemon=True,
        )
        self.config = config
        self.model_config = model_config
        self.labelmap = labelmap
        self.detection_queue = detection_queue
        self.detected_objects_queue = detected_objects_queue
        self.camera_metrics = camera_metrics
        self.ptz_metrics = ptz_metrics
        self.region_grid = region_grid
        self.publication_frame_count = publication_frame_count
        self.log_config = log_config

    def run(self) -> None:
        self.pre_run_setup(self.log_config)
        frame_queue = self.camera_metrics.frame_queue
        frame_shape = self.config.frame_shape

        motion_detector = ImprovedMotionDetector(
            frame_shape,
            self.config.motion,
            self.config.detect.fps,
            name=self.config.name,
            ptz_metrics=self.ptz_metrics,
        )
        object_detector = RemoteObjectDetector(
            self.config.name,
            self.labelmap,
            self.detection_queue,
            self.model_config,
            self.stop_event,
        )

        object_tracker = NorfairTracker(self.config, self.ptz_metrics)

        frame_manager = SharedMemoryFrameManager()

        # create communication for region grid updates
        requestor = InterProcessRequestor()

        process_frames(
            requestor,
            frame_queue,
            frame_shape,
            self.model_config,
            self.config,
            frame_manager,
            motion_detector,
            object_detector,
            object_tracker,
            self.detected_objects_queue,
            self.camera_metrics,
            self.stop_event,
            self.ptz_metrics,
            self.region_grid,
            self.publication_frame_count,
        )

        # empty the frame queue
        logger.info(f"{self.config.name}: emptying frame queue")
        while not frame_queue.empty():
            (frame_name, _) = frame_queue.get(False)
            frame_manager.delete(frame_name)

        logger.info(f"{self.config.name}: exiting subprocess")


def detect(
    detect_config: DetectConfig,
    object_detector,
    frame,
    model_config: ModelConfig,
    region,
    objects_to_track,
    object_filters,
    occupancy_candidates: list[tuple[Any, ...]] | None = None,
):
    """Return ordinary detections and optionally collect same-inference candidates."""
    tensor_input = create_tensor_input(frame, model_config, region)

    detections = []
    region_detections = object_detector.detect(
        tensor_input,
        threshold=OCCUPANCY_CANDIDATE_MIN_SCORE
        if occupancy_candidates is not None
        else 0.4,
    )
    for d in region_detections:
        box = d[2]
        size = region[2] - region[0]
        x_min = int(max(0, (box[1] * size) + region[0]))
        y_min = int(max(0, (box[0] * size) + region[1]))
        x_max = int(min(detect_config.width - 1, (box[3] * size) + region[0]))
        y_max = int(min(detect_config.height - 1, (box[2] * size) + region[1]))

        # ignore objects that were detected outside the frame
        if (x_min >= detect_config.width - 1) or (y_min >= detect_config.height - 1):
            continue

        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        ratio = width / max(1, height)
        det = (d[0], d[1], (x_min, y_min, x_max, y_max), area, ratio, region)
        if (
            occupancy_candidates is not None
            and d[0] in objects_to_track
            and width > 0
            and height > 0
        ):
            occupancy_candidates.append(det)
        # apply object filters
        if d[1] < 0.4 or is_object_filtered(det, objects_to_track, object_filters):
            continue
        detections.append(det)
    return detections


def process_frames(
    requestor: InterProcessRequestor,
    frame_queue: Queue,
    frame_shape: tuple[int, int],
    model_config: ModelConfig,
    camera_config: CameraConfig,
    frame_manager: FrameManager,
    motion_detector: MotionDetector,
    object_detector: RemoteObjectDetector,
    object_tracker: ObjectTracker,
    detected_objects_queue: Queue,
    camera_metrics: CameraMetrics,
    stop_event: MpEvent,
    ptz_metrics: PTZMetrics,
    region_grid: list[list[dict[str, Any]]],
    publication_frame_count: int,
    exit_on_empty: bool = False,
):
    next_region_update = get_tomorrow_at_time(2)
    config_subscriber = CameraConfigUpdateSubscriber(
        None,
        {camera_config.name: camera_config},
        [
            CameraConfigUpdateEnum.detect,
            CameraConfigUpdateEnum.enabled,
            CameraConfigUpdateEnum.motion,
            CameraConfigUpdateEnum.objects,
        ],
    )

    fps_tracker = EventsPerSecond()
    fps_tracker.start()

    startup_scan = True
    stationary_frame_counter = 0
    occupancy_continuity = OccupancyContinuity(
        camera_config.detect.max_disappeared / camera_config.detect.fps
    )
    last_occupancy_frame = float("-inf")
    camera_enabled = True
    publication_index = 0

    region_min_size = get_min_region_size(model_config)

    attributes_map = model_config.attributes_map
    all_attributes = model_config.all_attributes

    # remove license_plate from attributes if this camera is a dedicated LPR cam
    if camera_config.type == CameraTypeEnum.lpr:
        modified_attributes_map = model_config.attributes_map.copy()

        if (
            "car" in modified_attributes_map
            and "license_plate" in modified_attributes_map["car"]
        ):
            modified_attributes_map["car"] = [
                attr
                for attr in modified_attributes_map["car"]
                if attr != "license_plate"
            ]

            attributes_map = modified_attributes_map

        all_attributes = [
            attr for attr in model_config.all_attributes if attr != "license_plate"
        ]

    while not stop_event.is_set():
        updated_configs = config_subscriber.check_for_updates()

        if "enabled" in updated_configs:
            prev_enabled = camera_enabled
            camera_enabled = camera_config.enabled

        if "motion" in updated_configs:
            motion_detector.config = camera_config.motion
            motion_detector.update_mask()

        if (
            not camera_enabled
            and prev_enabled != camera_enabled
            and camera_metrics.frame_queue.empty()
        ):
            logger.debug(
                f"Camera {camera_config.name} disabled, clearing tracked objects"
            )
            prev_enabled = camera_enabled

            # Clear norfair's dictionaries
            object_tracker.tracked_objects.clear()
            object_tracker.disappeared.clear()
            object_tracker.stationary_box_history.clear()
            object_tracker.positions.clear()
            object_tracker.track_id_map.clear()

            # Clear internal norfair states
            for trackers_by_type in object_tracker.trackers.values():
                for tracker in trackers_by_type.values():
                    tracker.tracked_objects = []
            for tracker in object_tracker.default_tracker.values():
                tracker.tracked_objects = []

        if not camera_enabled:
            time.sleep(0.1)
            continue

        if datetime.now().astimezone(timezone.utc) > next_region_update:
            region_grid = requestor.send_data(REQUEST_REGION_GRID, camera_config.name)
            next_region_update = get_tomorrow_at_time(2)

        try:
            if exit_on_empty:
                frame_name, frame_time = frame_queue.get(False)
            else:
                frame_name, frame_time = frame_queue.get(True, 1)
        except queue.Empty:
            if exit_on_empty:
                logger.info("Exiting track_objects...")
                break
            continue

        camera_metrics.detection_frame.value = frame_time
        ptz_metrics.frame_time.value = frame_time

        frame = frame_manager.get_captured_frame(
            frame_name, (frame_shape[0] * 3 // 2, frame_shape[1]), frame_time
        )

        if frame is None:
            logger.debug(
                f"{camera_config.name}: frame {frame_time} is not in memory store."
            )
            continue

        # look for motion if enabled
        motion_boxes = motion_detector.detect(frame)

        regions = []
        consolidated_detections = []
        detector_times: dict[int, float | None] = {}
        occupancy = None
        occupancy_due = (
            bool(camera_config.detect.occupancy_zones)
            and frame_time - last_occupancy_frame >= 0.25
        )
        bounds = zone_bounds(camera_config) if occupancy_due else None
        coverage = []
        observed_detections = []
        observed_candidates = []

        # if detection is disabled
        if not camera_config.detect.enabled:
            object_tracker.match_and_update(frame, frame_time, [])
        else:
            # get stationary object ids
            # check every Nth frame for stationary objects
            # disappeared objects are not stationary
            # also check for overlapping motion boxes
            if stationary_frame_counter == camera_config.detect.stationary.interval:
                stationary_frame_counter = 0
                stationary_object_ids = []
            else:
                stationary_frame_counter += 1
                stationary_object_ids = [
                    obj["id"]
                    for obj in object_tracker.tracked_objects.values()
                    # if it has exceeded the stationary threshold
                    if obj["motionless_count"]
                    >= camera_config.detect.stationary.threshold
                    # and it hasn't disappeared
                    and object_tracker.disappeared[obj["id"]] == 0
                    # and it doesn't overlap with any current motion boxes when not calibrating
                    and not intersects_any(
                        obj["box"],
                        [] if motion_detector.is_calibrating() else motion_boxes,
                    )
                ]

            # get tracked object boxes that aren't stationary
            tracked_object_boxes = [
                (
                    # use existing object box for stationary objects
                    obj["estimate"]
                    if obj["motionless_count"]
                    < camera_config.detect.stationary.threshold
                    else obj["box"]
                )
                for obj in object_tracker.tracked_objects.values()
                if obj["id"] not in stationary_object_ids
            ]
            object_boxes = tracked_object_boxes + object_tracker.untracked_object_boxes

            # get consolidated regions for tracked objects
            regions = [
                get_cluster_region(
                    frame_shape, region_min_size, candidate, object_boxes
                )
                for candidate in get_cluster_candidates(
                    frame_shape, region_min_size, object_boxes
                )
            ]

            # only add in the motion boxes when not calibrating and a ptz is not moving via autotracking
            # ptz_moving_at_frame_time() always returns False for non-autotracking cameras
            if not motion_detector.is_calibrating() and not ptz_moving_at_frame_time(
                frame_time,
                ptz_metrics.start_time.value,
                ptz_metrics.stop_time.value,
            ):
                # find motion boxes that are not inside tracked object regions
                standalone_motion_boxes = [
                    b for b in motion_boxes if not inside_any(b, regions)
                ]

                if standalone_motion_boxes:
                    motion_clusters = get_cluster_candidates(
                        frame_shape,
                        region_min_size,
                        standalone_motion_boxes,
                    )
                    motion_regions = [
                        get_cluster_region_from_grid(
                            frame_shape,
                            region_min_size,
                            candidate,
                            standalone_motion_boxes,
                            region_grid,
                        )
                        for candidate in motion_clusters
                    ]
                    regions += motion_regions

            # if starting up, get the next startup scan region
            if startup_scan:
                for region in get_startup_regions(
                    frame_shape, region_min_size, region_grid
                ):
                    regions.append(region)
                startup_scan = False

            # Refresh only existing occupancy zones. Normal motion/track crops
            # already covering a zone cost no additional detector invocation.
            occupancy_stable = occupancy_due and not ptz_moving_at_frame_time(
                frame_time, ptz_metrics.start_time.value, ptz_metrics.stop_time.value
            )
            if occupancy_due and bounds and occupancy_stable:
                for box in bounds:
                    if not any(contains(region, box) for region in regions):
                        regions.append(
                            get_cluster_region(frame_shape, region_min_size, [0], [box])
                        )

            # resize regions and detect
            # seed with stationary objects
            detections = [
                (
                    obj["label"],
                    obj["score"],
                    obj["box"],
                    obj["area"],
                    obj["ratio"],
                    obj["region"],
                )
                for obj in object_tracker.tracked_objects.values()
                if obj["id"] in stationary_object_ids
            ]

            # Reduction retains selected tuple objects, so provenance follows the
            # exact winning box, including reused stationary detections.
            detector_times.update(
                {
                    id(detection): obj.get("detector_observed_at")
                    for detection, obj in zip(
                        detections,
                        (
                            obj
                            for obj in object_tracker.tracked_objects.values()
                            if obj["id"] in stationary_object_ids
                        ),
                    )
                }
            )
            for region in regions:
                candidates = [] if occupancy_due else None
                observed = detect(
                    camera_config.detect,
                    object_detector,
                    frame,
                    model_config,
                    region,
                    camera_config.objects.track,
                    camera_config.objects.filters,
                    candidates,
                )
                if (
                    candidates is not None
                    and object_detector.last_detection_successful is True
                ):
                    coverage.append(region)
                    # A plate or a face is a property of the thing carrying it.
                    # Tracking already refuses attributes; occupancy has to as
                    # well, or a plate becomes a second vehicle in the zone and
                    # a windshield face becomes an occupant of its own.
                    observed_detections.extend(
                        d for d in observed if d[0] not in all_attributes
                    )
                    observed_candidates.extend(
                        d for d in candidates if d[0] not in all_attributes
                    )
                for detection in observed:
                    detector_times[id(detection)] = frame_time
                detections.extend(observed)

            consolidated_detections = reduce_detections(frame_shape, detections)

            # if detection was run on this frame, consolidate
            if len(regions) > 0:
                tracked_detections = [
                    d for d in consolidated_detections if d[0] not in all_attributes
                ]
                # now that we have refined our detections, we need to track objects
                object_tracker.match_and_update(
                    frame,
                    frame_time,
                    tracked_detections,
                    detector_observed_at=[
                        detector_times[id(d)] for d in tracked_detections
                    ],
                )
            # else, just update the frame times for the stationary objects
            else:
                object_tracker.update_frame_times(frame, frame_time)

        if occupancy_due:
            if not camera_config.detect.enabled or not occupancy_stable:
                coverage = []
                observed_detections = []
                observed_candidates = []
            occupancy_detections = reduce_detections(frame_shape, observed_detections)
            occupancy_tracks = object_tracker.occupancy_tracks()
            occupancy = occupancy_frame(
                camera_config.name,
                frame_time,
                frame_shape,
                bounds,
                coverage,
                occupancy_detections,
                occupancy_tracks,
                [],
            )
            occupancy["regions"] = occupancy_continuity.observe(
                frame,
                frame_time,
                {
                    name: camera_config.zones[name].contour
                    for name in camera_config.detect.occupancy_zones
                }
                if bounds
                else {},
                # Each crop already has a bounded NMS result. The ordinary
                # reducer's 0.5 cutoff would discard this ambiguity again.
                observed_candidates,
                occupancy_tracks,
                occupancy["complete"],
            )
            last_occupancy_frame = frame_time

        # build detections
        detections = {}
        for obj in object_tracker.tracked_objects.values():
            detections[obj["id"]] = {**obj, "attributes": []}

        # Freeze the raw face regions before anything is assigned a parent.
        # attributes_map sends face to person alone and find_best_object needs
        # containment, so a windshield face on a car track ends up owned by
        # nothing - which is precisely the case a vehicle capture needs. None
        # means this frame had no face observation pass at all, which is a
        # different claim from an empty pass and stays distinguishable.
        face_regions: tuple[FrozenFace, ...] | None = None
        if regions and "face" in camera_config.objects.track:
            face_regions = tuple(
                FrozenFace(tuple(int(v) for v in d[2]), float(d[1]), frame_time)
                for d in consolidated_detections
                # Equality against the raw clock, before any rounding: a reused
                # detection shares a track and a frame time without ever having
                # been measured on these pixels.
                if d[0] == "face" and detector_times.get(id(d)) == frame_time
            )

        # assign each detected attribute to the best matching object.
        # iterate consolidated_detections once so attributes that appear under
        # multiple parent labels in attributes_map (e.g. license_plate is in
        # both "car" and "motorcycle") are not appended more than once
        all_objects: list[dict[str, Any]] = object_tracker.tracked_objects.values()
        detected_attributes = [
            TrackedObjectAttribute(d, detector_times.get(id(d)))
            for d in consolidated_detections
            if d[0] in all_attributes
        ]
        for attribute in detected_attributes:
            filtered_objects = filter(
                lambda o: attribute.label in attributes_map.get(o["label"], []),
                all_objects,
            )
            selected_object_id = attribute.find_best_object(filtered_objects)

            if selected_object_id is not None:
                detections[selected_object_id]["attributes"].append(
                    attribute.get_tracking_data()
                )

        # debug object tracking
        if False:
            bgr_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_YUV2BGR_I420,
            )
            object_tracker.debug_draw(bgr_frame, frame_time)
            cv2.imwrite(
                f"debug/frames/track-{'{:.6f}'.format(frame_time)}.jpg", bgr_frame
            )
        # debug
        if False:
            bgr_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_YUV2BGR_I420,
            )

            for m_box in motion_boxes:
                cv2.rectangle(
                    bgr_frame,
                    (m_box[0], m_box[1]),
                    (m_box[2], m_box[3]),
                    (0, 0, 255),
                    2,
                )

            for b in tracked_object_boxes:
                cv2.rectangle(
                    bgr_frame,
                    (b[0], b[1]),
                    (b[2], b[3]),
                    (255, 0, 0),
                    2,
                )

            for obj in object_tracker.tracked_objects.values():
                if obj["frame_time"] == frame_time:
                    thickness = 2
                    color = model_config.colormap.get(obj["label"], (255, 255, 255))
                else:
                    thickness = 1
                    color = (255, 0, 0)

                # draw the bounding boxes on the frame
                box = obj["box"]

                draw_box_with_label(
                    bgr_frame,
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    obj["label"],
                    obj["id"],
                    thickness=thickness,
                    color=color,
                )

            for region in regions:
                cv2.rectangle(
                    bgr_frame,
                    (region[0], region[1]),
                    (region[2], region[3]),
                    (0, 255, 0),
                    2,
                )

            cv2.imwrite(
                f"debug/frames/{camera_config.name}-{'{:.6f}'.format(frame_time)}.jpg",
                bgr_frame,
            )
        # add to the queue if not full
        if detected_objects_queue.full():
            continue

        # Hand the exact pixels this detection was measured on to every
        # downstream consumer. The capture slot the frame arrived in is already
        # being reused by the camera; publishing its name would make consumers
        # reopen a slot that no longer holds this frame. The original capture
        # clock rides along unchanged so every consumer still reads by the exact
        # clock that belongs to this measurement.
        if publication_frame_count < 1:
            continue

        published_name = publication_frame_name(camera_config.name, publication_index)
        if not frame_manager.write_captured_frame(
            published_name, frame.tobytes(), frame_time
        ):
            # Never publish a descriptor with no pixels behind it.
            logger.debug(f"{camera_config.name}: could not publish frame {frame_time}")
            continue
        publication_index = (publication_index + 1) % publication_frame_count

        fps_tracker.update()
        camera_metrics.process_fps.value = fps_tracker.eps()
        detected_objects_queue.put(
            (
                camera_config.name,
                published_name,
                frame_time,
                detections,
                motion_boxes,
                regions,
                occupancy,
                face_regions,
            )
        )
        camera_metrics.detection_fps.value = object_detector.fps.eps()

    motion_detector.stop()
    requestor.stop()
    config_subscriber.stop()
