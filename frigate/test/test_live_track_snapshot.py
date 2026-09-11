"""Live track metadata belongs to a published frame, not mutable tracker state."""

import unittest

from frigate.test import test_tracked_object_publication as tracked_fixture

CAMERA = tracked_fixture.CAMERA
EVENT_ID = tracked_fixture.EVENT_ID


class LivePersonFixture:
    """Drive real camera callbacks without processor sockets or persistence."""

    def __init__(self, position_changes=1):
        self.source = tracked_fixture.TestRecognizedPlatePublication()
        self.source.setUp()
        self.processor = self.source.processor
        self.state = self.source.state
        self.state.camera_config.enabled = True
        self.state.camera_config.detect.enabled = True
        self.position_changes = position_changes
        objects = self.state.camera_config.objects
        objects.track.append("person")
        objects.filters["person"] = objects.filters["car"].model_copy(deep=True)

    def close(self):
        self.source.doCleanups()

    def advance(self, present=True, score=0.9):
        """Publish a frame containing one stationary person or no detections."""
        self.source.frame_time += 1
        detections = {}
        if present:
            detection = {
                "id": EVENT_ID,
                "label": "person",
                "frame_time": self.source.frame_time,
                "start_time": 100.0,
                "score": score,
                "box": (100, 100, 200, 200),
                "centroid": (150, 150),
                "area": 10000,
                "ratio": 1.0,
                "region": (0, 0, 320, 240),
                "motionless_count": 100,
                "position_changes": self.position_changes,
                "attributes": [],
            }
            if EVENT_ID not in self.state.tracked_objects:
                detection["score_history"] = [score] * 3
            detections[EVENT_ID] = detection
        self.state.update(
            f"person-{self.source.frame_time}",
            self.source.frame_time,
            detections,
            [],
            [],
        )

    def start(self, score=0.9):
        """Allow initial true-positive and MQTT publication updates to settle."""
        for _ in range(4):
            self.advance(score=score)


class TestLiveTrackSnapshot(unittest.TestCase):
    def setUp(self):
        self.fixture = LivePersonFixture()
        self.addCleanup(self.fixture.close)
        self.state = self.fixture.state

    def test_unpublished_and_fresh_empty_snapshots_are_distinct(self):
        self.assertIsNone(self.state.get_live_tracks())
        self.fixture.advance(present=False)
        self.assertEqual(
            self.state.get_live_tracks(), (self.fixture.source.frame_time, ())
        )

    def test_stationary_true_positive_survives_mutable_object_changes(self):
        self.fixture.start()
        obj = self.state.tracked_objects[EVENT_ID]
        self.assertFalse(obj.false_positive)
        self.assertFalse(obj.is_active())
        before = self.state.get_live_tracks()
        self.assertEqual(
            before,
            (self.fixture.source.frame_time, ((EVENT_ID, "person", None),)),
        )
        obj.obj_data.update(id="changed", label="car", end_time=999)
        obj.path_data.extend([((0.5, 0.5), 100.0)] * 10000)
        self.assertEqual(self.state.get_live_tracks(), before)

    def test_metadata_advances_only_when_the_new_frame_is_published(self):
        self.fixture.advance()
        before = self.state.get_live_tracks()
        self.assertEqual(before[1], ())
        observed = []
        self.state.on(
            "camera_activity",
            lambda *_args: observed.append(self.state.get_live_tracks()),
        )
        self.fixture.advance()
        self.assertEqual(observed, [before])
        self.assertEqual(
            self.state.get_live_tracks(),
            (self.fixture.source.frame_time, ((EVENT_ID, "person", None),)),
        )

    def test_false_positive_is_never_published_as_live_metadata(self):
        self.fixture.start(score=0.1)
        self.assertTrue(self.state.tracked_objects[EVENT_ID].false_positive)
        self.assertEqual(self.state.get_live_tracks()[1], ())

    def test_real_end_time_is_retained_until_native_cleanup_is_published(self):
        self.fixture.start()
        self.fixture.advance(present=False)
        ended_at = self.fixture.source.frame_time
        self.assertEqual(
            self.state.get_live_tracks(),
            (ended_at, ((EVENT_ID, "person", ended_at),)),
        )
        self.state.finished(EVENT_ID)
        self.fixture.advance(present=False)
        self.assertEqual(self.state.get_live_tracks()[1], ())

    def test_missing_frame_invalidates_the_snapshot_until_a_new_publication(self):
        self.fixture.start()
        image = self.fixture.processor.frame_manager.get_captured_frame.return_value
        self.fixture.processor.frame_manager.get_captured_frame.return_value = None
        self.fixture.advance()
        self.assertIsNone(self.state.get_live_tracks())
        self.fixture.processor.frame_manager.get_captured_frame.return_value = image
        self.fixture.advance()
        self.assertEqual(
            self.state.get_live_tracks(),
            (self.fixture.source.frame_time, ((EVENT_ID, "person", None),)),
        )

    def test_resolution_reset_invalidates_snapshot_before_a_new_frame(self):
        self.fixture.start()
        self.state.camera_config.detect.width = 160
        self.state.camera_config.detect.height = 120
        self.state._discard_stale_resolution_state({})
        self.assertIsNone(self.state.get_live_tracks())
