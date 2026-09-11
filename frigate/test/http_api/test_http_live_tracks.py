"""The live-track API must work before event persistence and stay camera scoped."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from frigate.api import event
from frigate.api.auth import get_allowed_cameras_for_filter, require_admin_by_default
from frigate.models import Event
from frigate.test.http_api.base_http_test import AuthTestClient, BaseTestHttp
from frigate.test.test_live_track_snapshot import CAMERA, EVENT_ID, LivePersonFixture


class TestHttpLiveTracks(BaseTestHttp):
    def setUp(self):
        super().setUp([Event])
        self.app = super().create_app()
        self.fixture = LivePersonFixture(position_changes=0)
        self.addCleanup(self.fixture.close)
        self.fixture.state.camera_config.snapshots.enabled = True
        self.fixture.state.camera_config.record.enabled = True
        self.app.frigate_config = self.fixture.processor.config
        self.app.detected_frames_processor = self.fixture.processor
        self.app.dependency_overrides.pop(get_allowed_cameras_for_filter)
        self.fixture.start()
        self.now = self.fixture.source.frame_time + 0.1

    def tearDown(self):
        self.app.dependency_overrides.clear()
        super().tearDown()

    def get(self, client, params=None, **kwargs):
        """Read the actual endpoint with a fixed clock and no Event-table access."""
        with (
            patch("frigate.api.event.time.time", return_value=self.now),
            patch.object(Event, "select", side_effect=AssertionError("Event DB read")),
        ):
            return client.get(
                "/tracks", params={"cameras": CAMERA, **(params or {})}, **kwargs
            )

    def add_camera(self, name, snapshot):
        config = self.fixture.state.camera_config.model_copy(deep=True)
        self.app.frigate_config.cameras[name] = config
        state = SimpleNamespace(
            camera_config=config, get_live_tracks=Mock(return_value=snapshot)
        )
        self.fixture.processor.camera_states[name] = state
        return state

    def test_stationary_mqtt_person_absent_from_event_db_has_compact_metadata(self):
        self.assertFalse(Event.select().where(Event.id == EVENT_ID).exists())
        messages = [
            json.loads(call.args[1])
            for call in self.fixture.processor.dispatcher.publish.call_args_list
            if call.args[0] == "events"
        ]
        self.assertTrue(
            any(
                event["after"]["id"] == EVENT_ID
                and event["after"]["label"] == "person"
                and not event["after"]["false_positive"]
                for event in messages
            )
        )
        obj = self.fixture.state.tracked_objects[EVENT_ID]
        self.assertFalse(obj.false_positive)
        self.assertFalse(obj.is_active())
        self.assertEqual(obj.obj_data["position_changes"], 0)
        self.assertFalse(self.fixture.processor.should_save_snapshot(CAMERA, obj))
        self.assertFalse(self.fixture.processor.should_retain_recording(CAMERA, obj))
        self.assertFalse(obj.has_snapshot)
        self.assertFalse(obj.has_clip)
        obj.path_data.extend([((0.5, 0.5), 100.0)] * 10000)
        with AuthTestClient(self.app) as client:
            response = self.get(client, {"event_id": EVENT_ID, "labels": "person"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(
            response.json(),
            [{"id": EVENT_ID, "camera": CAMERA, "label": "person", "end_time": None}],
        )
        self.assertLess(len(response.content), 64 * 1024)

    def test_prepublication_miss_becomes_visible_on_next_snapshot(self):
        self.fixture.state.finished(EVENT_ID)
        self.fixture.advance(present=False)
        self.fixture.advance()
        self.now = self.fixture.source.frame_time + 1.1
        observed = []
        with AuthTestClient(self.app) as client:
            self.fixture.state.on(
                "camera_activity",
                lambda *_args: observed.append(
                    self.get(client, {"event_id": EVENT_ID})
                ),
            )
            self.fixture.advance()
            response = self.get(client, {"event_id": EVENT_ID})
        self.assertEqual(
            [(item.status_code, item.json()) for item in observed], [(200, [])]
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["id"], EVENT_ID)
        self.assertIsNone(response.json()[0]["end_time"])

    def test_live_and_ended_filters_preserve_only_the_actual_end_time(self):
        self.fixture.advance(present=False)
        self.now = self.fixture.source.frame_time + 0.1
        with AuthTestClient(self.app) as client:
            self.assertEqual(self.get(client, {"in_progress": 1}).json(), [])
            response = self.get(client, {"in_progress": 0, "event_id": EVENT_ID})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()[0]["end_time"], self.fixture.source.frame_time
            )
            self.assertEqual(self.get(client, {"labels": "car"}).json(), [])
            self.fixture.state.finished(EVENT_ID)
            self.fixture.advance(present=False)
            self.now = self.fixture.source.frame_time + 0.1
            self.assertEqual(self.get(client, {"event_id": EVENT_ID}).json(), [])

    def test_unavailable_camera_never_blocks_another_selected_camera(self):
        healthy = self.add_camera("secondary", (self.now, (("other", "person", None),)))
        with AuthTestClient(self.app) as client:
            for snapshot in (None, (self.now - 5.01, ()), (self.now + 0.01, ())):
                with (
                    self.subTest(snapshot=snapshot),
                    patch.object(
                        self.fixture.state, "get_live_tracks", return_value=snapshot
                    ),
                ):
                    response = self.get(client, {"cameras": f"{CAMERA},secondary"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual([row["id"] for row in response.json()], ["other"])
                    self.assertEqual(
                        self.get(client, {"event_id": EVENT_ID}).status_code, 503
                    )
        healthy.get_live_tracks.assert_called()

    def test_disabled_and_missing_camera_states_are_unknown(self):
        with AuthTestClient(self.app) as client:
            for option in ("camera", "detect", "missing"):
                with self.subTest(option=option):
                    state = self.fixture.state
                    if option == "camera":
                        state.camera_config.enabled = False
                    elif option == "detect":
                        state.camera_config.detect.enabled = False
                    else:
                        self.fixture.processor.camera_states.pop(CAMERA)
                    self.assertEqual(self.get(client).json(), [])
                    self.assertEqual(
                        self.get(client, {"event_id": EVENT_ID}).status_code, 503
                    )
                    state.camera_config.enabled = True
                    state.camera_config.detect.enabled = True
                    self.fixture.processor.camera_states[CAMERA] = state

    def test_authorized_camera_intersection_never_reads_forbidden_state(self):
        forbidden = self.add_camera(
            "private", (self.now, (("secret", "person", None),))
        )
        forbidden.get_live_tracks.side_effect = AssertionError(
            "unauthorized camera read"
        )
        self.app.frigate_config.auth.roles["limited"] = [CAMERA]
        headers = {"remote-user": "limited-user", "remote-role": "limited"}
        with AuthTestClient(self.app) as client:
            response = self.get(
                client, {"cameras": f"private,{CAMERA}"}, headers=headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual([row["camera"] for row in response.json()], [CAMERA])
            self.assertEqual(
                self.get(client, {"cameras": "private"}, headers=headers).json(), []
            )
            self.assertEqual(
                self.get(
                    client,
                    {"cameras": "private", "event_id": "secret"},
                    headers=headers,
                ).json(),
                [],
            )
        forbidden.get_live_tracks.assert_not_called()

    def test_query_bounds_require_explicit_cameras_and_single_camera_exact_lookup(self):
        invalid = [
            {"cameras": ""},
            {"cameras": "all"},
            {"cameras": f"{CAMERA},"},
            {"cameras": f"{CAMERA},secondary", "event_id": EVENT_ID},
            {"limit": 0},
            {"limit": 258},
            {"in_progress": 2},
            {"in_progress": -1},
        ]
        with AuthTestClient(self.app) as client:
            self.assertEqual(client.get("/tracks").status_code, 422)
            for params in invalid:
                with self.subTest(params=params):
                    self.assertEqual(self.get(client, params).status_code, 422)

    def test_limit_preserves_the_257th_row_overflow_witness(self):
        snapshot = (
            self.now,
            tuple((f"person-{index}", "person", None) for index in range(258)),
        )
        with (
            patch.object(self.fixture.state, "get_live_tracks", return_value=snapshot),
            AuthTestClient(self.app) as client,
        ):
            for limit in (1, 256, 257):
                with self.subTest(limit=limit):
                    response = self.get(client, {"limit": limit})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(len(response.json()), limit)
                    self.assertTrue(
                        all(
                            set(row) == {"id", "camera", "label", "end_time"}
                            for row in response.json()
                        )
                    )

    def test_unauthenticated_requests_are_rejected(self):
        with TestClient(self.app) as client:
            self.assertEqual(self.get(client).status_code, 401)

    def test_default_admin_guard_allows_only_camera_scoped_authenticated_reads(self):
        app = FastAPI(dependencies=[Depends(require_admin_by_default())])
        app.include_router(event.router)
        app.frigate_config = self.app.frigate_config
        app.detected_frames_processor = self.fixture.processor
        forbidden = self.add_camera(
            "private", (self.now, (("secret", "person", None),))
        )
        forbidden.get_live_tracks.side_effect = AssertionError(
            "unauthorized camera read"
        )
        app.frigate_config.auth.roles["limited"] = [CAMERA]
        with TestClient(app) as client:
            headers = {"remote-user": "limited-user", "remote-role": "limited"}
            response = self.get(
                client, {"cameras": f"{CAMERA},private"}, headers=headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual([row["camera"] for row in response.json()], [CAMERA])
            self.assertEqual(
                self.get(client, {"cameras": "private"}, headers=headers).json(), []
            )
            self.assertEqual(self.get(client).status_code, 401)
            self.assertEqual(
                self.get(
                    client,
                    headers={"remote-user": "anonymous", "remote-role": "viewer"},
                ).status_code,
                401,
            )
        forbidden.get_live_tracks.assert_not_called()
