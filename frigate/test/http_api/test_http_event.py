from datetime import datetime
from typing import Any
from unittest.mock import Mock

from fastapi.testclient import TestClient
from playhouse.shortcuts import model_to_dict

from frigate.api.auth import get_allowed_cameras_for_filter, get_current_user
from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.models import Event, Recordings, ReviewSegment, Timeline
from frigate.stats.emitter import StatsEmitter
from frigate.test.http_api.base_http_test import AuthTestClient, BaseTestHttp, Request
from frigate.test.test_storage import _insert_mock_event


class TestHttpApp(BaseTestHttp):
    def setUp(self):
        super().setUp([Event, Recordings, ReviewSegment, Timeline])
        self.app = super().create_app()

        # Mock get_current_user for all tests
        async def mock_get_current_user(request: Request):
            username = request.headers.get("remote-user")
            role = request.headers.get("remote-role")
            if not username or not role:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    content={"message": "No authorization headers."}, status_code=401
                )
            return {"username": username, "role": role}

        self.app.dependency_overrides[get_current_user] = mock_get_current_user

        async def mock_get_allowed_cameras_for_filter(request: Request):
            return ["front_door"]

        self.app.dependency_overrides[get_allowed_cameras_for_filter] = (
            mock_get_allowed_cameras_for_filter
        )

    def tearDown(self):
        self.app.dependency_overrides.clear()
        super().tearDown()

    ####################################################################################################################
    ###################################  GET /events Endpoint   #########################################################
    ####################################################################################################################
    def test_get_event_list_no_events(self):
        with AuthTestClient(self.app) as client:
            events = client.get("/events").json()
            assert len(events) == 0

    def test_get_event_track_view_excludes_large_history_and_thumbnail(self):
        event_id = "123456.long-lived"
        path_data = [[[0.25, 0.75], timestamp] for timestamp in range(10000)]
        with AuthTestClient(self.app) as client:
            super().insert_mock_event(event_id, data={"path_data": path_data})
            Event.update(label="person", end_time=None, thumbnail="thumbnail").where(
                Event.id == event_id
            ).execute()

            full_response = client.get("/events")
            assert full_response.status_code == 200
            assert len(full_response.content) > 64 * 1024
            assert full_response.json()[0]["data"]["path_data"] == path_data
            assert full_response.json()[0]["thumbnail"] == "thumbnail"
            assert client.get("/events", params={"view": "full"}).json() == (
                full_response.json()
            )

            for thumbnail_params in (
                {},
                {"include_thumbnails": 0},
                {"include_thumbnails": 1},
            ):
                with self.subTest(thumbnail_params=thumbnail_params):
                    response = client.get(
                        "/events", params={"view": "track", **thumbnail_params}
                    )
                    assert response.status_code == 200
                    assert response.json() == [
                        {
                            "id": event_id,
                            "camera": "front_door",
                            "label": "person",
                            "end_time": None,
                        }
                    ]
                    assert len(response.content) < 64 * 1024

    def test_get_event_track_view_preserves_filters_sort_and_limit(self):
        fixtures = [
            ("active", "front_door", "person", 100, None, 0.8),
            ("newer", "front_door", "person", 110, None, 0.6),
            ("ended", "front_door", "person", 90, 120, 0.9),
            ("other-label", "front_door", "car", 120, None, 0.7),
            ("other-camera", "back_door", "person", 130, None, 1.0),
        ]
        with AuthTestClient(self.app) as client:
            for event_id, camera, label, start, end, score in fixtures:
                super().insert_mock_event(
                    event_id, camera=camera, start_time=start, data={"score": score}
                )
                Event.update(label=label, end_time=end).where(
                    Event.id == event_id
                ).execute()

            cases = [
                ({}, ["other-label", "newer", "active", "ended"]),
                ({"labels": "person", "in_progress": 1}, ["newer", "active"]),
                ({"labels": "person", "in_progress": 0}, ["ended"]),
                ({"labels": "person", "limit": 1}, ["newer"]),
                (
                    {"labels": "person", "sort": "score_desc"},
                    ["ended", "active", "newer"],
                ),
                ({"labels": "person", "min_score": 0.7}, ["active", "ended"]),
                ({"after": 95, "before": 105}, ["active"]),
                (
                    {"cameras": "front_door,back_door", "labels": "person"},
                    ["newer", "active", "ended"],
                ),
                ({"cameras": "back_door"}, []),
                ({"camera": "back_door"}, []),
                ({"event_id": "other-camera"}, []),
                ({"event_id": "ended", "labels": "person"}, ["ended"]),
                ({"event_id": "missing"}, []),
            ]
            for params, expected_ids in cases:
                with self.subTest(params=params):
                    response = client.get("/events", params={"view": "track", **params})
                    assert response.status_code == 200
                    events = response.json()
                    assert [event["id"] for event in events] == expected_ids
                    assert events == [
                        {
                            key: event[key]
                            for key in ("id", "camera", "label", "end_time")
                        }
                        for event in client.get("/events", params=params).json()
                    ]
                    for event in events:
                        assert event["end_time"] == (
                            120 if event["id"] == "ended" else None
                        )

    def test_get_event_track_view_requires_authentication(self):
        with TestClient(self.app) as client:
            for view in ("full", "track"):
                with self.subTest(view=view):
                    response = client.get("/events", params={"view": view})
                    assert response.status_code == 401
        with AuthTestClient(self.app) as client:
            response = client.get(
                "/events",
                params={"view": "track"},
                headers={"remote-user": "viewer", "remote-role": "viewer"},
            )
            assert response.status_code == 200

    def test_get_event_list_rejects_unknown_view(self):
        with AuthTestClient(self.app) as client:
            response = client.get("/events", params={"view": "unknown"})
            assert response.status_code == 422

    def test_get_event_list_no_match_event_id(self):
        id = "123456.random"
        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id)
            events = client.get("/events", params={"event_id": "abc"}).json()
            assert len(events) == 0

    def test_get_event_list_match_event_id(self):
        id = "123456.random"
        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id)
            events = client.get("/events", params={"event_id": id}).json()
            assert len(events) == 1
            assert events[0]["id"] == id

    def test_get_event_list_match_length(self):
        now = int(datetime.now().timestamp())

        id = "123456.random"
        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id, now, now + 1)
            events = client.get(
                "/events", params={"max_length": 1, "min_length": 1}
            ).json()
            assert len(events) == 1
            assert events[0]["id"] == id

    def test_get_event_list_no_match_max_length(self):
        now = int(datetime.now().timestamp())

        with AuthTestClient(self.app) as client:
            id = "123456.random"
            super().insert_mock_event(id, now, now + 2)
            events = client.get("/events", params={"max_length": 1}).json()
            assert len(events) == 0

    def test_get_event_list_no_match_min_length(self):
        now = int(datetime.now().timestamp())

        with AuthTestClient(self.app) as client:
            id = "123456.random"
            super().insert_mock_event(id, now, now + 2)
            events = client.get("/events", params={"min_length": 3}).json()
            assert len(events) == 0

    def test_get_event_list_limit(self):
        now = datetime.now().timestamp()
        id = "123456.random"
        id2 = "54321.random"

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id, start_time=now + 1)
            events = client.get("/events").json()
            assert len(events) == 1
            assert events[0]["id"] == id

            super().insert_mock_event(id2, start_time=now)
            events = client.get("/events").json()
            assert len(events) == 2

            events = client.get("/events", params={"limit": 1}).json()
            assert len(events) == 1
            assert events[0]["id"] == id

            events = client.get("/events", params={"limit": 3}).json()
            assert len(events) == 2

    def test_get_event_list_no_match_has_clip(self):
        now = int(datetime.now().timestamp())

        with AuthTestClient(self.app) as client:
            id = "123456.random"
            super().insert_mock_event(id, now, now + 2)
            events = client.get("/events", params={"has_clip": 0}).json()
            assert len(events) == 0

    def test_get_event_list_has_clip(self):
        with AuthTestClient(self.app) as client:
            id = "123456.random"
            super().insert_mock_event(id, has_clip=True)
            events = client.get("/events", params={"has_clip": 1}).json()
            assert len(events) == 1
            assert events[0]["id"] == id

    def test_get_event_list_sort_score(self):
        with AuthTestClient(self.app) as client:
            id = "123456.random"
            id2 = "54321.random"
            super().insert_mock_event(id, top_score=37, score=37, data={"score": 50})
            super().insert_mock_event(id2, top_score=47, score=47, data={"score": 20})
            events = client.get("/events", params={"sort": "score_asc"}).json()
            assert len(events) == 2
            assert events[0]["id"] == id2
            assert events[1]["id"] == id

            events = client.get("/events", params={"sort": "score_desc"}).json()
            assert len(events) == 2
            assert events[0]["id"] == id
            assert events[1]["id"] == id2

    def test_get_event_list_sort_start_time(self):
        now = int(datetime.now().timestamp())

        with AuthTestClient(self.app) as client:
            id = "123456.random"
            id2 = "54321.random"
            super().insert_mock_event(id, start_time=now + 3)
            super().insert_mock_event(id2, start_time=now)
            events = client.get("/events", params={"sort": "date_asc"}).json()
            assert len(events) == 2
            assert events[0]["id"] == id2
            assert events[1]["id"] == id

            events = client.get("/events", params={"sort": "date_desc"}).json()
            assert len(events) == 2
            assert events[0]["id"] == id
            assert events[1]["id"] == id2

    def test_get_event_list_match_multilingual_attribute(self):
        event_id = "123456.zh"
        attribute = "中文标签"

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(event_id, data={"custom_attr": attribute})

            events = client.get("/events", params={"attributes": attribute}).json()
            assert len(events) == 1
            assert events[0]["id"] == event_id

            events = client.get(
                "/events", params={"attributes": "%E4%B8%AD%E6%96%87%E6%A0%87%E7%AD%BE"}
            ).json()
            assert len(events) == 1
            assert events[0]["id"] == event_id

    def test_events_search_match_multilingual_attribute(self):
        event_id = "123456.zh.search"
        attribute = "中文标签"
        mock_embeddings = Mock()
        mock_embeddings.search_thumbnail.return_value = [(event_id, 0.05)]

        self.app.frigate_config.semantic_search.enabled = True
        self.app.embeddings = mock_embeddings

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(event_id, data={"custom_attr": attribute})

            events = client.get(
                "/events/search",
                params={
                    "search_type": "similarity",
                    "event_id": event_id,
                    "attributes": attribute,
                },
            ).json()
            assert len(events) == 1
            assert events[0]["id"] == event_id

            events = client.get(
                "/events/search",
                params={
                    "search_type": "similarity",
                    "event_id": event_id,
                    "attributes": "%E4%B8%AD%E6%96%87%E6%A0%87%E7%AD%BE",
                },
            ).json()
            assert len(events) == 1
            assert events[0]["id"] == event_id

    def test_similarity_search_hides_unauthorized_anchor_event(self):
        mock_embeddings = Mock()
        self.app.frigate_config.semantic_search.enabled = True
        self.app.embeddings = mock_embeddings

        with AuthTestClient(self.app) as client:
            super().insert_mock_event("hidden.anchor", camera="back_door")
            response = client.get(
                "/events/search",
                params={
                    "search_type": "similarity",
                    "event_id": "hidden.anchor",
                },
            )

        assert response.status_code == 404
        assert response.json()["message"] == "Event not found"
        mock_embeddings.search_thumbnail.assert_not_called()

    def test_get_good_event(self):
        id = "123456.random"

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id)
            event = client.get(f"/events/{id}").json()

        assert event
        assert event["id"] == id
        assert event["id"] == model_to_dict(Event.get(Event.id == id))["id"]

    def test_get_bad_event(self):
        id = "123456.random"
        bad_id = "654321.other"

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id)
            event_response = client.get(f"/events/{bad_id}")
            assert event_response.status_code == 404
            assert event_response.json() == "Event not found"

    def test_delete_event(self):
        id = "123456.random"

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id)
            event = client.get(f"/events/{id}").json()
            assert event
            assert event["id"] == id
            response = client.delete(f"/events/{id}", headers={"remote-role": "admin"})
            assert response.status_code == 200
            event_after_delete = client.get(f"/events/{id}")
            assert event_after_delete.status_code == 404

    def test_event_retention(self):
        id = "123456.random"

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(id)
            client.post(f"/events/{id}/retain", headers={"remote-role": "admin"})
            event = client.get(f"/events/{id}").json()
            assert event
            assert event["id"] == id
            assert event["retain_indefinitely"] is True
            client.delete(f"/events/{id}/retain", headers={"remote-role": "admin"})
            event = client.get(f"/events/{id}").json()
            assert event
            assert event["id"] == id
            assert event["retain_indefinitely"] is False

    def test_event_time_filtering(self):
        morning_id = "123456.random"
        evening_id = "654321.random"
        morning = 1656590400  # 06/30/2022 6 am (GMT)
        evening = 1656633600  # 06/30/2022 6 pm (GMT)

        with AuthTestClient(self.app) as client:
            super().insert_mock_event(morning_id, morning)
            super().insert_mock_event(evening_id, evening)
            # both events come back
            events = client.get("/events").json()
            assert events
            assert len(events) == 2
            # morning event is excluded
            events = client.get(
                "/events",
                params={"time_range": "07:00,24:00"},
            ).json()
            assert events
            assert len(events) == 1
            # evening event is excluded
            events = client.get(
                "/events",
                params={"time_range": "00:00,18:00"},
            ).json()
            assert events
            assert len(events) == 1

    def test_set_delete_sub_label(self):
        mock_event_updater = Mock(spec=EventMetadataPublisher)
        app = super().create_app(event_metadata_publisher=mock_event_updater)
        id = "123456.random"
        sub_label = "sub"

        def update_event(payload: Any, topic: str):
            event = Event.get(id=id)
            event.sub_label = payload[1]
            event.save()

        mock_event_updater.publish.side_effect = update_event

        with AuthTestClient(app) as client:
            super().insert_mock_event(id)
            new_sub_label_response = client.post(
                f"/events/{id}/sub_label",
                json={"subLabel": sub_label},
                headers={"remote-role": "admin"},
            )
            assert new_sub_label_response.status_code == 200
            event = client.get(f"/events/{id}").json()
            assert event
            assert event["id"] == id
            assert event["sub_label"] == sub_label
            empty_sub_label_response = client.post(
                f"/events/{id}/sub_label",
                json={"subLabel": ""},
                headers={"remote-role": "admin"},
            )
            assert empty_sub_label_response.status_code == 200
            event = client.get(f"/events/{id}").json()
            assert event
            assert event["id"] == id
            assert event["sub_label"] == None

    def test_sub_label_list(self):
        mock_event_updater = Mock(spec=EventMetadataPublisher)
        app = super().create_app(event_metadata_publisher=mock_event_updater)
        app.event_metadata_publisher = mock_event_updater
        id = "123456.random"
        sub_label = "sub"

        def update_event(payload: Any, _: str):
            event = Event.get(id=id)
            event.sub_label = payload[1]
            event.save()

        mock_event_updater.publish.side_effect = update_event

        with AuthTestClient(app) as client:
            super().insert_mock_event(id)
            client.post(
                f"/events/{id}/sub_label",
                json={"subLabel": sub_label},
                headers={"remote-role": "admin"},
            )
            sub_labels = client.get("/sub_labels").json()
            assert sub_labels
            assert sub_labels == [sub_label]

    ####################################################################################################################
    ###################################  GET /metrics Endpoint   #########################################################
    ####################################################################################################################
    def test_get_metrics(self):
        """ensure correct prometheus metrics api response"""
        with AuthTestClient(self.app) as client:
            ts_start = datetime.now().timestamp()
            ts_end = ts_start + 30
            _insert_mock_event(
                id="abcde.random", start=ts_start, end=ts_end, retain=True
            )
            _insert_mock_event(
                id="01234.random", start=ts_start, end=ts_end, retain=True
            )
            _insert_mock_event(
                id="56789.random", start=ts_start, end=ts_end, retain=True
            )
            _insert_mock_event(
                id="101112.random",
                label="outside",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="131415.random",
                label="outside",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="161718.random",
                camera="porch",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="192021.random",
                camera="porch",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="222324.random",
                camera="porch",
                label="inside",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="252627.random",
                camera="porch",
                label="inside",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="282930.random",
                label="inside",
                start=ts_start,
                end=ts_end,
                retain=True,
            )
            _insert_mock_event(
                id="313233.random",
                label="inside",
                start=ts_start,
                end=ts_end,
                retain=True,
            )

            stats_emitter = Mock(spec=StatsEmitter)
            stats_emitter.get_latest_stats.return_value = self.test_stats
            self.app.stats_emitter = stats_emitter
            event = client.get("/metrics")

        assert "# TYPE frigate_detection_total_fps gauge" in event.text
        assert "frigate_detection_total_fps 13.7" in event.text
        assert (
            "# HELP frigate_camera_events_total Count of camera events since exporter started"
            in event.text
        )
        assert "# TYPE frigate_camera_events_total counter" in event.text
        assert (
            'frigate_camera_events_total{camera="front_door",label="Mock"} 3.0'
            in event.text
        )
        assert (
            'frigate_camera_events_total{camera="front_door",label="inside"} 2.0'
            in event.text
        )
        assert (
            'frigate_camera_events_total{camera="front_door",label="outside"} 2.0'
            in event.text
        )
        assert (
            'frigate_camera_events_total{camera="porch",label="Mock"} 2.0' in event.text
        )
        assert 'frigate_camera_events_total{camera="porch",label="inside"} 2.0'
