"""Facilitates communication between processes."""

from typing import Any

from frigate.events.types import EventStateEnum, EventTypeEnum

from .zmq_proxy import Publisher, Subscriber

# (source type, state, camera, frame name, frame clock, event data). The frame
# clock is the exact capture clock of the named frame, so a subscriber reads
# those pixels by the clock that belongs to this message. Events that name no
# frame carry 0.0, which the exact-clock read refuses.
EventUpdate = tuple[
    EventTypeEnum, EventStateEnum, str | None, str, float, dict[str, Any]
]


class EventUpdatePublisher(Publisher[EventUpdate]):
    """Publishes events (objects, audio, manual)."""

    topic_base = "event/"

    def __init__(self) -> None:
        super().__init__("update")

    def publish(
        self,
        payload: EventUpdate,
        sub_topic: str = "",
    ) -> None:
        super().publish(payload, sub_topic)


class EventUpdateSubscriber(Subscriber):
    """Receives event updates."""

    topic_base = "event/"

    def __init__(self) -> None:
        super().__init__("update")


class EventEndPublisher(
    Publisher[tuple[EventTypeEnum, EventStateEnum, str, dict[str, Any]]]
):
    """Publishes events that have ended."""

    topic_base = "event/"

    def __init__(self) -> None:
        super().__init__("finalized")

    def publish(
        self,
        payload: tuple[EventTypeEnum, EventStateEnum, str, dict[str, Any]],
        sub_topic: str = "",
    ) -> None:
        super().publish(payload, sub_topic)


class EventEndSubscriber(Subscriber):
    """Receives events that have ended."""

    topic_base = "event/"

    def __init__(self) -> None:
        super().__init__("finalized")
