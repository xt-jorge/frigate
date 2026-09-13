"""Facilitates communication between processes for object detection signals."""

import threading

import numpy as np
import zmq

SOCKET_PUB = "ipc:///tmp/cache/detector_pub"
SOCKET_SUB = "ipc:///tmp/cache/detector_sub"


class ZmqProxyRunner(threading.Thread):
    def __init__(self, context: zmq.Context[zmq.Socket]) -> None:
        super().__init__(name="detector_proxy")
        self.context = context

    def run(self) -> None:
        """Run the proxy."""
        incoming = self.context.socket(zmq.XSUB)
        incoming.bind(SOCKET_PUB)
        outgoing = self.context.socket(zmq.XPUB)
        outgoing.bind(SOCKET_SUB)

        # Blocking: This will unblock (via exception) when we destroy the context
        # The incoming and outgoing sockets will be closed automatically
        # when the context is destroyed as well.
        try:
            zmq.proxy(incoming, outgoing)
        except zmq.ZMQError:
            pass


class DetectorProxy:
    """Proxies object detection signals."""

    def __init__(self) -> None:
        self.context = zmq.Context()
        self.runner = ZmqProxyRunner(self.context)
        self.runner.start()

    def stop(self) -> None:
        # destroying the context will tell the proxy to stop
        self.context.destroy()
        self.runner.join()


class ObjectDetectorPublisher:
    """Publishes signal for object detection to different processes."""

    topic_base = "object_detector/"

    def __init__(self, topic: str = "") -> None:
        self.topic = f"{self.topic_base}{topic}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(SOCKET_PUB)

    def publish(
        self, sub_topic: str, request_id: str, detections: np.ndarray | None
    ) -> None:
        """Publish one generation and its immutable bounded output snapshot."""
        payload = b""
        if detections is not None:
            output = np.asarray(detections, dtype=np.float32)
            if output.shape == (20, 6) and np.isfinite(output).all():
                payload = output.tobytes()
        self.socket.send_multipart(
            [f"{self.topic}{sub_topic}/".encode(), request_id.encode(), payload]
        )

    def stop(self) -> None:
        self.socket.close()
        self.context.destroy()


class ObjectDetectorSubscriber:
    """Simplifies receiving a signal for object detection."""

    topic_base = "object_detector/"

    def __init__(self, topic: str = "") -> None:
        self.topic = f"{self.topic_base}{topic}/"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.socket.connect(SOCKET_SUB)

    def check_for_update(
        self, timeout: float = 5
    ) -> tuple[str, np.ndarray | None] | None:
        """Returns message or None if no update."""
        try:
            has_update, _, _ = zmq.select([self.socket], [], [], timeout)

            if has_update:
                parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) != 3 or parts[0] != self.topic.encode():
                    return None
                request_id = parts[1].decode("ascii")
                if len(parts[2]) != 20 * 6 * 4:
                    return request_id, None
                output = np.frombuffer(parts[2], dtype=np.float32).reshape((20, 6))
                return request_id, output if np.isfinite(output).all() else None
        except (zmq.ZMQError, UnicodeDecodeError):
            pass

        return None

    def stop(self) -> None:
        self.socket.close()
        self.context.destroy()
