import json
import pickle
import socket
import time

from worker_runner import run_worker

CONFIG_PATH = "config.json"


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_left_right_neighbor_ip(left_peer_rank, right_peer_rank):
    config = load_json_config(CONFIG_PATH)
    ip_list = config["ip_list"]
    left_ip = ip_list[left_peer_rank]
    right_ip = ip_list[right_peer_rank]
    return left_ip, right_ip


def create_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class SocketEndpoint:
    def __init__(
        self,
        conn: socket.socket,
        listener: socket.socket | None = None,
        *,
        rank: int | None = None,
        direction: str | None = None,
    ):
        self._conn = conn
        self._listener = listener
        self._rank = rank
        self._direction = direction

    def send(self, payload):
        try:
            print(
                f"[socket.send] rank={self._rank} direction={self._direction} starting",
                flush=True,
            )
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            header = len(raw).to_bytes(4, byteorder="big")
            self._conn.sendall(header + raw)
            print(
                f"[socket.send] rank={self._rank} direction={self._direction} completed bytes={len(header) + len(raw)}",
                flush=True,
            )
        except Exception as error:
            try:
                peer = self._conn.getpeername()
            except Exception:
                peer = None
            print(
                f"[socket.send] rank={self._rank} direction={self._direction} peer={peer} error={error}",
                flush=True,
            )
            raise

    def recv(self):
        try:
            print(
                f"[socket.recv] rank={self._rank} direction={self._direction} starting",
                flush=True,
            )
            data = bytearray()
            while len(data) < 4:
                chunk = self._conn.recv(4 - len(data))
                if not chunk:
                    raise ConnectionError("socket closed while receiving payload")
                data.extend(chunk)
            header = bytes(data)
            size = int.from_bytes(header, byteorder="big")
            data = bytearray()
            while len(data) < size:
                chunk = self._conn.recv(size - len(data))
                if not chunk:
                    raise ConnectionError("socket closed while receiving payload")
                data.extend(chunk)
            raw = bytes(data)
            print(
                f"[socket.recv] rank={self._rank} direction={self._direction} completed bytes={4 + size}",
                flush=True,
            )
            return pickle.loads(raw)
        except Exception as error:
            try:
                peer = self._conn.getpeername()
            except Exception:
                peer = None
            print(
                f"[socket.recv] rank={self._rank} direction={self._direction} peer={peer} error={error}",
                flush=True,
            )
            raise

    def close(self):
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()

        if self._listener is not None:
            self._listener.close()


def build_distributed_topology(algo, rank):
    if algo == "ring":
        config = load_json_config(CONFIG_PATH)
        local_ip = config["ip_list"][rank]
        base_port = int(config.get("base_port", 5000))
        local_port = base_port + rank
        left_ip = config["ip_list"][(rank - 1) % config["world_size"]]
        right_ip = config["ip_list"][(rank + 1) % config["world_size"]]
        listener = create_socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(f"[rank {rank}] binding {local_ip}:{base_port}", flush=True)
        listener.bind((local_ip, base_port))
        print(f"[rank {rank}] listening on {local_ip}:{base_port}", flush=True)
        listener.listen(1)

        right_conn = create_socket()
        while True:
            try:
                print(f"[rank {rank}] connecting to {right_ip}:{base_port}", flush=True)
                right_conn.connect((right_ip, base_port))
                break
            except OSError as error:
                print(
                    f"[rank {rank}] connect failed to {right_ip}:{base_port}: {error}",
                    flush=True,
                )
                time.sleep(0.2)

        print(f"[rank {rank}] waiting for accept from {left_ip}:{base_port}", flush=True)
        left_conn, _ = listener.accept()
        return {
            "left_endpoint": SocketEndpoint(left_conn, listener=listener, rank=rank, direction="left"),
            "right_endpoint": SocketEndpoint(right_conn, rank=rank, direction="right"),
            "left_endpoint_info": {
                "peer_rank": (rank - 1) % config["world_size"],
                "direction": "left",
                "transport": "socket",
            },
            "right_endpoint_info": {
                "peer_rank": (rank + 1) % config["world_size"],
                "direction": "right",
                "transport": "socket",
            },
        }

    if algo == "tree":
        return {}

    if algo == "parameter_server":
        return {}

    raise ValueError("Unknown algo")

def launch_distributed(config):
    topo = build_distributed_topology(
        config["algo"],
        config["rank"],
    )

    worker_config = {
        **config,
        **topo,
    }
    run_worker(worker_config)