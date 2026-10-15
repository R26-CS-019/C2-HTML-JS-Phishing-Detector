"""Start the local webpage analysis interface around the frozen model."""
import argparse
from pathlib import Path

from detector_runtime.model import FrozenDetector
from detector_runtime.ui import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local port from 1024 to 65535.")
    try:
        detector = FrozenDetector(args.project_dir, args.freeze_report, args.threads)
        server = make_server(detector, args.port)
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"Interface stopped: {e}")
        return 1
    print("Frozen model verified. No model trained.")
    print(f"Open http://127.0.0.1:{args.port} in your browser.")
    print("Keep this terminal open. Ctrl+C stops the interface.")
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nInterface stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
