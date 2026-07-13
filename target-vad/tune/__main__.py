"""python3 -m tune [--port 8765] [--host 127.0.0.1] [--config config.yaml]"""

import argparse
import os

from tune.kiosk_proc import KioskProcess
from tune.server import TuningServer


def main():
    ap = argparse.ArgumentParser(description="Kiosk tuning console")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to tune from another machine on the LAN")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    kproc = KioskProcess(cwd=os.path.dirname(config_path))
    server = TuningServer(config_path=config_path, kproc=kproc,
                          host=args.host, port=args.port)
    print(f"[tune] console at http://{args.host}:{server.port}/  "
          f"(config: {config_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        kproc.stop()       # never orphan a kiosk
        server.shutdown()
        print("[tune] stopped.")


if __name__ == "__main__":
    main()
