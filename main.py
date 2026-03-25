"""BTCDump - Professional Bitcoin Signal Tool.

Usage:
    python main.py          # CLI mode
    python main.py --web    # Web UI mode
    python run_web.py       # Web UI mode (shortcut)
"""

import sys


def main() -> None:
    if "--web" in sys.argv:
        from btcdump.web.server import run_server
        port = 8000
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        run_server(port=port)
    else:
        from btcdump.app import BTCDumpApp
        app = BTCDumpApp()
        app.run()


if __name__ == "__main__":
    main()
