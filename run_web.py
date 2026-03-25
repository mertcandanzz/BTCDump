"""Start the BTCDump Web UI server."""

from btcdump.web.server import run_server

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8000)
