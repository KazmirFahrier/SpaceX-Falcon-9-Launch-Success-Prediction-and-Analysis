"""WSGI entry point for the Falcon 9 dashboard."""

from spacex_falcon.dashboard import create_app

app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8051)
