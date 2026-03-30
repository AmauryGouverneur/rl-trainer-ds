import os
import dash

from .layout import build_layout

# assets/ sits one level above this package directory
_ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

app = dash.Dash(
    __name__,
    assets_folder=os.path.abspath(_ASSETS),
    suppress_callback_exceptions=True,
)
app.title = "RL Trainer DS"
app.layout = build_layout


def run():
    from . import callbacks  # noqa: F401  register callbacks
    app.run(debug=True, dev_tools_ui=False, dev_tools_props_check=False)
