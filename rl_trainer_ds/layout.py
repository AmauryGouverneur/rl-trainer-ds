from dash import dcc, html
import plotly.graph_objects as go

LIBRARY = {
    "CLASSIC CONTROL": ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1", "Acrobot-v1"],
    "BOX2D": ["LunarLander-v2", "LunarLanderContinuous-v2", "BipedalWalker-v3", "CarRacing-v2"],
    "MUJOCO": ["HalfCheetah-v4", "Ant-v4", "Hopper-v4", "Walker2d-v4", "Humanoid-v4", "Swimmer-v4", "Reacher-v2"],
    "ATARI": ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"],
}

ALGOS = [
    ("PPO", "on-policy"),
    ("SAC", "off-pol"),
    ("A2C", "on-policy"),
    ("DQN", "off-pol"),
]

ALGO_DEFAULTS = {
    "PPO": {"learning_rate": "3e-4", "n_steps": "2048", "batch_size": "64", "n_epochs": "10", "gamma": "0.99"},
    "SAC": {"learning_rate": "3e-4", "buffer_size": "1000000", "batch_size": "256", "gamma": "0.99", "tau": "0.005"},
    "A2C": {"learning_rate": "7e-4", "n_steps": "5", "gamma": "0.99"},
    "DQN": {"learning_rate": "1e-4", "buffer_size": "1000000", "batch_size": "32", "gamma": "0.99"},
}


def _speaker():
    return html.Div(className="speaker-dots", children=[
        html.Div(className="speaker-dot") for _ in range(8)
    ])


_CHART_AXIS_STYLE = dict(
    color="#4a7a9a",
    tickfont=dict(family="Share Tech Mono", size=8, color="#4a7a9a"),
    showline=False, zeroline=False,
)

def _empty_chart():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0a0f1a",
        plot_bgcolor="#0a0f1a",
        margin=dict(l=10, r=10, t=5, b=5),
        xaxis=dict(
            visible=True, showgrid=False, autorange=True,
            tickformat=",d",
            **_CHART_AXIS_STYLE,
        ),
        yaxis=dict(
            visible=True, showgrid=True, autorange=True,
            gridcolor="#1a2535", gridwidth=0.5,
            tickformat=".1f",
            **_CHART_AXIS_STYLE,
        ),
        showlegend=False,
    )
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="lines",
        line=dict(color="#4a9eff", width=1.5),
        hoverinfo="none",
    ))
    return fig


def _algo_params_panel():
    """Inline param inputs rendered via callback; placeholder here."""
    return html.Div(id="algo-params-panel", style={"display": "none"})


def build_layout():
    return html.Div(className="ds-wrapper", children=[
        # ── Stores ──────────────────────────────────────────────────
        dcc.Store(id="env-store", data={}),
        dcc.Store(id="algo-store", data={"algo": "PPO", "params": ALGO_DEFAULTS["PPO"]}),
        dcc.Store(id="agent-store", data={
            "arch": "MLP",
            "obs_norm": True, "frame_stack": False, "frame_stack_n": 4,
            "act_clip": True, "rew_clip": False, "rew_scale": False, "rew_scale_v": 1.0,
            "wandb_enabled": False, "wandb_project": "rl-trainer-ds", "wandb_run": "",
        }),
        dcc.Store(id="training-active", data=False),
        dcc.Interval(id="chart-interval", interval=500, disabled=True),
        dcc.Store(id="lib-state", data={"collapsed": {}, "selected": ""}),
        # Custom loader stores
        dcc.Store(id="custom-env-store", data={}),   # {class_name, env_id, filepath}
        dcc.Store(id="custom-algo-store", data={}),  # {class_name, filepath}

        # ── DS Body ─────────────────────────────────────────────────
        html.Div(style={"position": "relative"}, children=[
            html.Div(className="ds-body", children=[

                # ══ TOP HALF ══════════════════════════════════════
                html.Div(className="ds-top-half", children=[
                    html.Div("L", className="shoulder-btn shoulder-l"),
                    html.Div("R", className="shoulder-btn shoulder-r"),
                    html.Button("WB", id="wb-btn", n_clicks=0,
                                className="shoulder-mini shoulder-wb dimmed",
                                title="Start training first"),
                    html.Div(id="power-led", className="power-led"),
                    html.Div(className="wifi-led"),

                    # Speakers
                    html.Div(className="speaker-row", children=[_speaker(), _speaker()]),

                    # Top Screen bezel
                    html.Div(className="screen-bezel", children=[
                        html.Div(className="screen top-screen", children=[
                            # Header
                            html.Div(className="screen-header", children=[
                                html.Span("ALGO SELECT", className="screen-title"),
                                html.Span(id="env-display", children="▶ no cartridge",
                                          className="screen-status"),
                            ]),
                            # Algo grid (5 built-in + 1 custom)
                            html.Div(className="algo-grid", children=[
                                *[
                                    html.Div(
                                        id=f"algo-card-{name}",
                                        className="algo-card" + (" selected" if name == "PPO" else ""),
                                        n_clicks=0,
                                        children=[
                                            html.Span(name, className="algo-name"),
                                            html.Span(label, className="algo-type"),
                                        ]
                                    )
                                    for name, label in ALGOS
                                ],
                                # Custom algo card
                                html.Div(
                                    id="algo-card-CUSTOM",
                                    className="algo-card-custom dimmed",
                                    n_clicks=0,
                                    children=[
                                        html.Span(id="custom-algo-card-name",
                                                  children="+ CUST", className="algo-name-custom"),
                                        html.Span(id="custom-algo-card-type",
                                                  children="user algo", className="algo-type-custom"),
                                    ],
                                ),
                            ]),
                            # Reward chart
                            html.Div(className="chart-area", children=[
                                html.Div("EPISODE REWARD", className="chart-label"),
                                dcc.Graph(
                                    id="reward-chart",
                                    figure=_empty_chart(),
                                    config={"displayModeBar": False},
                                    style={"height": "112px"},
                                ),
                            ]),
                        ]),
                    ]),
                ]),

                # ══ HINGE ═════════════════════════════════════════
                html.Div(className="ds-hinge", children=[
                    html.Div(className="ds-hinge-dot"),
                    html.Div(className="ds-hinge-dot"),
                ]),

                # ══ BOTTOM HALF ═══════════════════════════════════
                html.Div(className="ds-bottom-half", children=[
                    html.Div(className="screen-bezel", style={"marginTop": "8px"}, children=[
                        html.Div(className="screen bottom-screen", children=[
                            html.Div(className="screen-header", children=[
                                html.Span("AGENT CONFIG",
                                          className="screen-title screen-title-orange"),
                                html.Span(id="train-status", children="● READY",
                                          className="screen-status"),
                            ]),
                            html.Div(className="bottom-content", children=[
                                # Left: architecture + hyperparams
                                html.Div(className="config-panel", children=[
                                    html.Div("ARCHITECTURE", className="config-panel-title"),
                                    html.Div(className="arch-options", children=[
                                        html.Div(
                                            arch,
                                            id=f"arch-btn-{arch}",
                                            className="arch-btn" + (" active" if arch == "MLP" else ""),
                                            n_clicks=0,
                                        )
                                        for arch in ["MLP", "CNN", "LSTM"]
                                    ]),
                                    # Param rows rendered by callback
                                    html.Div(id="param-rows"),
                                ]),
                                # Right: shaping toggles
                                html.Div(className="config-panel", children=[
                                    html.Div("SHAPING", className="config-panel-title"),
                                    *[
                                        html.Div(className="toggle-row", children=[
                                            html.Span(label, className="toggle-label"),
                                            html.Div(
                                                id=f"toggle-{key}",
                                                className="toggle" + (" on" if default else ""),
                                                n_clicks=0,
                                                children=html.Div(className="toggle-dot"),
                                            ),
                                        ])
                                        for key, label, default in [
                                            ("obs_norm",    "obs norm",    True),
                                            ("frame_stack", "frame stack", False),
                                            ("act_clip",    "act clip",    True),
                                            ("rew_clip",    "rew clip",    False),
                                            ("rew_scale",   "rew scale",   False),
                                        ]
                                    ],
                                    # Conditional extras
                                    html.Div(id="frame-stack-extra"),
                                    html.Div(id="rew-scale-extra"),
                                    # W&B toggle
                                    html.Div(className="config-panel-title",
                                             style={"marginTop": "6px"}, children="LOGGING"),
                                    html.Div(className="toggle-row", children=[
                                        html.Span("wandb", className="toggle-label"),
                                        html.Div(
                                            id="toggle-wandb",
                                            className="toggle",
                                            n_clicks=0,
                                            children=html.Div(className="toggle-dot"),
                                        ),
                                    ]),
                                    html.Div(id="wandb-extras"),
                                ]),
                            ]),
                        ]),
                    ]),

                    # Controls row
                    html.Div(className="ds-controls", children=[
                        # D-pad
                        html.Div(className="dpad", children=[
                            html.Div(className="dpad-h"),
                            html.Div(className="dpad-v"),
                            html.Div(className="dpad-center"),
                        ]),
                        # Center buttons
                        html.Div(className="center-buttons", children=[
                            html.Div(className="start-select", children=[
                                html.Div("SELECT", className="mini-btn"),
                                html.Div("START",  className="mini-btn"),
                            ]),
                            html.Button(
                                "▶ TRAIN",
                                id="train-btn",
                                className="train-btn",
                                n_clicks=0,
                                disabled=True,
                            ),
                        ]),
                        # Face buttons  (TB top-right, Y top-left, A mid-right, B mid-left)
                        html.Div(className="face-buttons", children=[
                            html.Div(style={"width": "20px"}),
                            # Wrapper with display:contents keeps the grid slot intact;
                            # dcc.Tooltip targets its first sibling when targetable=True
                            html.Div(style={"display": "contents"}, children=[
                                html.Div("TB", id="face-tb-btn", n_clicks=0,
                                         className="face-btn btn-tb dimmed"),
                                dcc.Tooltip(
                                    "START TRAINING FIRST",
                                    id="face-tb-tooltip",
                                    targetable=True,
                                ),
                            ]),
                            html.Div("Y", className="face-btn btn-y"),
                            html.Div("A", className="face-btn btn-a"),
                            html.Div("B", className="face-btn btn-b"),
                            html.Div(style={"width": "20px"}),
                        ]),
                    ]),

                    # Cartridge slots (standard + custom)
                    html.Div(className="cartridge-area", children=[
                        # Standard cartridge
                        html.Div(
                            className="cartridge-slot",
                            id="cartridge-slot",
                            n_clicks=0,
                            title="Click to load environment",
                            children=[
                                html.Div(className="cartridge-inserted", children=[
                                    html.Span(id="cart-label", className="cartridge-label",
                                              children="NO CART"),
                                ]),
                            ],
                        ),
                        # Custom env cartridge
                        html.Div(
                            className="cartridge-slot cartridge-slot-custom",
                            id="custom-cart-slot",
                            n_clicks=0,
                            title="Click to load custom .py environment",
                            children=[
                                html.Div(className="cartridge-inserted cartridge-inserted-custom",
                                         children=[
                                    html.Span(id="custom-cart-label",
                                              className="cartridge-label cartridge-label-custom",
                                              children="CUSTOM"),
                                ]),
                            ],
                        ),
                    ]),

                    # Log output (bottom of bottom half)
                    html.Pre(id="log-pre", className="log-pre", children=""),
                ]),
            ]),
        ]),

        # ── Library / env modal ──────────────────────────────────────
        html.Div(
            id="env-modal",
            style={"display": "none"},
            className="modal-overlay",
            children=[
                html.Div(className="modal-box library-box", children=[
                    # Header row
                    html.Div(className="modal-header-row", children=[
                        html.Div("SELECT ENV", className="modal-title"),
                        html.Button("×", id="modal-cancel", n_clicks=0,
                                    className="modal-close-btn"),
                    ]),
                    # Library browser (scrollable)
                    html.Div(className="library-browser", children=[
                        html.Div(className="lib-category", children=[
                            html.Div(
                                children=[
                                    html.Span(id={"type": "cat-arrow", "cat": cat}, children="▼ "),
                                    cat,
                                ],
                                id={"type": "cat-toggle", "cat": cat},
                                className="lib-cat-header",
                                n_clicks=0,
                            ),
                            html.Div(
                                id={"type": "cat-body", "cat": cat},
                                className="lib-cat-body",
                                children=[
                                    html.Div(
                                        env_id,
                                        id={"type": "env-row", "env": env_id},
                                        className="lib-env-row",
                                        n_clicks=0,
                                    )
                                    for env_id in envs
                                ],
                            ),
                        ])
                        for cat, envs in LIBRARY.items()
                    ]),
                    # Custom env input + actions
                    html.Div(className="modal-bottom", children=[
                        dcc.Input(
                            id="env-input",
                            type="text",
                            placeholder="e.g. CartPole-v1  (or type custom)",
                            className="modal-input",
                            debounce=False,
                            value="",
                        ),
                        html.Div(id="env-error", className="modal-error", children=""),
                        html.Div(className="modal-buttons", children=[
                            html.Button("CANCEL",  id="modal-cancel-bottom", n_clicks=0,
                                        className="modal-btn modal-btn-cancel"),
                            html.Button("CONFIRM", id="modal-confirm", n_clicks=0,
                                        className="modal-btn modal-btn-confirm"),
                        ]),
                    ]),
                ]),
            ],
        ),

        # ── Custom env modal ─────────────────────────────────────────
        html.Div(
            id="custom-env-modal",
            style={"display": "none"},
            className="modal-overlay",
            children=[
                html.Div(className="modal-box modal-box-custom", children=[
                    html.Div(className="modal-header-row", children=[
                        html.Div("LOAD CUSTOM ENV", className="modal-title modal-title-custom"),
                        html.Button("×", id="custom-env-modal-cancel", n_clicks=0,
                                    className="modal-close-btn"),
                    ]),
                    dcc.Input(
                        id="custom-env-filepath",
                        type="text",
                        placeholder="/absolute/path/to/env.py",
                        className="modal-input modal-input-custom",
                        debounce=False,
                        value="",
                        style={"marginBottom": "6px"},
                    ),
                    html.Div(id="custom-env-feedback", className="modal-error", children=""),
                    html.Div(id="custom-env-success", className="modal-success", children=""),
                    html.Div(className="modal-buttons", children=[
                        html.Button("CANCEL", id="custom-env-modal-cancel-bottom", n_clicks=0,
                                    className="modal-btn modal-btn-cancel"),
                        html.Button("LOAD", id="custom-env-load-btn", n_clicks=0,
                                    className="modal-btn modal-btn-confirm-custom"),
                        html.Button("CONFIRM", id="custom-env-confirm-btn", n_clicks=0,
                                    className="modal-btn modal-btn-confirm-custom",
                                    disabled=True,
                                    style={"opacity": "0.4"}),
                    ]),
                ]),
            ],
        ),

        # ── Custom algo modal ────────────────────────────────────────
        html.Div(
            id="custom-algo-modal",
            style={"display": "none"},
            className="modal-overlay",
            children=[
                html.Div(className="modal-box modal-box-custom", children=[
                    html.Div(className="modal-header-row", children=[
                        html.Div("LOAD CUSTOM ALGO", className="modal-title modal-title-custom"),
                        html.Button("×", id="custom-algo-modal-cancel", n_clicks=0,
                                    className="modal-close-btn"),
                    ]),
                    dcc.Input(
                        id="custom-algo-filepath",
                        type="text",
                        placeholder="/absolute/path/to/algo.py",
                        className="modal-input modal-input-custom",
                        debounce=False,
                        value="",
                        style={"marginBottom": "6px"},
                    ),
                    html.Div(id="custom-algo-feedback", className="modal-error", children=""),
                    html.Div(id="custom-algo-success", className="modal-success", children=""),
                    html.Div(className="modal-buttons", children=[
                        html.Button("CANCEL", id="custom-algo-modal-cancel-bottom", n_clicks=0,
                                    className="modal-btn modal-btn-cancel"),
                        html.Button("LOAD", id="custom-algo-load-btn", n_clicks=0,
                                    className="modal-btn modal-btn-confirm-custom"),
                        html.Button("CONFIRM", id="custom-algo-confirm-btn", n_clicks=0,
                                    className="modal-btn modal-btn-confirm-custom",
                                    disabled=True,
                                    style={"opacity": "0.4"}),
                    ]),
                ]),
            ],
        ),
    ])
