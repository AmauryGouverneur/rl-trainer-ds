import gymnasium
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, no_update, ALL

from .app import app
from .layout import ALGO_DEFAULTS, ALGOS, LIBRARY
from .training import runner

# ── helpers ──────────────────────────────────────────────────────────


def _truncate_cart(name: str, maxlen: int = 12) -> str:
    upper = name.upper()
    return upper[:maxlen - 1] + "…" if len(upper) > maxlen else upper


_AXIS = dict(
    color="#4a7a9a",
    tickfont=dict(family="Share Tech Mono", size=8, color="#4a7a9a"),
    showline=False, zeroline=False,
)

def _make_chart(rewards, timesteps):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0a0f1a",
        plot_bgcolor="#0a0f1a",
        margin=dict(l=10, r=10, t=5, b=5),
        xaxis=dict(visible=True, showgrid=False, autorange=True,
                   rangemode="tozero", tickformat=",d", **_AXIS),
        yaxis=dict(visible=True, showgrid=True, autorange=True,
                   gridcolor="#1a2535", gridwidth=0.5,
                   tickformat=".1f", **_AXIS),
        showlegend=False,
    )
    if rewards:
        fig.add_trace(go.Scatter(
            x=timesteps, y=rewards,
            mode="lines",
            line=dict(color="#4a9eff", width=1.5, shape="spline"),
            hoverinfo="none",
        ))
    return fig


def _param_rows(algo: str, current_params: dict):
    from dash import html, dcc
    params = current_params or ALGO_DEFAULTS.get(algo, {})
    rows = []
    for k, v in params.items():
        rows.append(
            html.Div(className="config-row", children=[
                html.Span(k, className="config-label"),
                dcc.Input(
                    id={"type": "param-input", "key": k},
                    type="text",
                    value=str(v),
                    className="param-input",
                    debounce=True,
                ),
            ])
        )
    return rows


# ── Library modal — open / close ──────────────────────────────────────

@app.callback(
    Output("env-modal", "style"),
    Output("env-error", "children"),
    Input("cartridge-slot", "n_clicks"),
    Input("modal-cancel", "n_clicks"),
    Input("modal-cancel-bottom", "n_clicks"),
    Input("modal-confirm", "n_clicks"),
    State("env-input", "value"),
    prevent_initial_call=True,
)
def toggle_modal(cart_c, cancel_c, cancel_bottom_c, confirm_c, env_input):
    triggered = ctx.triggered_id
    if triggered == "cartridge-slot":
        return {"display": "flex"}, ""
    if triggered in ("modal-cancel", "modal-cancel-bottom"):
        return {"display": "none"}, ""
    if triggered == "modal-confirm":
        env_id = (env_input or "").strip()
        if not env_id:
            return {"display": "flex"}, "ENTER AN ENV ID"
        try:
            e = gymnasium.make(env_id)
            e.close()
        except Exception as exc:
            return {"display": "flex"}, f"INVALID: {str(exc)[:40]}"
        return {"display": "none"}, ""
    return no_update, no_update


@app.callback(
    Output("env-store", "data"),
    Output("cart-label", "children"),
    Output("env-display", "children"),
    Output("train-btn", "disabled"),
    Input("modal-confirm", "n_clicks"),
    State("env-input", "value"),
    prevent_initial_call=True,
)
def save_env(confirm_clicks, env_input):
    env_id = (env_input or "").strip()
    if not env_id:
        return no_update, no_update, no_update, no_update
    try:
        e = gymnasium.make(env_id)
        e.close()
    except Exception:
        return no_update, no_update, no_update, no_update
    return {"env_id": env_id}, _truncate_cart(env_id), f"▶ {env_id}", False


# ── Library modal — category collapse ─────────────────────────────────

@app.callback(
    Output({"type": "cat-body", "cat": ALL}, "className"),
    Output({"type": "cat-arrow", "cat": ALL}, "children"),
    Output("lib-state", "data"),
    Input({"type": "cat-toggle", "cat": ALL}, "n_clicks"),
    State("lib-state", "data"),
    prevent_initial_call=True,
)
def toggle_category(clicks, lib_state):
    triggered = ctx.triggered_id
    cats = list(LIBRARY.keys())
    collapsed = dict(lib_state.get("collapsed", {}))

    if triggered and isinstance(triggered, dict) and triggered.get("type") == "cat-toggle":
        cat = triggered["cat"]
        collapsed[cat] = not collapsed.get(cat, False)

    body_classes = [
        "lib-cat-body collapsed" if collapsed.get(c, False) else "lib-cat-body"
        for c in cats
    ]
    arrows = ["▶ " if collapsed.get(c, False) else "▼ " for c in cats]
    lib_state = dict(lib_state)
    lib_state["collapsed"] = collapsed
    return body_classes, arrows, lib_state


# ── Library modal — env row selection ─────────────────────────────────

_ALL_ENVS = [env for envs in LIBRARY.values() for env in envs]


@app.callback(
    Output({"type": "env-row", "env": ALL}, "className"),
    Output("env-input", "value"),
    Output("lib-state", "data", allow_duplicate=True),
    Input({"type": "env-row", "env": ALL}, "n_clicks"),
    State("lib-state", "data"),
    prevent_initial_call=True,
)
def select_env_row(clicks, lib_state):
    triggered = ctx.triggered_id
    selected = ""
    if triggered and isinstance(triggered, dict) and triggered.get("type") == "env-row":
        selected = triggered["env"]

    row_classes = [
        "lib-env-row selected" if env == selected else "lib-env-row"
        for env in _ALL_ENVS
    ]
    lib_state = dict(lib_state)
    lib_state["selected"] = selected
    return row_classes, selected, lib_state


# ── Algo card selection ───────────────────────────────────────────────

@app.callback(
    *[Output(f"algo-card-{name}", "className") for name, _ in ALGOS],
    Output("algo-card-CUSTOM", "className"),
    Output("algo-store", "data"),
    Output("param-rows", "children"),
    Output("custom-algo-modal", "style", allow_duplicate=True),
    *[Input(f"algo-card-{name}", "n_clicks") for name, _ in ALGOS],
    Input("algo-card-CUSTOM", "n_clicks"),
    State("algo-store", "data"),
    State("custom-algo-store", "data"),
    prevent_initial_call=True,
)
def select_algo(*args):
    from dash import html, dcc as _dcc
    n_algos = len(ALGOS)
    # args: n_algos built-in clicks + 1 custom click + algo_data + custom_algo_data
    clicks = args[:n_algos + 1]
    algo_data = args[n_algos + 1]
    custom_algo_data = args[n_algos + 2]

    triggered = ctx.triggered_id
    modal_style = no_update

    if triggered == "algo-card-CUSTOM":
        # Open the custom algo modal regardless of loaded state
        modal_style = {"display": "flex"}
        # Don't change selection yet — wait for confirm
        cur_algo = (algo_data or {}).get("algo", "PPO")
        builtin_classes = [
            "algo-card selected" if name == cur_algo else "algo-card"
            for name, _ in ALGOS
        ]
        custom_cls = _custom_algo_card_class(custom_algo_data, cur_algo == "CUSTOM")
        return *builtin_classes, custom_cls, no_update, no_update, modal_style

    if not triggered or not triggered.startswith("algo-card-"):
        cur_algo = (algo_data or {}).get("algo", "PPO")
        builtin_classes = ["algo-card selected" if name == cur_algo else "algo-card"
                           for name, _ in ALGOS]
        custom_cls = _custom_algo_card_class(custom_algo_data, cur_algo == "CUSTOM")
        return *builtin_classes, custom_cls, no_update, no_update, modal_style

    selected = triggered.replace("algo-card-", "")
    builtin_classes = [
        "algo-card selected" if name == selected else "algo-card"
        for name, _ in ALGOS
    ]
    custom_cls = _custom_algo_card_class(custom_algo_data, False)
    new_params = ALGO_DEFAULTS.get(selected, {})
    return (*builtin_classes, custom_cls,
            {"algo": selected, "params": new_params},
            _param_rows(selected, new_params),
            modal_style)


def _custom_algo_card_class(custom_algo_data, selected: bool) -> str:
    loaded = bool((custom_algo_data or {}).get("class_name"))
    parts = ["algo-card-custom"]
    if selected:
        parts.append("selected")
    elif not loaded:
        parts.append("dimmed")
    return " ".join(parts)


@app.callback(
    Output("param-rows", "children", allow_duplicate=True),
    Input("algo-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def init_param_rows(algo_data):
    algo = (algo_data or {}).get("algo", "PPO")
    params = (algo_data or {}).get("params", ALGO_DEFAULTS["PPO"])
    return _param_rows(algo, params)


# ── Architecture buttons ──────────────────────────────────────────────

@app.callback(
    Output("arch-btn-MLP",  "className"),
    Output("arch-btn-CNN",  "className"),
    Output("arch-btn-LSTM", "className"),
    Output("agent-store", "data"),
    Input("arch-btn-MLP",  "n_clicks"),
    Input("arch-btn-CNN",  "n_clicks"),
    Input("arch-btn-LSTM", "n_clicks"),
    State("agent-store", "data"),
    prevent_initial_call=True,
)
def select_arch(mlp_c, cnn_c, lstm_c, agent_data):
    triggered = ctx.triggered_id
    arch = triggered.replace("arch-btn-", "") if triggered else "MLP"
    classes = {a: "arch-btn active" if a == arch else "arch-btn"
               for a in ["MLP", "CNN", "LSTM"]}
    data = dict(agent_data or {})
    data["arch"] = arch
    return classes["MLP"], classes["CNN"], classes["LSTM"], data


# ── Shaping toggles ───────────────────────────────────────────────────

_TOGGLE_KEYS = ["obs_norm", "frame_stack", "act_clip", "rew_clip", "rew_scale"]


@app.callback(
    *[Output(f"toggle-{k}", "className") for k in _TOGGLE_KEYS],
    Output("agent-store", "data", allow_duplicate=True),
    Output("frame-stack-extra", "children"),
    Output("rew-scale-extra", "children"),
    *[Input(f"toggle-{k}", "n_clicks") for k in _TOGGLE_KEYS],
    State("agent-store", "data"),
    prevent_initial_call=True,
)
def toggle_shaping(*args):
    from dash import html, dcc
    n_toggles = len(_TOGGLE_KEYS)
    agent_data = dict(args[n_toggles] or {})

    triggered = ctx.triggered_id
    if triggered and triggered.startswith("toggle-"):
        key = triggered.replace("toggle-", "")
        agent_data[key] = not agent_data.get(key, False)

    classes = [
        "toggle on" if agent_data.get(k, False) else "toggle"
        for k in _TOGGLE_KEYS
    ] + [html.Div(className="toggle-dot")]

    # frame_stack extra
    fs_extra = []
    if agent_data.get("frame_stack"):
        fs_extra = [html.Div(className="config-row", children=[
            html.Span("n_stack", className="config-label"),
            dcc.Input(id="frame-stack-n", type="number", value=agent_data.get("frame_stack_n", 4),
                      min=2, max=16, className="param-input", debounce=True),
        ])]

    # rew_scale extra
    rs_extra = []
    if agent_data.get("rew_scale"):
        rs_extra = [html.Div(className="config-row", children=[
            html.Span("scale", className="config-label"),
            dcc.Input(id="rew-scale-v", type="number", value=agent_data.get("rew_scale_v", 1.0),
                      step=0.1, className="param-input", debounce=True),
        ])]

    toggle_classes = [
        "toggle on" if agent_data.get(k, False) else "toggle"
        for k in _TOGGLE_KEYS
    ]
    return *toggle_classes, agent_data, fs_extra, rs_extra


# ── W&B toggle ───────────────────────────────────────────────────────

@app.callback(
    Output("toggle-wandb", "className"),
    Output("agent-store", "data", allow_duplicate=True),
    Output("wandb-extras", "children"),
    Input("toggle-wandb", "n_clicks"),
    State("agent-store", "data"),
    prevent_initial_call=True,
)
def toggle_wandb(n_clicks, agent_data):
    from dash import html, dcc as _dcc
    data = dict(agent_data or {})
    data["wandb_enabled"] = not data.get("wandb_enabled", False)

    cls = "toggle on" if data["wandb_enabled"] else "toggle"

    extras = []
    if data["wandb_enabled"]:
        extras = [
            html.Div(className="config-row", children=[
                html.Span("project", className="config-label"),
                _dcc.Input(id="wandb-project", type="text",
                           value=data.get("wandb_project", "rl-trainer-ds"),
                           className="param-input", debounce=True),
            ]),
            html.Div(className="config-row", children=[
                html.Span("run", className="config-label"),
                _dcc.Input(id="wandb-run", type="text",
                           value=data.get("wandb_run", ""),
                           placeholder="auto",
                           className="param-input", debounce=True),
            ]),
        ]

    return cls, data, extras


@app.callback(
    Output("agent-store", "data", allow_duplicate=True),
    Input("wandb-project", "value"),
    Input("wandb-run", "value"),
    State("agent-store", "data"),
    prevent_initial_call=True,
)
def update_wandb_fields(project, run, agent_data):
    data = dict(agent_data or {})
    if project is not None:
        data["wandb_project"] = project
    if run is not None:
        data["wandb_run"] = run
    return data


# ── Train / Stop button ───────────────────────────────────────────────

@app.callback(
    Output("train-btn", "children"),
    Output("train-status", "children"),
    Output("train-status", "className"),
    Output("chart-interval", "disabled"),
    Output("power-led", "className"),
    Output("training-active", "data"),
    Output("face-tb-btn", "className"),
    Output("wb-btn", "className"),
    Input("train-btn", "n_clicks"),
    State("env-store", "data"),
    State("algo-store", "data"),
    State("agent-store", "data"),
    State("training-active", "data"),
    prevent_initial_call=True,
)
def toggle_training(n_clicks, env_data, algo_data, agent_data, is_active):
    if is_active:
        runner.stop_training()
        return ("▶ TRAIN", "● READY", "screen-status", True,
                "power-led", False,
                "face-btn btn-tb dimmed", "shoulder-mini dimmed")

    env_id = (env_data or {}).get("env_id")
    if not env_id:
        return (no_update,) * 8

    algo   = (algo_data or {}).get("algo", "PPO")
    params = (algo_data or {}).get("params", {})
    arch   = (agent_data or {}).get("arch", "MLP")
    shaping = {k: agent_data.get(k, False) for k in _TOGGLE_KEYS}
    shaping["frame_stack_n"] = agent_data.get("frame_stack_n", 4)
    shaping["rew_scale_v"]   = agent_data.get("rew_scale_v", 1.0)

    wandb_cfg = {
        "enabled":  agent_data.get("wandb_enabled", False),
        "project":  agent_data.get("wandb_project", "rl-trainer-ds"),
        "run_name": agent_data.get("wandb_run", f"{env_id}_{algo}"),
    }

    runner.start_training(env_id, algo, params, arch, shaping, wandb_cfg)

    wb_class = "shoulder-mini active" if wandb_cfg["enabled"] else "shoulder-mini dimmed"
    return ("■ STOP", "◉ TRAINING", "screen-status training", False,
            "power-led training", True,
            "face-btn btn-tb", wb_class)


# ── TB / WB clientside openers ────────────────────────────────────────

app.clientside_callback(
    """function(n, cls) {
        if (n && cls && !cls.includes('dimmed')) {
            window.open('http://localhost:6006', '_blank');
        }
        return window.dash_clientside.no_update;
    }""",
    Output("face-tb-btn", "title"),
    Input("face-tb-btn", "n_clicks"),
    State("face-tb-btn", "className"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """function(n, cls) {
        if (n && cls && !cls.includes('dimmed')) {
            window.open('https://wandb.ai', '_blank');
        }
        return window.dash_clientside.no_update;
    }""",
    Output("wb-btn", "title"),
    Input("wb-btn",  "n_clicks"),
    State("wb-btn",  "className"),
    prevent_initial_call=True,
)


# ── Live chart update ─────────────────────────────────────────────────

@app.callback(
    Output("reward-chart", "figure"),
    Output("log-pre", "children"),
    Input("chart-interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_chart(n):
    state = runner.get_state()
    if state is None:
        return no_update, no_update
    rewards = list(state.rewards)        # full history, no windowing
    timesteps = list(state.timesteps)    # full history from timestep 0
    log_text = "\n".join(state.log_lines[-8:])
    return _make_chart(rewards, timesteps), log_text


# ── Custom env modal ──────────────────────────────────────────────────

@app.callback(
    Output("custom-env-modal", "style"),
    Output("custom-env-feedback", "children"),
    Output("custom-env-success", "children"),
    Output("custom-env-confirm-btn", "disabled"),
    Output("custom-env-confirm-btn", "style"),
    Input("custom-cart-slot", "n_clicks"),
    Input("custom-env-modal-cancel", "n_clicks"),
    Input("custom-env-modal-cancel-bottom", "n_clicks"),
    Input("custom-env-confirm-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_custom_env_modal(open_c, cancel1, cancel2, confirm_c):
    triggered = ctx.triggered_id
    hidden = {"display": "none"}
    visible = {"display": "flex"}
    btn_disabled = {"opacity": "0.4"}
    btn_enabled = {"opacity": "1"}

    if triggered == "custom-cart-slot":
        return visible, "", "", True, btn_disabled
    if triggered in ("custom-env-modal-cancel", "custom-env-modal-cancel-bottom"):
        return hidden, "", "", True, btn_disabled
    if triggered == "custom-env-confirm-btn":
        return hidden, "", "", True, btn_disabled
    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("custom-env-feedback", "children", allow_duplicate=True),
    Output("custom-env-success", "children", allow_duplicate=True),
    Output("custom-env-confirm-btn", "disabled", allow_duplicate=True),
    Output("custom-env-confirm-btn", "style", allow_duplicate=True),
    Output("custom-env-store", "data", allow_duplicate=True),
    Input("custom-env-load-btn", "n_clicks"),
    State("custom-env-filepath", "value"),
    prevent_initial_call=True,
)
def load_custom_env_btn(n_clicks, filepath):
    from .training.custom_loader import load_custom_env
    fp = (filepath or "").strip()
    if not fp:
        return "ENTER A FILE PATH", "", True, {"opacity": "0.4"}, no_update
    try:
        class_name, env_id = load_custom_env(fp)
    except Exception as exc:
        return str(exc)[:80].upper(), "", True, {"opacity": "0.4"}, no_update
    success_msg = f"✓ {class_name}  →  {env_id}"
    return "", success_msg, False, {"opacity": "1"}, {"class_name": class_name, "env_id": env_id, "filepath": fp}


@app.callback(
    Output("custom-cart-label", "children"),
    Output("env-store", "data", allow_duplicate=True),
    Output("train-btn", "disabled", allow_duplicate=True),
    Output("env-display", "children", allow_duplicate=True),
    Input("custom-env-confirm-btn", "n_clicks"),
    State("custom-env-store", "data"),
    prevent_initial_call=True,
)
def confirm_custom_env(n_clicks, custom_env_data):
    if not custom_env_data or not custom_env_data.get("env_id"):
        return no_update, no_update, no_update, no_update
    env_id = custom_env_data["env_id"]
    class_name = custom_env_data["class_name"]
    label = _truncate_cart(class_name, 12)
    return label, {"env_id": env_id}, False, f"▶ {env_id}"


# ── Custom algo modal ─────────────────────────────────────────────────

@app.callback(
    Output("custom-algo-modal", "style"),
    Output("custom-algo-feedback", "children"),
    Output("custom-algo-success", "children"),
    Output("custom-algo-confirm-btn", "disabled"),
    Output("custom-algo-confirm-btn", "style"),
    Input("custom-algo-modal-cancel", "n_clicks"),
    Input("custom-algo-modal-cancel-bottom", "n_clicks"),
    Input("custom-algo-confirm-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_custom_algo_modal(cancel1, cancel2, confirm_c):
    triggered = ctx.triggered_id
    hidden = {"display": "none"}
    btn_disabled = {"opacity": "0.4"}

    if triggered in ("custom-algo-modal-cancel", "custom-algo-modal-cancel-bottom",
                     "custom-algo-confirm-btn"):
        return hidden, "", "", True, btn_disabled
    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("custom-algo-feedback", "children", allow_duplicate=True),
    Output("custom-algo-success", "children", allow_duplicate=True),
    Output("custom-algo-confirm-btn", "disabled", allow_duplicate=True),
    Output("custom-algo-confirm-btn", "style", allow_duplicate=True),
    Output("custom-algo-store", "data", allow_duplicate=True),
    Input("custom-algo-load-btn", "n_clicks"),
    State("custom-algo-filepath", "value"),
    prevent_initial_call=True,
)
def load_custom_algo_btn(n_clicks, filepath):
    from .training.custom_loader import load_custom_algo
    fp = (filepath or "").strip()
    if not fp:
        return "ENTER A FILE PATH", "", True, {"opacity": "0.4"}, no_update
    try:
        class_name, _ = load_custom_algo(fp)
    except Exception as exc:
        return str(exc)[:80].upper(), "", True, {"opacity": "0.4"}, no_update
    success_msg = f"✓ {class_name} loaded"
    return "", success_msg, False, {"opacity": "1"}, {"class_name": class_name, "filepath": fp}


@app.callback(
    Output("custom-algo-card-name", "children"),
    Output("custom-algo-card-type", "children"),
    Output("algo-card-CUSTOM", "className", allow_duplicate=True),
    Output("algo-store", "data", allow_duplicate=True),
    Output("param-rows", "children", allow_duplicate=True),
    Input("custom-algo-confirm-btn", "n_clicks"),
    State("custom-algo-store", "data"),
    State("algo-store", "data"),
    prevent_initial_call=True,
)
def confirm_custom_algo(n_clicks, custom_algo_data, algo_data):
    from dash import html, dcc as _dcc
    if not custom_algo_data or not custom_algo_data.get("class_name"):
        return no_update, no_update, no_update, no_update, no_update

    class_name = custom_algo_data["class_name"]
    truncated = class_name[:6].upper() if len(class_name) > 6 else class_name.upper()

    # JSON param input for custom algos
    param_rows = [
        html.Div(className="config-row", children=[
            html.Span("params", className="config-label"),
            _dcc.Input(
                id={"type": "param-input", "key": "json_params"},
                type="text",
                value='{"learning_rate": 3e-4, "gamma": 0.99}',
                className="param-input",
                debounce=True,
                style={"width": "100%", "textAlign": "left"},
            ),
        ]),
    ]

    new_algo_data = {"algo": "CUSTOM", "params": {"json_params": '{"learning_rate": 3e-4, "gamma": 0.99}'}}
    return truncated, class_name[:10], "algo-card-custom selected", new_algo_data, param_rows
