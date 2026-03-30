import atexit
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback

from .wrappers import make_env
from .custom_loader import CUSTOM_ALGOS

# Module-level training state (single-process Dash dev mode)
_state = None
_tb_proc: Optional[subprocess.Popen] = None


@dataclass
class TrainingState:
    running: bool = False
    rewards: List[float] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    log_lines: List[str] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    tb_log_dir: str = ""
    custom_env_path: Optional[str] = None
    custom_algo_path: Optional[str] = None


class RewardCallback(BaseCallback):
    def __init__(self, state: TrainingState):
        super().__init__(verbose=0)
        self.state = state

    def _on_step(self) -> bool:
        if self.state.stop_event.is_set():
            self.state.running = False
            return False
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_r = float(info["episode"]["r"])
                self.state.rewards.append(ep_r)
                self.state.timesteps.append(self.num_timesteps)
                self.state.log_lines.append(
                    f"[{self.num_timesteps:>8d}] ep_reward={ep_r:.2f}"
                )
                if len(self.state.log_lines) > 200:
                    self.state.log_lines = self.state.log_lines[-200:]
        return True


_ALGO_MAP = {
    "PPO": sb3.PPO,
    "SAC": sb3.SAC,
    "A2C": sb3.A2C,
    "DQN": sb3.DQN,
}

_POLICY_MAP = {
    "MLP":  "MlpPolicy",
    "CNN":  "CnnPolicy",
    "LSTM": "MlpLstmPolicy",
}

_FLOAT_PARAMS = {"learning_rate", "gamma", "tau"}
_INT_PARAMS   = {"n_steps", "batch_size", "n_epochs", "buffer_size"}


def _cast_params(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if k in _FLOAT_PARAMS:
            out[k] = float(v)
        elif k in _INT_PARAMS:
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                out[k] = v
    return out


def _is_port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) == 0


def _ensure_tensorboard():
    global _tb_proc
    if _is_port_open(6006):
        return  # already running
    try:
        _tb_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "./tb_logs", "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # tensorboard not installed; silently skip


@atexit.register
def _cleanup():
    global _tb_proc
    if _tb_proc is not None:
        _tb_proc.terminate()
        _tb_proc = None


def start_training(env_id: str, algo_name: str, algo_params: dict,
                   arch: str, shaping_config: dict,
                   wandb_config: Optional[dict] = None) -> TrainingState:
    global _state

    if _state and _state.running:
        stop_training()

    _state = TrainingState()
    _state.running = True
    _state.tb_log_dir = "./tb_logs"

    _ensure_tensorboard()

    def _run():
        try:
            env = make_env(env_id, shaping_config)
            policy = _POLICY_MAP.get(arch, "MlpPolicy")

            if algo_name == "CUSTOM":
                if not CUSTOM_ALGOS:
                    _state.log_lines.append("[error] no custom algo loaded")
                    _state.running = False
                    return
                class_name = next(iter(CUSTOM_ALGOS))
                AlgoClass = CUSTOM_ALGOS[class_name]
                import json
                raw = algo_params.get("json_params", "{}")
                try:
                    params = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except json.JSONDecodeError as e:
                    _state.log_lines.append(f"[error] bad JSON params: {e}")
                    _state.running = False
                    return
            else:
                AlgoClass = _ALGO_MAP[algo_name]
                params = _cast_params(algo_params)

            if policy == "MlpLstmPolicy":
                try:
                    from sb3_contrib import RecurrentPPO  # noqa: F401
                except ImportError:
                    policy = "MlpPolicy"
                    _state.log_lines.append("[warn] sb3-contrib not found, using MlpPolicy")

            # wandb init
            _wandb_ok = False
            if wandb_config and wandb_config.get("enabled"):
                try:
                    import wandb
                    wandb.init(
                        project=wandb_config.get("project", "rl-trainer-ds"),
                        name=wandb_config.get("run_name", f"{env_id}_{algo_name}"),
                        sync_tensorboard=True,
                    )
                    _wandb_ok = True
                    _state.log_lines.append("[wandb] run started")
                except Exception as exc:
                    _state.log_lines.append(f"[wandb] init failed: {exc}")

            model = AlgoClass(
                policy, env, verbose=0,
                tensorboard_log=_state.tb_log_dir,
                **params,
            )
            callback = RewardCallback(_state)
            _state.log_lines.append(f"▶ {algo_name} on {env_id} ({policy})")
            model.learn(
                total_timesteps=int(1e7),
                callback=callback,
                reset_num_timesteps=True,
                progress_bar=False,
            )
        except Exception as exc:
            _state.log_lines.append(f"[error] {exc}")
        finally:
            _state.running = False
            if _wandb_ok:
                try:
                    import wandb
                    wandb.finish()
                    _state.log_lines.append("[wandb] run finished")
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return _state


def stop_training():
    global _state
    if _state:
        _state.stop_event.set()
        _state.running = False


def get_state() -> Optional[TrainingState]:
    return _state


def tb_is_running() -> bool:
    return _is_port_open(6006)
