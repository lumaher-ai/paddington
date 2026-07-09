import shutil
from datetime import datetime
from pathlib import Path


class DebugRecorder:
    """Saves per-step debug screenshots to disk for a single agent run.

    One recorder == one run. It owns *where + naming + pruning*; the
    BrowserSession owns *when + bytes* and calls write() after each browser
    action. This is a developer debugging trail only — the files never reach
    the LLM, the conversation, or the token budget.

    Layout: ``{root}/{safe_thread_id}/{run_timestamp}/{step}_{tool}_{hhmmss}.png``
    """

    def __init__(self, thread_id: str, root: Path, max_runs: int) -> None:
        # The scoped thread id is "{user_id}:{client_thread_id}" — sanitize the
        # ':' (and any '/') so it is a single safe path segment.
        safe = thread_id.replace("/", "_").replace(":", "_")
        thread_dir = root / safe
        thread_dir.mkdir(parents=True, exist_ok=True)

        self._prune(thread_dir, max_runs)

        # Timestamped folder name sorts chronologically (lexical == time order).
        self.run_dir = thread_dir / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        # A PII-handling phase can flip this off to suppress on-disk screenshots for the
        # rest of the run without discarding the recorder (see BrowserSession._capture_debug).
        self.enabled = True

    def write(self, tool_name: str, png: bytes) -> None:
        """Persist one PNG frame for the current step, then advance the step."""
        name = f"{self._step:02d}_{tool_name}_{datetime.now():%H%M%S}.png"
        (self.run_dir / name).write_bytes(png)
        self._step += 1

    @staticmethod
    def _prune(thread_dir: Path, max_runs: int) -> None:
        """Keep at most ``max_runs`` run folders, deleting the oldest first.

        Called before the new run folder is created, so we make room for it by
        keeping at most ``max_runs - 1`` of the existing ones.
        """
        runs = sorted(d for d in thread_dir.iterdir() if d.is_dir())
        for old in runs[: max(0, len(runs) - (max_runs - 1))]:
            shutil.rmtree(old, ignore_errors=True)
