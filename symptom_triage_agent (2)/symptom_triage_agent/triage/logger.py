"""Registro de ejecución en JSON Lines: una línea por evento, con marca de tiempo."""
import json
from datetime import datetime, timezone


class RunLogger:
    def __init__(self, path=None):
        self._fh = open(path, "w", encoding="utf-8") if path else None

    def log(self, episode, phase, **data):
        """phase: perception | decision | action | feedback | episode_end"""
        if self._fh is None:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "episode": episode,
               "phase": phase, **data}
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
