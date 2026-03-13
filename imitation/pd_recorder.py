# imitation/pd_recorder.py
import json
import os
import time
from pathlib import Path


class PDRecorder:
    def __init__(self, out_dir: str, flush_every: int = 100):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"session_{ts}.jsonl"

        self.fp = open(self.path, "a", encoding="utf-8")
        self.flush_every = flush_every
        self.count = 0

        self.meta = {
            "type": "pd_record_session",
            "version": 1,
            "created_at": ts,
        }
        self.fp.write(json.dumps({"meta": self.meta}, ensure_ascii=False) + "\n")

    def write(self, row: dict):
        self.fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % self.flush_every == 0:
            self.fp.flush()

    def close(self):
        if not self.fp.closed:
            self.fp.flush()
            self.fp.close()