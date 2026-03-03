from __future__ import annotations
import sys
import logging

logger = logging.getLogger(__name__)


class ProgressBar:

    def __init__(self, text: str, count: int, transient: bool = True):
        self.text = text
        self.count = count
        self.transient = transient
        self._last_pct = -1

    def __enter__(self):
        if self.text:
            logger.info("%s (0/%d)", self.text, self.count)
            sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.text and not self.transient:
            logger.info("%s done (%d/%d)", self.text, self.count, self.count)

    def update(self, value: int):
        if self.text and self.count > 0:
            pct = int(value * 100 / self.count)
            if pct >= self._last_pct + 10 or value == self.count:
                self._last_pct = pct
                logger.info("%s %d%%  (%d/%d)", self.text, pct, value, self.count)
                sys.stdout.flush()

    def new_task(self, text: str, count: int):
        self.text = text
        self.count = count
        self._last_pct = -1
        if self.text:
            logger.info("%s (0/%d)", self.text, self.count)
