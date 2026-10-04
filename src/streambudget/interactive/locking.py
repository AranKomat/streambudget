"""Single-writer process lock. OS releases it after a crash; ledgers still gate recovery."""
from pathlib import Path
from ..types import ContractError


class RunLock:
    def __init__(self, path: Path):
        try:
            import fcntl
        except ImportError as exc:
            raise ContractError("Interactive runner currently supports Linux/macOS process locking") from exc
        self._fcntl = fcntl
        self.file = path.open("a+")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise ContractError("Another process owns this run") from None

    def close(self):
        if not self.file.closed:
            self._fcntl.flock(self.file.fileno(), self._fcntl.LOCK_UN)
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
