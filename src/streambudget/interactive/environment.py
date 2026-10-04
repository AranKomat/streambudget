"""Controller adapter: rendered pixels and bounded buttons, never semantic game APIs."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from ..types import ContractError
from .contracts import ActionSpec


@dataclass(frozen=True)
class Screen:
    image: Image.Image
    frame_number: int


@dataclass(frozen=True)
class Receipt:
    action_id: str
    start_frame: int
    end_frame: int
    press_frames: int
    release_frames: int
    status: str = "executed_not_verified"


class Environment(Protocol):
    def capture(self) -> Screen: ...
    def execute(self, action: ActionSpec) -> Receipt: ...
    def checkpoint(self, path: Path) -> None: ...
    def close(self) -> None: ...


class PyBoyEnvironment:
    """Game Boy / Game Boy Color only. FireRed/GBA is deliberately rejected.

    The simulator advances only during execute(). It is paused during inference.
    Do not advertise these results as real-time action-game performance.
    """
    def __init__(self, rom: Path, *, load_state: Path | None = None, boot_frames=120,
                 window="null", factory=None):
        rom = rom.expanduser().resolve()
        if rom.suffix.lower() not in (".gb", ".gbc") or not rom.is_file():
            raise ContractError("Provide your own .gb/.gbc ROM. PyBoy does not emulate FireRed (.gba)")
        if factory is None:
            try:
                from pyboy import PyBoy
            except ImportError as exc:
                raise ContractError("Install the optional gameboy extra: pip install -e '.[gameboy]'") from exc
            factory = PyBoy
        self._device = factory(str(rom), window=window)
        self.frame_number = 0
        self.rom_sha256 = hashlib.sha256(rom.read_bytes()).hexdigest()
        self.initialization = "operator_state" if load_state else "rom_boot"
        self.initial_state_sha256 = hashlib.sha256(load_state.read_bytes()).hexdigest() if load_state else None
        self.closed = False
        self._device.set_emulation_speed(0)
        if load_state:
            with load_state.expanduser().open("rb") as f:
                self._device.load_state(f)
        else:
            self._advance(boot_frames)

    def _advance(self, n):
        for _ in range(n):
            if not self._device.tick(1, True):
                raise ContractError("Emulator stopped")
            self.frame_number += 1

    def capture(self):
        if self.closed:
            raise ContractError("Emulator closed")
        return Screen(self._device.screen.image.convert("RGB").copy(), self.frame_number)

    def execute(self, action):
        if self.closed:
            raise ContractError("Emulator closed")
        start = self.frame_number
        pressed = action.button != "wait"
        if pressed:
            self._device.button_press(action.button)
        try:
            self._advance(action.press_frames)
        finally:
            # Release even if model/controller execution is interrupted. No retry.
            if pressed:
                self._device.button_release(action.button)
        self._advance(action.release_frames)
        return Receipt(action.id, start, self.frame_number, action.press_frames, action.release_frames)

    def checkpoint(self, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            self._device.save_state(f)
        tmp.replace(path)

    def close(self):
        if not self.closed:
            self.closed = True
            self._device.stop(save=False)
