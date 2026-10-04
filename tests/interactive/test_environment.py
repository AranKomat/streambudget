import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from streambudget.interactive.contracts import ActionSpec
from streambudget.interactive.environment import PyBoyEnvironment
from streambudget.types import ContractError

class Device:

    def __init__(self, *a, **kw):
        self.screen = SimpleNamespace(image=Image.new('RGB', (160, 144)))
        self.events = []
        self.frame = 0
        self.fail = False

    def set_emulation_speed(self, n):
        self.events.append(('speed', n))

    def tick(self, n, render):
        if self.fail:
            raise RuntimeError('tick failed')
        self.frame += n
        return True

    def button_press(self, b):
        self.events.append(('press', b))

    def button_release(self, b):
        self.events.append(('release', b))

    def save_state(self, f):
        f.write(b'fixture state')

    def load_state(self, f):
        self.events.append(('loaded', f.read()))

    def stop(self, save):
        self.events.append(('stop', save))

def env(tmp_path):
    rom = tmp_path / 'test.gb'
    rom.write_bytes(b'not a playable ROM')
    return PyBoyEnvironment(rom, boot_frames=0, factory=Device)

def test_press_release_and_frame_accounting(tmp_path):
    e = env(tmp_path)
    r = e.execute(ActionSpec(id='RIGHT', button='right', press_frames=4, release_frames=3))
    assert r.end_frame - r.start_frame == 7
    assert e._device.events[-2:] == [('press', 'right'), ('release', 'right')]
    e.close()
    assert e._device.events[-1] == ('stop', False)

def test_release_on_failed_tick(tmp_path):
    e = env(tmp_path)
    e._device.fail = True
    with pytest.raises(RuntimeError):
        e.execute(ActionSpec(id='A', button='a'))
    assert e._device.events[-1] == ('release', 'a')
    e.close()

def test_wait_has_no_button(tmp_path):
    e = env(tmp_path)
    e.execute(ActionSpec(id='WAIT', button='wait'))
    assert not [x for x in e._device.events if x[0] in ('press', 'release')]
    e.close()

def test_firered_gba_explicitly_rejected(tmp_path):
    p = tmp_path / 'FireRed.gba'
    p.write_bytes(b'x')
    with pytest.raises(ContractError):
        PyBoyEnvironment(p, factory=Device)

def test_no_privileged_device_properties_used():
    import streambudget.interactive.environment as m
    tree = ast.parse(Path(m.__file__).read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {'memory', 'game_area', 'game_wrapper', 'tilemap_background', 'get_sprite', 'memory_scanner'}

def test_capture_is_copy(tmp_path):
    e = env(tmp_path)
    s = e.capture()
    s.image.putpixel((0, 0), (255, 0, 0))
    assert e.capture().image.getpixel((0, 0)) != (255, 0, 0)
    e.close()

def test_native_checkpoint_wrapper(tmp_path):
    e = env(tmp_path)
    p = tmp_path / 'state'
    e.checkpoint(p)
    assert p.read_bytes() == b'fixture state' and (not (tmp_path / 'state.tmp').exists())
    e.close()


def test_loaded_operator_state_is_labelled(tmp_path):
    import hashlib
    rom = tmp_path / 'test.gb'
    rom.write_bytes(b'not a playable ROM')
    state = tmp_path / 'operator.state'
    state.write_bytes(b'operator-provided test state')
    e = PyBoyEnvironment(rom, load_state=state, factory=Device)
    assert e.initialization == 'operator_state'
    assert e.initial_state_sha256 == hashlib.sha256(state.read_bytes()).hexdigest()
    e.close()


def test_release_even_if_press_raises(tmp_path):
    e = env(tmp_path)

    def fail(button):
        raise RuntimeError('press uncertain')

    e._device.button_press = fail
    with pytest.raises(RuntimeError):
        e.execute(ActionSpec(id='A', button='a'))
    assert e._device.events[-1] == ('release', 'a')
    assert e.frame_number == 0
    e.close()


def test_release_on_keyboard_interrupt(tmp_path):
    e = env(tmp_path)

    def interrupt(n, render):
        raise KeyboardInterrupt

    e._device.tick = interrupt
    with pytest.raises(KeyboardInterrupt):
        e.execute(ActionSpec(id='A', button='a'))
    assert e._device.events[-1] == ('release', 'a')
    e.close()


def test_failed_boot_closes_native_device(tmp_path):
    devices = []

    class FailedBoot(Device):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail = True
            devices.append(self)

    rom = tmp_path / 'test.gb'
    rom.write_bytes(b'not a playable ROM')
    with pytest.raises(RuntimeError):
        PyBoyEnvironment(rom, factory=FailedBoot)
    assert devices[0].events[-1] == ('stop', False)


def test_failed_checkpoint_keeps_previous_bytes(tmp_path):
    e = env(tmp_path)
    state = tmp_path / 'previous.state'
    state.write_bytes(b'important previous checkpoint')

    def partial(f):
        f.write(b'partial state')
        raise RuntimeError('save failed')

    e._device.save_state = partial
    with pytest.raises(RuntimeError):
        e.checkpoint(state)
    assert state.read_bytes() == b'important previous checkpoint'
    e.close()


def test_native_pyboy_bundled_demo_roundtrip(tmp_path):
    """Optional real emulator smoke test, not Pokemon or a learned-model result."""
    pyboy = pytest.importorskip('pyboy')
    rom = Path(pyboy.__file__).parent / 'default_rom.gb'
    if not rom.is_file():
        pytest.skip('Installed PyBoy has no bundled demo ROM')
    first = PyBoyEnvironment(rom)
    try:
        before = first.capture()
        assert before.image.size == (160, 144) and before.frame_number == 120
        assert before.image.getextrema() != ((255, 255), (255, 255), (255, 255))
        state = tmp_path / 'native.state'
        first.checkpoint(state)
        receipt = first.execute(ActionSpec(id='A', button='a'))
        assert receipt.end_frame - receipt.start_frame == 8
        expected = first.capture().image.tobytes()
    finally:
        first.close()
    second = PyBoyEnvironment(rom, load_state=state)
    try:
        assert second.capture().image.tobytes() == before.image.tobytes()
        receipt = second.execute(ActionSpec(id='A', button='a'))
        assert receipt.start_frame == 0 and receipt.end_frame == 8
        assert second.capture().image.tobytes() == expected
    finally:
        second.close()
