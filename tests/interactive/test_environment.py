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
