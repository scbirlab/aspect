"""CheMeleon weights download: single-flight, atomic, self-repairing. No network."""

import hashlib
import multiprocessing as mp
import os
from pathlib import Path

import pytest

import aspect.transform.chemeleon as ch

GOOD = b"chemeleon-weights" * 1000
BAD = b"truncated"


def _marker_download(markers, delay=0.):
    """Fake `_download` writing GOOD atomically and leaving a marker per call."""
    def _fake_download(url, dest, timeout, payload=GOOD):
        import time
        time.sleep(delay)
        (markers / f"{os.getpid()}-{time.time_ns()}").touch()
        tmp = dest.parent / f"{dest.name}.part.test{os.getpid()}"
        tmp.write_bytes(payload)
        os.replace(tmp, dest)
    return _fake_download


def _hold(lock_path, ready, release):
    from filelock import FileLock
    with FileLock(lock_path):
        ready.set()
        release.wait(10)


def _spawned_worker(cache, markers):
    """Runs in a fresh (spawned) interpreter, so it re-applies the fakes."""
    from pathlib import Path
    import aspect.transform.chemeleon as c
    os.environ["PROVIDENCE_CACHE"] = cache
    c.CHEMELEON_SHA256 = hashlib.sha256(GOOD).hexdigest()
    c._download = _marker_download(Path(markers), delay=0.5)
    return str(c._chemeleon_checkpoint())


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Point the shared cache at tmp_path; downloads write GOOD and leave a marker."""
    monkeypatch.setenv("PROVIDENCE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ch, "CHEMELEON_SHA256", hashlib.sha256(GOOD).hexdigest())
    markers = tmp_path / "downloads"
    markers.mkdir()
    monkeypatch.setattr(ch, "_download", _marker_download(markers))
    return tmp_path / "cache" / "chemeleon" / "chemeleon_mp.pt", markers


def _n(markers):
    return len(list(markers.iterdir()))


def test_downloads_once_then_reuses(fake):
    path, markers = fake
    assert ch._chemeleon_checkpoint() == path
    assert ch._chemeleon_checkpoint() == path
    assert path.read_bytes() == GOOD
    assert _n(markers) == 1


def test_dir_follows_providence_cache(fake, tmp_path):
    path, _ = fake
    assert ch._chemeleon_dir() == tmp_path / "cache" / "chemeleon"
    assert ch._chemeleon_checkpoint() == path


def test_dir_default_is_providence_default(monkeypatch):
    from providence.package_data import DEFAULT_CACHE
    monkeypatch.delenv("PROVIDENCE_CACHE", raising=False)
    assert ch._chemeleon_dir() == Path(DEFAULT_CACHE).resolve() / "chemeleon"


def test_construction_downloads_nothing(fake):
    pytest.importorskip("chemprop")
    _, markers = fake
    ch.CheMeleon(device="cpu")
    assert _n(markers) == 0


def test_corrupt_cache_is_replaced(fake):
    path, markers = fake
    path.parent.mkdir(parents=True)
    path.write_bytes(BAD)
    assert ch._chemeleon_checkpoint() == path
    assert path.read_bytes() == GOOD
    assert _n(markers) == 1


def test_persistent_mismatch_raises(fake, monkeypatch):
    path, markers = fake
    orig = ch._download
    monkeypatch.setattr(ch, "_download", lambda u, d, timeout: orig(u, d, timeout, payload=BAD))
    with pytest.raises(OSError, match="after re-download"):
        ch._chemeleon_checkpoint()
    assert _n(markers) == 2


def test_lock_timeout_raises(fake, monkeypatch):
    path, markers = fake
    path.parent.mkdir(parents=True)
    monkeypatch.setenv("ASPECT_CHEMELEON_LOCK_TIMEOUT", "0.2")
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold, args=(str(path) + ".lock", ready, release))
    holder.start()
    try:
        assert ready.wait(10)
        with pytest.raises(OSError, match="timed out"):
            ch._chemeleon_checkpoint()
    finally:
        release.set()
        holder.join(10)
    assert _n(markers) == 0


def test_bad_env_timeout(fake, monkeypatch):
    monkeypatch.setenv("ASPECT_CHEMELEON_LOCK_TIMEOUT", "soon")
    with pytest.raises(ValueError, match="ASPECT_CHEMELEON_LOCK_TIMEOUT"):
        ch._chemeleon_checkpoint()


def test_single_flight_across_processes(fake):
    path, markers = fake
    with mp.get_context("spawn").Pool(8) as pool:
        out = pool.starmap(_spawned_worker, [(str(path.parent.parent), str(markers))] * 8)
    assert set(out) == {str(path)}
    assert _n(markers) == 1
    assert path.read_bytes() == GOOD


def test_real_download_cleans_up_on_failure(tmp_path):
    """`_download` leaves neither a partial nor a final file when the transfer fails."""
    dest = tmp_path / "chemeleon_mp.pt"
    with pytest.raises(Exception):
        ch._download("http://127.0.0.1:9/nothing", dest, timeout=2)
    assert list(tmp_path.iterdir()) == []


def test_real_download_atomic(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(GOOD)
    dest = tmp_path / "out" / "chemeleon_mp.pt"
    dest.parent.mkdir()
    ch._download(src.as_uri(), dest, timeout=5)
    assert dest.read_bytes() == GOOD
    assert [p.name for p in dest.parent.iterdir()] == ["chemeleon_mp.pt"]
