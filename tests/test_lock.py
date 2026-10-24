import threading

import pytest

from servebench.lock import LockTimeout, MeasurementLock


def test_acquire_creates_file_and_release_deletes_it(tmp_path):
    path = tmp_path / "m.lock"
    lock = MeasurementLock(path, "P4")
    lock.acquire()
    assert path.exists()
    assert path.read_text().startswith("P4 ")
    lock.release()
    assert not path.exists()


def test_second_owner_waits_until_the_first_releases(tmp_path):
    path = tmp_path / "m.lock"
    first = MeasurementLock(path, "P1")
    first.acquire()
    second = MeasurementLock(path, "P4", poll_s=0.01, timeout_s=0.3)
    with pytest.raises(LockTimeout):
        second.acquire()

    releaser = threading.Timer(0.1, first.release)
    releaser.start()
    third = MeasurementLock(path, "P4", poll_s=0.01, timeout_s=5)
    third.acquire()
    releaser.join()
    assert third.waited_s >= 0.05
    assert path.read_text().startswith("P4 ")
    third.release()


def test_release_does_not_delete_another_owners_lock(tmp_path):
    path = tmp_path / "m.lock"
    path.write_text("P2 2026-10-24T00:00:00Z\n")
    lock = MeasurementLock(path, "P4")
    lock.held = True  # pretend; the file belongs to P2
    lock.release()
    assert path.exists()


def test_context_manager_releases_on_error(tmp_path):
    path = tmp_path / "m.lock"
    with pytest.raises(RuntimeError):
        with MeasurementLock(path, "P4"):
            assert path.exists()
            raise RuntimeError("boom")
    assert not path.exists()


def test_exclusive_creation_under_contention(tmp_path):
    path = tmp_path / "m.lock"
    winners = []
    barrier = threading.Barrier(8)

    def grab(n):
        lock = MeasurementLock(path, f"T{n}", poll_s=0.001, timeout_s=0.0)
        barrier.wait()
        try:
            lock.acquire()
            winners.append(n)
        except LockTimeout:
            pass

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1
