"""Shared pytest settings for the Tester tests.

An exception raised while Python cleans up an object (for example in __del__, or a fixture object without a
close() method) is "unraisable": by default pytest only reports it as PytestUnraisableExceptionWarning and the test
still passes. Here such exceptions, and exceptions in background threads, fail the test that caused them.
"""
import gc

import pytest


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnraisableExceptionWarning")
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnhandledThreadExceptionWarning")


@pytest.fixture(autouse=True)
def collect_tk_garbage_on_the_main_thread():
    """A destroyed Tester window leaves its Tk variables in reference cycles. Whoever triggers the next garbage
    collection frees them, and when that is a worker thread of a later test, Tkinter raises "main thread is not in
    main loop" (seen in CI). Autouse fixtures are set up first and torn down last, so this collection runs after every
    other fixture of the test (also monkeypatch, which may still reference the app) has let go, on the main thread."""
    yield
    gc.collect()
