"""Shared pytest settings for the Tester tests.

An exception raised while Python cleans up an object (for example in __del__, or a fixture object without a
close() method) is "unraisable": by default pytest only reports it as PytestUnraisableExceptionWarning and the test
still passes. Here such exceptions, and exceptions in background threads, fail the test that caused them.
"""


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnraisableExceptionWarning")
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnhandledThreadExceptionWarning")
