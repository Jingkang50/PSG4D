# Copyright (c) Open-MMLab. All rights reserved.
import re
import time

def convert_seconds(seconds: int) -> str:
    """Author: liang.zhihao
    convert seconds into "{hours}:{minutes}:{seconds}"
    mainly use tp calculate remain time

    Args:
        second (int): count of seconds

    Returns:
        str: "{hours}:{minutes}:{seconds}"
    """
    return time.strftime("%d days %H:%M:%S", time.gmtime(seconds))

def timestamp() -> str:
    r"""Author: liang.zhihao
    Get time stamp

    Returns:
        str: str of time stamp
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

class TimerError(Exception):
    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    r"""A flexible Timer class.

    Example:

    >>> import time
    >>> import gorilla
    >>> with gorilla.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with gorilla.Timer(print_tmpl="it takes {:.1f} seconds"):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = gorilla.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, print_tmpl=None, start=True):
        self._is_running = False
        if (print_tmpl is not None) and not re.findall(r"({:.*\df})", print_tmpl):
            print_tmpl += " {:.3f}"
            # raise ValueError("`print_tmpl` must has the `{:.nf}` to show time.")
        self.print_tmpl = print_tmpl if print_tmpl else "{:.3f}"
        if start:
            self.start()

    @property
    def is_running(self):
        r"""bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last()))
        self._is_running = False

    def start(self):
        r"""Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()


    def since_start(self):
        r"""Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        self._t_last = time.time()
        return self._t_last - self._t_start

    def since_last(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last ` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        dur = time.time() - self._t_last
        self._t_last = time.time()
        return dur

    def reset(self):
        r"""
        Reset a new _t_start.
        """
        self._is_running = False
        self.start()


_g_timers = {}  # global timers


def check_time(timer_id):
    """Add check points in a single line.

    This method is suitable for running a task on a list of items. A timer will
    be registered when the method is called for the first time.

    :Example:

    >>> import time
    >>> import gorilla
    >>> for i in range(1, 6):
    >>>     # simulate a code block
    >>>     time.sleep(i)
    >>>     gorilla.check_time("task1")
    2.000
    3.000
    4.000
    5.000

    Args:
        timer_id (str): Timer identifier.
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last()

        