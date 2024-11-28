from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from typing import Callable, ContextManager, Concatenate, Generator
from functools import wraps
import logging

class ProgressTickCallback[**P](ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def tick(self, *args : P.args, **kwargs : P.kwargs):
        raise NotImplementedError()

class ProgressTracker[**P](ContextManager[ProgressTickCallback[P]]):
    pass

def decorator[**D, T, R](
        tracker : ProgressTracker[T],  func : Callable[D, Generator[T, None, R]]) -> Callable[D, R]:
    @wraps(func)
    def wrapped(*args : D.args, **kwargs : D.kwargs):
        with tracker as tick_callback:
            gen = iter(func(*args, **kwargs))
            try:
                while True:
                    yielded = next(gen)
                    tick_callback.tick(yielded)
            except StopIteration as s:
                return s.value
    return wrapped

class IntervalProgressTracker[**P](ProgressTracker[P]):
    def __init__(
            self,
            interval : float | timedelta,
            tick_callback : Callable[Concatenate[timedelta, P], None],
            start_callback : Callable[[], None] | None = None,
            end_callback :  Callable[[timedelta], None] | None = None,
            do_first_tick : bool = False
        ):
        if isinstance(interval, (int, float)):
            interval = timedelta(seconds=interval)
        self.interval : timedelta = interval
        self.tick_callback = tick_callback
        self.start_callback : Callable[[], None] = start_callback or (lambda : None)
        self.end_callback : Callable[[timedelta], None] = end_callback or (lambda _ : None)
        self.do_first_tick = do_first_tick
        self.callback_object = None
    
    def __enter__(self) -> ProgressTickCallback[P]:
        if self.callback_object is not None:
            raise ValueError("Context manager already in use")
        class _CMObject(ProgressTickCallback[P]):
            def __init__(self, cm : IntervalProgressTracker):
                self.cm = cm
                self.enter_time = datetime.now()
                if cm.do_first_tick:
                    self.last_tick = datetime.min
                else:
                    
                    self.last_tick = self.enter_time
            def tick(self, *args : P.args, **kwargs: P.kwargs):
                now = datetime.now()
                if now - self.last_tick >= self.cm.interval:
                    self.cm.tick_callback(now - self.enter_time, *args, **kwargs)
                    self.last_tick = now
        self.callback_object = _CMObject(self)
        return self.callback_object

    def __exit__(self, _t, _v, _tb) -> bool | None:
        if self.callback_object is None:
            raise ValueError("Context manager not in use")
        else:
            now = datetime.now()
            delta : timedelta = now - self.callback_object.enter_time
            self.end_callback(delta)
            self.callback_object = None


class CountingTracker(ProgressTracker[int]):
    def __init__(self, counter_name : str, inner_tracker : ProgressTracker[str], start_value : int = 0):
        self.counter_name = counter_name
        self.inner_tracker = inner_tracker
        self.start_value = start_value

    def __enter__(self) -> ProgressTickCallback[int]:
        inner_tick_callback = self.inner_tracker.__enter__()
        class _TickCallback(ProgressTickCallback[int]):
            def __init__(self, counter_name : str, start_value : int):
                self.counter_name = counter_name
                self.counter = start_value
        
            def tick(self, amount : int):
                self.counter += amount
                inner_tick_callback.tick(f"{self.counter_name} = {self.counter}")
        return _TickCallback(self.counter_name, self.start_value)


    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        return self.inner_tracker.__exit__(exc_type, exc_value, traceback)


class IntervalLogger(IntervalProgressTracker[str]):
    def __init__(
        self,
        logger : logging.Logger, 
        op_name : str, 
        interval : float | timedelta = timedelta(minutes=2),
        level : int =  logging.INFO,
        log_start : bool | Callable[[], None] = True,
        log_end : bool | Callable[[timedelta], None] = True
    ):
        if not isinstance(interval, timedelta):
            interval = timedelta(seconds=interval)
        tick_callback = lambda delta, msg : logger.log(level, f'\t{op_name} running longer than {interval} ({delta}). {msg}')
        
        def default_or_none(value, default):
            return {True: default, False : None}.get(value, value)
        start_callback = lambda : logger.log(level, f"{op_name} starting")
        start_callback = default_or_none(log_start, start_callback)
        end_callback = lambda delta : logger.log(level, f'{op_name} complete. ({delta}).')
        end_callback = default_or_none(log_end, end_callback)
        super().__init__(interval, tick_callback, start_callback, end_callback)

    def counting(self, counter_name : str, start_value : int = 0) -> CountingTracker:
        return CountingTracker(counter_name, self, start_value)

