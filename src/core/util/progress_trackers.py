"""Utility classes for keeping track of progress on long tasks
"""
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from typing import Callable, ContextManager, Concatenate, Generator, TextIO, Any
from contextlib import contextmanager
from functools import wraps
import logging
import warnings

class ProgressTracker:
    def __init__(
            self, 
            expected_total : float | Any | None = None,
            callbacks : list[Callable[['ProgressTracker'], None]] | None = None,
            counter_name : str | None = ""
        ):
        self.progress : float = 0
        self.expected_total : float | None = self.to_expected_total(expected_total)
        self.callbacks = callbacks or []
        self.start_time = datetime.now()
        self.counter_name = counter_name
    
    @staticmethod
    def to_expected_total(expected_total : float | Any | None) -> float | None:
        if expected_total is None:
            return None
        elif isinstance(expected_total, (int, float)):
            return float(expected_total)
        else:
            try:
                return len(expected_total)
            except:
                return None

    def tick(self, steps : float = 1, expected_total : float | Any | None = None):
        """Ticks the tracker, adding `steps` to progress counter

        Args:
            steps (float): Amount of steps taken since last tick
            expected_total (float | None, optional): The total expected number of steps. 
                Can be used to defer determining the total or to modify the total later. Defaults to None.
        """
        self.set_progress(self.progress + steps, expected_total)

    
    def set_progress(self, progress : float, expected_total : float | Any | None = None):
        """Ticks the tracker, overwritin progress with the given argument

        Args:
            progress (float): Amount of steps taken since last tick
            expected_total (float | None, optional): The total expected number of steps. 
                Can be used to defer determining the total or to modify the total later. Defaults to None.
        """
        self.progress = progress
        if expected_total is not None:
            self.expected_total = self.to_expected_total(expected_total)
        if self.expected_total is not None and self.expected_total < self.progress:
            self.expected_total = self.progress
        for callback in self.callbacks:
            callback(self)
    
    def elapsed_time(self) -> timedelta:
        return datetime.now() - self.start_time

    def time_per_step(self) -> timedelta:
        return self.elapsed_time() / self.progress
    
    def percent_done(self) -> float | None:
        if self.expected_total is None:
            return None
        return self.progress / self.expected_total
    
    def percent_left(self) -> float | None:
        done = self.percent_done()
        if done is None:
            return None
        return 1.0 - done
    
    def steps_left(self) -> float | None:
        if self.expected_total is None:
            return None
        return self.expected_total - self.progress

    def estimated_time_left(self) -> timedelta | None:
        steps = self.steps_left()
        if steps is None:
            return None
        return self.time_per_step() * steps

class ProgressContextManager:
    def _enter(self, 
               task_name : str, 
               counter_name : str | None = None,
               expected_total : float | Any | None = None,
               **kwargs) -> ProgressTracker:
        return ProgressTracker(counter_name=counter_name, expected_total=expected_total)

    def _exit(self, tracker : ProgressTracker) -> None: 
        pass

    @contextmanager
    def track(self, 
              task_name : str, 
              counter_name : str | None = None,
              expected_total : float | Any | None = None,
              **kwargs) -> Generator[ProgressTracker, None, None]:
        tracker = self._enter(task_name, counter_name, expected_total, **kwargs)
        try:
            yield tracker
        finally:
            self._exit(tracker)

def null_progress_context_manager():
    class NullProgressTracker(ProgressTracker):
        def __init__(self):
            super().__init__()
        def set_progress(self, progress: float, expected_total: float | Any | None = None):
            pass
    class NullProgressContextManager(ProgressContextManager):
        def _enter(self, task_name : str, counter_name : str | None = None, expected_total : float | Any | None = None, **kwargs) -> ProgressTracker:
            return NullProgressTracker()

        def _exit(self, tracker : ProgressTracker) -> None:
            pass
    return NullProgressContextManager()

NULL_PROGRESS_CM = null_progress_context_manager()

class LogProgressTracker(ProgressTracker):
        def __init__(self, task_name : str, cooldown : timedelta | float, log_callback : Callable[[str], None], **kwargs):
            super().__init__(**kwargs)
            self.last_log = self.start_time
            if not isinstance(cooldown, timedelta):
                cooldown = timedelta(seconds=cooldown)
            self.cooldown = cooldown
            self.task_name = task_name
            self.log_callback = log_callback
            self.callbacks.append(self.tick_callback)


        def tick_callback(self,_):
            now = datetime.now()
            if now - self.last_log >= self.cooldown:
                self.last_log = now
                self.log_callback(self.produce_message())

        def produce_message(self):
            message = (f'\t{self.task_name} taking longer than {self.cooldown} ({self.elapsed_time()}).\n' 
                f'\t\tProgress: {self.progress}')
            if self.counter_name is not None:
                message+=f' {self.counter_name}'
            if self.expected_total is not None:
                estimated : timedelta = self.estimated_time_left() #type:ignore
                percent_done : float = self.percent_done() #type:ignore
                message += (f' / {self.expected_total} ({percent_done:.2%})\n'
                        f'\t\tEstimated time left: {estimated}'
                )
            return message

class LogProgressContextManager(ProgressContextManager):
    def __init__(self, 
                 target_out : logging.Logger | TextIO, 
                 log_level : int = logging.INFO, 
                 cooldown : timedelta | float = timedelta(minutes=2)):
        self.log_level = log_level
        self.cooldown = cooldown
        self.target_out = target_out
    
    def get_callback(self, 
                     log_callback : Callable[[str], None] | None,
                     logger : logging.Logger | None,
                     stream : TextIO | None,
                     log_level : int | None) -> Callable[[str], None]:
        if log_level is None:
            log_level = self.log_level
        if sum(1 if x is not None else 0 for x in (log_callback, logger, stream)) > 1:
            warnings.warn("Only one of log_callback, logger or stream should be provided")
        if log_callback is not None:
            return log_callback
        else:
            if logger is None and stream is None:
                if isinstance(self.target_out, logging.Logger):
                    logger = self.target_out
                else:
                    stream = self.target_out
            if logger is not None:
                return lambda message: logger.log(log_level, message)
            else:
                return lambda message: print(message, file=stream)

    def _enter(self, 
               task_name : str, 
               counter_name : str | None = None,
               expected_total : float | Any | None = None,
               cooldown : timedelta | float | None = None,
               log_callback : Callable[[str], None] | None = None,
               logger : logging.Logger | None = None,
               log_level : int | None = None,
               stream : TextIO | None = None,
               **kwargs) -> ProgressTracker:
        cooldown = cooldown if cooldown is not None else self.cooldown
        log_callback = self.get_callback(log_callback, logger, stream, log_level)
        tracker = LogProgressTracker(
            task_name=task_name,
            log_callback=log_callback,
            cooldown=cooldown,
            counter_name=counter_name,
            expected_total=expected_total,
            **kwargs
        )
        tracker.log_callback(f'{task_name} starting')
        return tracker


    def _exit(self, tracker : LogProgressTracker) -> None:
        tracker.log_callback(f'{tracker.task_name} complete ({tracker.elapsed_time()})')

    def but(self, log_level : int | None = None, cooldown : timedelta | float | None = None) -> 'LogProgressContextManager':
        """Return a clone of this context manager with a new level/cooldown

        Args:
            log_level (int | None): the new level or `None` to not change

        Returns:
            LogProgressContextManager: a clone of this context manager with the new log level/ cooldonw
        """
        log_level = log_level if log_level is not None else self.log_level
        cooldown = cooldown if cooldown is not None else self.cooldown
        return LogProgressContextManager(self.target_out, log_level, cooldown)



# def trackable[**P, R](task_name : str | None = None, counter_name : str | None = None, **dec_kwargs) -> Callable[
#     [Callable[P, Generator[float, None, R]]],
#     Callable[Concatenate[ProgressContextManager, P], R]
# ]:
#     def decorator(func : Callable[P, Generator[float, None, R]]):
#         if task_name is None:
#             tname = func.__qualname__
#         else:
#             tname = task_name
#         @wraps(func)
#         def wrapper(*args, progress_cm=NULL_PROGRESS_CM, **kwargs):
#             with progress_cm.track(tname, counter_name=counter_name, **dec_kwargs) as tracker:
#                 it = iter(func(*args, **kwargs))
#                 try:
#                     while True:
#                         steps = next(it)
#                         if steps is None:
#                             steps = 1
#                         tracker.tick(steps)
#                 except StopIteration as e:
#                     return e.value
#         return wrapper
#     return decorator