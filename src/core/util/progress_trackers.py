"""Utility classes for keeping track of progress on long tasks
"""
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from typing import Callable, ContextManager, Concatenate, Generator, TextIO
from contextlib import contextmanager
from functools import wraps
import logging

class ProgressTracker:
    def __init__(
            self, 
            expected_total : float | None = None,
            callbacks : list[Callable[['ProgressTracker'], None]] | None = None,
            counter_name : str | None = ""
        ):
        self.progress : float = 0
        self.expected_total = expected_total
        self.callbacks = callbacks or []
        self.start_time = datetime.now()
        self.counter_name = counter_name
    
    
    def tick(self, steps : float = 1, expected_total : float | None = None):
        """Ticks the tracker, adding `steps` to progress counter

        Args:
            steps (float): Amount of steps taken since last tick
            expected_total (float | None, optional): The total expected number of steps. 
                Can be used to defer determining the total or to modify the total later. Defaults to None.
        """
        self.set_progress(self.progress + steps)

    
    def set_progress(self, progress : float, expected_total : float | None = None):
        """Ticks the tracker, overwritin progress with the given argument

        Args:
            progress (float): Amount of steps taken since last tick
            expected_total (float | None, optional): The total expected number of steps. 
                Can be used to defer determining the total or to modify the total later. Defaults to None.
        """
        self.progress = progress
        if expected_total is not None:
            self.expected_total = expected_total
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

class LogProgressTracker(ProgressTracker):
        def __init__(self, task_name : str, cooldown : timedelta, log_callback : Callable[[str], None], **kwargs):
            super().__init__(**kwargs)
            self.last_log = self.start_time
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

@contextmanager
def log_cooldown(
        logger : logging.Logger | TextIO, 
        task_name : str,
        cooldown : float | timedelta = timedelta(minutes=2),
        counter_name : str | None = None,
        log_level : int = logging.INFO,
        expected_total : int | None = None
    ):
    if not isinstance(cooldown, timedelta):
        cooldown = timedelta(seconds=cooldown)
    def log(msg):
        if isinstance(logger, logging.Logger):
            logger.log(log_level, msg)
        else:
            logger.write(msg + '\n')
    log(f'{task_name} starting')

    tracker = LogProgressTracker(
        cooldown=cooldown,
        task_name= task_name,
        expected_total=expected_total,
        log_callback=log,
        counter_name=counter_name
    )
    try:
        yield tracker
    finally:
        log(f'{task_name} complete ({tracker.elapsed_time()})')



