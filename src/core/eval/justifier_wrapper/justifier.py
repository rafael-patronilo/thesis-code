from abc import ABC, abstractmethod
import multiprocessing
from concurrent.futures import Executor
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import AbstractContextManager
import queue
from pathlib import Path
from subprocess import Popen
import subprocess
import threading
import warnings
from concurrent.futures import Future

from core.eval.justifier_wrapper.justifier_result import Justification
from core.init import options
from typing import Coroutine, Iterable, Literal, NamedTuple, Optional, Protocol, Iterator, Callable, Any, Generator
import logging

from core.eval.justifier_wrapper.justifier_output_handler import parse_json
from core.util.progress_trackers import ProgressTracker, NULL_PROGRESS_TRACKER

logger = logging.getLogger(__name__)


class JustifierConfig(NamedTuple):
    ontology_file: Path


class JustifierContext(NamedTuple):
    process: Popen
    config: JustifierConfig

class _ThreadLocal(threading.local):
    def __init__(self, /, **_):
        self.context : Optional[JustifierContext] = None
_thread_local = _ThreadLocal()


def _create_context(config: JustifierConfig):
    return JustifierContext(
        process=Popen(
            ['java', '-jar', options.justifier_jar, str(config.ontology_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ),
        config=config
    )

def init_justifier_worker(config: JustifierConfig):
    """
    Should be called by threads/processes intended to handle justification jobs
    :param config: The config that will be used in each thread or process
    :return:
    """
    if multiprocessing.parent_process() is None:
        logger.warning("Justifier worker is being initialized in the main process. This is not recommended.")
    global _thread_local
    if _thread_local.context is not None:
        logger.warning("Justifier worker is already running.")
        return
    _thread_local.context = _create_context(config)


class JustifierArgs(NamedTuple):
    entailment_class : str
    observations : list[tuple[str, float]]

class JustifierResult(NamedTuple):
    args : JustifierArgs
    justifications : list[Justification]

def _inject_input(context: JustifierContext, args : JustifierArgs):
    process = context.process
    stdin = process.stdin
    assert stdin is not None
    stdin.write(f"{args.entailment_class}\n".encode())
    for concept, belief in args.observations:
        stdin.write(f"{concept}, {belief}\n".encode())
    stdin.write("\n")

def _extract_output(context : JustifierContext) -> str:
    process = context.process
    stdout = process.stdout
    assert stdout is not None
    lines = []
    line = stdout.readline()
    while len(line) > 0:
        lines.append(line)
        line = stdout.readline()
    return "".join(lines)

def _justify(context: JustifierContext, args : JustifierArgs) -> JustifierResult:
    _inject_input(context, args)
    output = _extract_output(context)
    return JustifierResult(
        args,
        parse_json(output, args.entailment_class, args.observations)
    )


class AbstractJustifierWrapper(AbstractContextManager, ABC):
    def __init__(self, config : JustifierConfig):
        self.config = config

    @abstractmethod
    def justify(self, args : JustifierArgs) -> JustifierResult:
        raise NotImplementedError()
    
    def justify_multiple(self, queries : Iterable[JustifierArgs]) -> Iterable[JustifierResult]:
        return map(self.justify, queries)


class JustifierWrapper(AbstractJustifierWrapper):
    def __init__(self, config : JustifierConfig):
        self.context : Optional[JustifierContext] = None
        super().__init__(config)

    def __enter__(self) -> JustifierContext:
        if self.context is not None:
            raise RuntimeError("JustifierWrapper is already open")
        self.context = _create_context(self.config)
        return self.context

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context is None:
            raise RuntimeError("JustifierWrapper is not open")
        self.context.process.kill()
        try:
            self.context.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Justifier process did not terminate in time; left dangling")
        self.context = None
        return False

    def justify(self, args: JustifierArgs) -> JustifierResult:
        if self.context is None:
            raise RuntimeError("JustifierWrapper is not open")
        return _justify(self.context, args)

class JustifierExecutorFactoryProtocol(Protocol):
    def __call__(self, config : JustifierConfig) -> Executor:
        ...

class JustifierExecutorFactory:
    def __init__(
            self,
            executor_type : Literal['process', 'thread'] = 'process',
            num_workers : int = options.num_justifiers
    ):
        self.executor_type = executor_type
        self.num_workers = num_workers

    def __call__(self, config : JustifierConfig) -> Executor:
        num_workers = self.num_workers if self.num_workers > 0 else None
        match self.executor_type:
            case 'process':
                return ProcessPoolExecutor(
                    num_workers,
                    initializer=init_justifier_worker,
                    initargs=(config,)
                )
            case 'thread':
                return ThreadPoolExecutor(num_workers,
                    initializer=init_justifier_worker,
                    initargs=(config,))
            case other:
                raise ValueError(f"Invalid executor type: {other}")

def _thread_local_justify(args : JustifierArgs) -> JustifierResult:
    if _thread_local.context is None:
        raise RuntimeError("Justifier worker is not initialized")
    return _justify(_thread_local.context, args)

class _PrePostProcessingJustifyJob[P, T]:
    def __init__(
            self,
            preprocessor : Callable[[P], JustifierArgs],
            processor : Callable[[JustifierResult], T],
    ):
        self.preprocessor = preprocessor
        self.processor = processor

    def __call__(self, payload : P) -> T:
        x = self.preprocessor(payload)
        x = _thread_local_justify(x)
        return self.processor(x)

class ParallelJustifierWrapper(AbstractJustifierWrapper):
    def __init__(
            self,
            config : JustifierConfig,
            executor_factory : Optional[JustifierExecutorFactoryProtocol] = None,
            chunk_size : int = 1
    ):
        super().__init__(config)
        self.executor_factory = executor_factory or JustifierExecutorFactory()
        self.chunk_size = chunk_size
        self.executor : Optional[Executor] = None
        self.warning_given = False

    def __enter__(self):
        if self.executor is not None:
            raise RuntimeError("ParallelJustifierWrapper is already open")
        self.executor = self.executor_factory(self.config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.executor is None:
            raise RuntimeError("ParallelJustifierWrapper is not open")
        self.executor.shutdown()
        self.executor = None
        return False

    def justify(self, args: JustifierArgs) -> JustifierResult:
        if not self.warning_given:
            warnings.warn(f"{self.__class__.__name__}.justify is performed synchronously and "
                      "does not take advantage of parallelism; "
                      "use justify_multiple or justify_future instead")
            self.warning_given = True
        return self.justify_future(args).result()
        
    def justify_future(self, args : JustifierArgs) -> Future[JustifierResult]:
        if self.executor is None:
            raise RuntimeError("ParallelJustifierWrapper is not open")
        return self.executor.submit(_thread_local_justify, args)

    def justify_multiple(self, queries: Iterable[JustifierArgs]) -> Iterable[JustifierResult]:
        if self.executor is None:
            raise RuntimeError("ParallelJustifierWrapper is not open")
        return self.executor.map(_thread_local_justify,
                                 queries, chunksize=self.chunk_size)

    def from_producer[P, T, C](self,
                               producer : Iterable[P],
                               preprocessor : Callable[[P], JustifierArgs],
                               postprocessor : Callable[[JustifierResult], T],
                               reducer : Callable[[Iterable[T]], C],
                               progress_tracker : ProgressTracker = NULL_PROGRESS_TRACKER,
                               queue_maxsize : int = 64
                               ) -> C:
        """
        Parallelize the justification of a sequence of items produced by a generator.

        Note that the preprocessor and postprocessor will be submitted to the Justifier's Executor
        to run in parallel for each task.
        This means that if the Executor is a ProcessPoolExecutor, the preprocessor and postprocessor
        must be picklable objects such as global functions.

        The producer and reducer will always run in the calling process. The producer will run
        in the current thread, while the reducer will run in a new thread. The producer will submit
        the tasks to the Executor and send the Futures to the reducer thread through a queue.

        :param producer: An Iterable (probably a generator) that produces items to be justified.
            The producer will be run in the current thread.
        :param preprocessor: A preprocessing callable that will be called in parallel between
            the producer and the justifier. If using a ProcessPoolExecutor, this callable must be
            a global function or another picklable object (see above).
        :param postprocessor: A postprocessing callable that will be called in parallel between
            the justifier and the reducer. If using a ProcessPoolExecutor, this callable must be
            a global function or another picklable object (see above).
        :param reducer: A reduction function that iterates all results. This function will run
            in a new threading.Thread in parallel with the producer.
        :param progress_tracker: An optional progress tracker for results.
        :param queue_maxsize: The maximum amount of Futures that can be queued at any time.
        :return: The result of the reducer.
        """
        executor = self.executor
        if executor is None:
            raise RuntimeError("ParallelJustifierWrapper is not open")
        job = _PrePostProcessingJustifyJob(preprocessor, postprocessor)
        producer_queue : queue.Queue[Optional[Future[T]]] = queue.Queue(queue_maxsize)
        def future_producer():
            for x in producer:
                future = executor.submit(job, x)
                producer_queue.put(future)

        def future_iterable():
            while True:
                future = producer_queue.get()
                if future is None:
                    break
                yield future.result()
                progress_tracker.tick()

        reducer_result : list[C] = []
        def future_reducer():
            reducer_result.append(reducer(future_iterable()))
        reducer_thread = threading.Thread(
            target=future_reducer
        )
        reducer_thread.start()
        future_producer()
        producer_queue.put(None)
        reducer_thread.join()
        return reducer_result[0]



