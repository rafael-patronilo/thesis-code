from typing import Iterable, Optional, Callable, Sequence, Literal, TYPE_CHECKING
import logging
import re
import signal
if TYPE_CHECKING:
    import torch

def multiline_str(obj, level = 0) -> str:
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        body = ",\n".join([f"{'\t'*(level + 1)}{k}: {multiline_str(v, level=level+2)}" for k, v in obj.items()])
        return ("{\n" + 
                body +
                "\n" + "\t"*level + "}"
                )
    elif isinstance(obj, Iterable):
        enclosing = ('(', ')')
        if isinstance(obj, list):
            enclosing = ('[', ']')
        elif isinstance(obj, set):
            enclosing = ('{', '}')
        body = ",\n".join(f"{'\t'*(level + 1)}{multiline_str(x, level=level+2)}" for x in obj)
        return (f"{enclosing[0]}\n" + body + f"\n{'\t'*level}{enclosing[1]}")
    else:
        return str(obj)

def safe_div(a, b):
    if b == 0:
        return float('nan')
    return a / b

def assert_type(obj, type_):
    if not isinstance(obj, type_):
        raise TypeError(f"Expected type {type_}, got {type(obj)}")
    
_IMPORT_REGEX = re.compile(r"^[a-zA-Z0-9_]+(.[a-zA-Z0-9_]+)*$")

def is_import_safe(import_str : str) -> bool:
    return _IMPORT_REGEX.fullmatch(import_str) is not None

def conv_out_shape(
        in_shape : Sequence[int], 
        kernel_size : int, 
        stride : int = 1, 
        padding : Literal['same'] | int ='same', 
        dilation : int = 1
    ):
    if padding == 'same':
        padding = (kernel_size - 1) // 2
    out_shape = []
    for size in in_shape:
        value = size + 2 * padding - dilation * (kernel_size - 1) - 1
        value = value // stride
        value += 1
        out_shape.append(value)
    return out_shape

def transposed_conv_out_shape(
        in_shape : Sequence[int], 
        kernel_size : int, 
        stride : int = 1, 
        padding : Literal['same'] | int ='same', 
        dilation : int = 1
    ):
    if padding == 'same':
        padding = (kernel_size - 1) // 2
    out_shape = []
    for size in in_shape:
        value = (size - 1) * stride - 2 * padding + dilation*(kernel_size-1) + 1 
        out_shape.append(value)
    return out_shape


class NoInterrupt:
    """
    Context object to trap interrupt signals to prevent abrupt termination of certain sections of the program.
    If a signal is received during the with context, it is delayed until its end.
    """
    attempts = 0

    class ForcedInterruptException(Exception):
        def __init__(self, signal):
            self.signal = signal

    class InterruptException(Exception):
        def __init__(self, signal):
            self.signal = signal
    
    def __init__(
            self,
            reason : Optional[str] = None,
            logger :  Optional[logging.Logger] = None,
            signals : list[signal.Signals] = [signal.Signals.SIGINT, signal.Signals.SIGTERM], 
            keyboard_interrupt = True):
        self.reason = reason
        self.logger = logger or logging.getLogger('InterruptHandler')
        self.signals = signals
        self.interrupt_signal = None
        self.keyboard_interrupt = keyboard_interrupt
        self.old_handlers = {}

    def signal_handler(self, sig, frame):
        if self.attempts > 0:
            self.logger.error(f"Received signal {signal.Signals(sig).name} again, exiting immediately")
            raise NoInterrupt.InterruptException(self.interrupt_signal)
        self.attempts += 1
        self.interrupt_signal = signal.Signals(sig)
        self.logger.warning(f"Intercepted signal {self.interrupt_signal.name} with reason {self.reason}.\n"
                            "Attempting graceful exit")
    
    def __enter__(self):
        for sig in self.signals:
            self.old_handlers[sig] = signal.signal(sig, self.signal_handler)
        self.logger.debug(f"Interrupt signals trapped with reason {self.reason}")

    def __exit__(self, exc_type, exc_value, traceback):
        for sig, handler in self.old_handlers.items():
            signal.signal(sig, handler)
        self.logger.debug(f"Released interrupt signals")
        if exc_value is not None:
            raise exc_value
        if self.interrupt_signal is not None:
            if self.keyboard_interrupt and self.interrupt_signal == signal.Signals.SIGINT:
                raise KeyboardInterrupt()
            else:
                raise NoInterrupt.InterruptException(self.interrupt_signal)

class DecoratedTorchMetric:

    def __init__(
            self, 
            metric : Callable[['torch.Tensor', 'torch.Tensor'], 'torch.Tensor'],
            flatten_tensors : bool = False
        ) -> None:
        self.metric = metric
        self.flatten_tensors = flatten_tensors

    def __call__(self, y_pred, y_true, **kwargs):
        if self.flatten_tensors:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        return self.metric(y_pred, y_true).item()