import functools
import os
from typing import Callable


# If the environment variable 'LOCAL_RANK' is not set, a single
# process is running, so os.environ.get('LOCAL_RANK', -1) returns
# -1.
def is_multi_process():
    return int(os.environ.get('LOCAL_RANK', -1)) != -1


def is_main_process():
    return int(os.environ.get('LOCAL_RANK', -1)) in [-1, 0]


def on_process(func: Callable, process_index: int):

    @functools.wraps(func)
    def _wrapper(model, *args, **kwargs):
        curr_process_index = int(os.environ.get('LOCAL_RANK', -1))

        if curr_process_index == -1 or (process_index == curr_process_index):
            print(f"Applying {func.__name__} on process index {curr_process_index}")
            return func(model, *args, **kwargs)
        else:
            print(f"Skipping function {func.__name__} on process index {curr_process_index}")
            return model

    return _wrapper


on_main_process = functools.partial(on_process, process_index=0)
