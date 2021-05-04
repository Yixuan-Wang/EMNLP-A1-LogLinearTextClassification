from rich.console import Console
from rich.progress import (BarColumn, Progress, TaskID, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.prompt import Confirm

console = Console()
print = console.print

progress_layout = (
    "[progress.description]{task.description:<50}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

def use_hook(hook=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if hook != None: hook(None)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def hook_advance(progress: Progress, task: TaskID):
    return lambda _: progress.advance(task_id=task)

def format_bool(val: bool):
    return '[green]true[/green]' if val else '[red]false[/red]'
