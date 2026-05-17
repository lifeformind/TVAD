"""Shared error-message + exit-code helpers for the CLI passes.

Used by transcribe.py, sentiment.py, metrics.py, and prosody.py to print
consistent red-formatted error messages plus optional dim hints, and return
the correct exit code. Centralizes the rich-formatting conventions so all
4 passes produce identical-looking error output.

Exit code semantics (matches the docs/INTEGRATION.md contract):
  2 = user-bad-input  (missing/malformed JSON, missing prerequisite, etc.)
  3 = env / IO / model failure
"""

from rich.console import Console

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_ENV_OR_IO = 3

# Module-level console — same default as the CLI passes use.
_console = Console()


def print_bad_input(reason: str, hint: str | None = None) -> int:
    """Print a red error for user-supplied bad input and return EXIT_BAD_INPUT.

    Args:
        reason: the primary error message. Will be wrapped in [red]...[/]. If
            the caller wants inline rich markup in `reason` (e.g. [bold]xyz[/]),
            include it directly — only the outer [red]…[/] wrapper is added here.
        hint: optional follow-up message printed in [dim]...[/] (e.g. "Run
            transcribe.py first."). Pass None to omit.

    Returns:
        EXIT_BAD_INPUT (2). Caller does `return print_bad_input(...)`.
    """
    _console.print(f"[red]{reason}[/]")
    if hint:
        _console.print(f"[dim]{hint}[/]")
    return EXIT_BAD_INPUT


def print_env_failure(reason: str, hint: str | None = None) -> int:
    """Print a red error for environment/IO/model failures and return EXIT_ENV_OR_IO.

    Same shape as print_bad_input but signals "fix your environment" rather
    than "fix your input". Returns 3.
    """
    _console.print(f"[red]{reason}[/]")
    if hint:
        _console.print(f"[dim]{hint}[/]")
    return EXIT_ENV_OR_IO
