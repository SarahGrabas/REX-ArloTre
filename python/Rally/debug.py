VERBOSITY : int = 1 # 0 means no debug info, 1 means all, and higher values means less verbosity

# in case verbosity is false, we want debug calls to not throw an unknown expression error
def debug(*args, **kwargs): pass

# if verbosity is true, a debug function which prints to out in a standard way, is defined
if VERBOSITY:
    # defines elapsed_time function for getting time since program start, used by the debug print function
    from time import perf_counter

    _START_TIME = perf_counter()

    def _elapsed_time():
        elapsed_ms = int((perf_counter() - _START_TIME) * 1000)

        minutes = elapsed_ms // 60_000
        seconds = (elapsed_ms // 1_000) % 60
        milliseconds = elapsed_ms % 1_000

        return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    # defines the debug print function
    def debug(*values: object, verbosity:int=9):
        """A debug print function used for printing formatted debug information to out, if verbosity is true.\\
        The higher `verbosity` is set (at least 1), the higher the value, the more important the debug info is."""
        if verbosity >= VERBOSITY:
            print(
                f"[{_elapsed_time()}] ({verbosity}) ",
                *values,
                file = None 
            )
        else:
            assert verbosity >= 1, f"Error: Expected verbosity to be positive, but recieved {verbosity}"

