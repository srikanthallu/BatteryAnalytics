import datetime
import time

def time_function(msg="Elapsed Time:"):
    def real_timing_function(function):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = function(*args, **kwargs)
            elapsed = time.time() - start_time
            print(msg, datetime.timedelta(seconds=elapsed))
            return res
        return wrapper
    return real_timing_function
