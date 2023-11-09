from contextlib import contextmanager


class Retry:
    def __init__(self, n_allowed_fails=0):
        self.n_allowed_fails = n_allowed_fails
        self.n_fails = 0

    @contextmanager
    def catch_assert(self, i: int):
        try:
            yield
        except AssertionError as err:
            print(err)
            self.n_fails += 1
            msg = f"Failed {self.n_fails} times after {i + 1} tries"
            if self.n_fails > self.n_allowed_fails:
                raise AssertionError(msg) from err

    def reset(self):
        self.n_fails = 0
