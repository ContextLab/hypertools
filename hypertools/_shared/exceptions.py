class HypertoolsError(Exception):
    pass


class HypertoolsBackendError(HypertoolsError):
    def __init__(self, message):
        super.__init__(message)
        self.message = message