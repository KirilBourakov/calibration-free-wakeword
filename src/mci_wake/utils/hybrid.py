class hybrid_method:
    """A descriptor that behaves as a classmethod when called from a class,

    and an instancemethod when called from an instance.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner=None):
        if instance is None:
            # Called via Class.method(...)
            return lambda *args, **kwargs: self.func(owner, *args, **kwargs)
        # Called via instance.method(...)
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)