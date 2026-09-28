"""Bounded CPU cache of exact official encoder outputs, scoped to one pipeline."""
from collections import OrderedDict
import hashlib
import inspect
import weakref


def fingerprint(value):
    from PIL import Image
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(fingerprint(item) for item in value))
    if isinstance(value, Image.Image):
        # Include actual pixels, not object identity or a filename that may change.
        digest = hashlib.sha256(value.tobytes())
        if value.mode == 'P':
            digest.update(bytes(value.getpalette() or []))
        return ('image', value.mode, value.size, digest.digest())
    raise TypeError('Unsupported prompt cache input')


def tensors_map(value, operation):
    import torch
    if isinstance(value, torch.Tensor):
        return operation(value)
    if isinstance(value, tuple):
        return tuple(tensors_map(item, operation) for item in value)
    if isinstance(value, list):
        return [tensors_map(item, operation) for item in value]
    if value is None:
        return None
    raise TypeError('Unsupported encoder output')


class PromptCache:
    def __init__(self, original, max_bytes=64*2**20, max_entries=4):
        # A strong bound-method reference would form a pipeline/cache cycle and
        # delay releasing GPU weights on checkpoint changes.
        self._original_ref = weakref.WeakMethod(original) if inspect.ismethod(original) else None
        self._original_func = None if self._original_ref is not None else original
        self.signature = inspect.signature(original)
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(1, int(max_entries))
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def original(self):
        method = self._original_ref() if self._original_ref is not None else self._original_func
        if method is None:
            raise RuntimeError('Boogu encoder has been released')
        return method

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def __call__(self, *args, **kwargs):
        if not self.max_bytes:
            return self.original(*args, **kwargs)
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        device = bound.arguments.get('device')
        if device is None:
            device = getattr(getattr(self.original, '__self__', None), '_execution_device', 'cpu')
        try:
            key = tuple((name, fingerprint(value)) for name, value in bound.arguments.items() if name != 'device')
        except TypeError:
            return self.original(*args, **kwargs)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            stored, _ = self.entries[key]
            # A caller may mutate views or masks; never expose cached storage.
            return tensors_map(stored, lambda tensor: tensor.to(device=device, copy=True))
        self.misses += 1
        output = self.original(*args, **kwargs)
        size = 0
        def count(tensor):
            nonlocal size
            size += tensor.numel() * tensor.element_size()
            return tensor
        try:
            tensors_map(output, count)
        except TypeError:
            return output
        if size <= self.max_bytes:
            while self.entries and (self.bytes+size>self.max_bytes or len(self.entries)>=self.max_entries):
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            stored = tensors_map(output, lambda tensor: tensor.detach().to(device='cpu', copy=True))
            self.entries[key] = (stored, size)
            self.bytes += size
        return output


def install(pipe, megabytes=64):
    # Releasing/rebuilding the pipeline invalidates all cached text and references,
    # including checkpoint, encoder and adapter changes in pipeline.get's identity.
    if megabytes <= 0:
        return
    cache = PromptCache(pipe._get_instruction_feature_embeds, max_bytes=int(megabytes*2**20))
    pipe._get_instruction_feature_embeds = cache
    pipe._boogu_prompt_cache = cache
