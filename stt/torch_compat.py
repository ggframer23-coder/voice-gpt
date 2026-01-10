"""
Compatibility fixes for PyTorch 2.6+ with WhisperX.

PyTorch 2.6+ changed torch.load() default from weights_only=False to weights_only=True,
which breaks WhisperX model loading. This module patches torch.load() to restore the old behavior.

This must be imported before any code that uses torch.load() (especially WhisperX).
"""

def patch_torch_load():
    """Patch torch.load to use weights_only=False for WhisperX compatibility."""
    try:
        import torch
        if not hasattr(torch, '_stt_original_load'):
            torch._stt_original_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                # Force weights_only=False for backward compatibility
                kwargs['weights_only'] = False
                return torch._stt_original_load(*args, **kwargs)

            torch.load = _patched_torch_load
    except ImportError:
        # torch not installed, no need to patch
        pass


# Apply patch when this module is imported
patch_torch_load()
