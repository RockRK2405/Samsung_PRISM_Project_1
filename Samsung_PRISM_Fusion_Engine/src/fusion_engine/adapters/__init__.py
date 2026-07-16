"""Per-modality adapters — each wraps one teammate's model behind a
uniform ``predict(...) -> ModalityResult`` call.

Adapters are imported lazily (inside ``FusionEngine.__init__``, not
here) so that a missing dependency or checkpoint for one modality
(e.g. Image's torchvision, or a not-yet-trained Text model) never
prevents importing the package as a whole.
"""
