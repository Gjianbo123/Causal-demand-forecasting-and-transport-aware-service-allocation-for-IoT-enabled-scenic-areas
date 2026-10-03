"""Integration boundary. Deliberately does not pretend to reconstruct missing code."""
class OriginalBackend:
    def __init__(self, config):
        raise NotImplementedError(
            "Original ScenicFlow-Sim and frozen E-STGNN were not provided. "
            "Implement the Backend/Environment contracts in iotexp/contracts.py. "
            "See docs/INTEGRATION_ZH.md before running manuscript experiments."
        )
