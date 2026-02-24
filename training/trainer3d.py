"""
Trainer3D – thin subclass of Trainer that swaps in TrainingVisualizer3D.

All training, checkpointing, validation, EMA, and mixed-precision logic
is inherited unchanged from Trainer.  Only the visualizer is replaced so
that 5-D tensors (B, 1, D, H, W) are handled correctly.

Usage
-----
Replace every occurrence of ``Trainer`` with ``Trainer3D`` in the 3-D
training script.  All constructor arguments are identical.
"""

from .trainer    import Trainer
from .visualizer3d import TrainingVisualizer3D


class Trainer3D(Trainer):
    """
    Like :class:`Trainer` but uses :class:`TrainingVisualizer3D` so that
    5-D (B, C, D, H, W) tensors are rendered as orthogonal slice montages.
    """

    def __init__(
        self,
        *args,
        visualize: bool = True,
        viz_dir:   str  = "visualizations",
        **kwargs,
    ):
        # Tell the parent NOT to create a 2-D visualizer.
        super().__init__(*args, visualize=False, viz_dir=viz_dir, **kwargs)

        # Install 3-D visualizer in its place.
        self.visualize  = visualize
        self.visualizer = TrainingVisualizer3D(save_dir=viz_dir) if visualize else None
