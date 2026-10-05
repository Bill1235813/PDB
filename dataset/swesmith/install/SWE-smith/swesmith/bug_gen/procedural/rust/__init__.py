from swesmith.bug_gen.procedural.base import ProceduralModifier
from swesmith.bug_gen.procedural.rust.control_flow import (
    ControlIfElseInvertModifier,
    ControlShuffleLinesModifier,
)
from swesmith.bug_gen.procedural.rust.operations import (
    OperationBreakChainsModifier,
    OperationChangeConstantsModifier,
    OperationChangeModifier,
    OperationFlipOperatorModifier,
    OperationSwapOperandsModifier,
)
from swesmith.bug_gen.procedural.rust.remove import (
    RemoveAssignModifier,
    RemoveConditionalModifier,
    RemoveLoopModifier,
)

MODIFIERS_RUST: list[ProceduralModifier] = [
    ControlIfElseInvertModifier(likelihood=0.5),
    ControlShuffleLinesModifier(likelihood=0.5),
    RemoveAssignModifier(likelihood=0.5),
    RemoveConditionalModifier(likelihood=0.5),
    RemoveLoopModifier(likelihood=0.5),
    OperationBreakChainsModifier(likelihood=0.5),
    OperationChangeConstantsModifier(likelihood=0.5),
    OperationChangeModifier(likelihood=0.5),
    OperationFlipOperatorModifier(likelihood=0.5),
    OperationSwapOperandsModifier(likelihood=0.5),
]
