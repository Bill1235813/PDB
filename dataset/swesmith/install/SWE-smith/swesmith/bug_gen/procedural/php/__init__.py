"""
PHP procedural modifiers for bug generation.

Uses PHP-specific tree-sitter grammar for accurate AST parsing of PHP syntax
(e.g., ->, ::, foreach, typed catches, string concatenation with .).
"""

from swesmith.bug_gen.procedural.base import ProceduralModifier

from swesmith.bug_gen.procedural.php.operations import (
    OperationChangeModifier,
    OperationFlipOperatorModifier,
    OperationSwapOperandsModifier,
    OperationChangeConstantsModifier,
    OperationBreakChainsModifier,
    AugmentedAssignmentSwapModifier,
    TernaryOperatorSwapModifier,
    FunctionArgumentSwapModifier,
)
from swesmith.bug_gen.procedural.php.control_flow import (
    ControlIfElseInvertModifier,
    ControlShuffleLinesModifier,
)
from swesmith.bug_gen.procedural.php.remove import (
    RemoveLoopModifier,
    RemoveConditionalModifier,
    RemoveAssignmentModifier,
    RemoveTernaryModifier,
)

MODIFIERS_PHP: list[ProceduralModifier] = [
    # Operation modifiers (8)
    OperationChangeModifier(likelihood=0.5),
    OperationFlipOperatorModifier(likelihood=0.5),
    OperationSwapOperandsModifier(likelihood=0.5),
    OperationChangeConstantsModifier(likelihood=0.5),
    OperationBreakChainsModifier(likelihood=0.5),
    AugmentedAssignmentSwapModifier(likelihood=0.5),
    TernaryOperatorSwapModifier(likelihood=0.5),
    FunctionArgumentSwapModifier(likelihood=0.5),
    # Control flow modifiers (2)
    ControlIfElseInvertModifier(likelihood=0.5),
    ControlShuffleLinesModifier(likelihood=0.5),
    # Remove modifiers (4)
    RemoveLoopModifier(likelihood=0.5),
    RemoveConditionalModifier(likelihood=0.5),
    RemoveAssignmentModifier(likelihood=0.5),
    RemoveTernaryModifier(likelihood=0.5),
]
