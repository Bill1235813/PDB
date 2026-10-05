"""
JavaScript procedural modifiers for bug generation.

KNOWN ISSUES / TODO:
=====================

1. RemoveAssignmentModifier - Only handles `var`, not `let`/`const`
   - Location: remove.py line 157-161 and adapters/javascript.py line 67
   - The modifier only removes `variable_declaration` nodes (var).
   - Modern JS `let`/`const` use `lexical_declaration` which is not handled.
   - Fix: Add "lexical_declaration" to both the adapter's HAS_ASSIGNMENT detection
     and the modifier's collect_assignments() function.

2. ControlShuffleLinesModifier - Requires HAS_LOOP condition (inherited)
   - Location: base.py CommonPMs.CONTROL_SHUFFLE_LINES
   - Functions without loops cannot have their lines shuffled due to condition.
   - This may be intentional design, but limits applicability to modern JS code.
   - Consider: Either remove HAS_LOOP requirement or create a separate modifier.

3. OperationBreakChainsModifier - Cannot handle parenthesized expressions
   - Location: operations.py line 365-405
   - Only checks if child.type == "binary_expression" directly.
   - When parentheses are used like `x * (y * z)`, the right side is
     `parenthesized_expression` containing binary_expression, not detected.
   - Fix: Unwrap parenthesized_expression nodes when checking for chains.

4. RemoveAssignmentModifier - Produces incorrect indentation after removal
   - Location: remove.py line 180-193
   - After removing an assignment, remaining code has extra indentation.
   - This is cosmetic but may cause issues with whitespace-sensitive tooling.
   - Fix: Adjust byte offset handling to properly handle leading whitespace.

5. JavaScriptEntity adapter - Missing lexical_declaration in assignment detection
   - Location: adapters/javascript.py line 67
   - Related to issue #1 above.
   - `let` and `const` declarations don't get HAS_ASSIGNMENT tag.
   - Fix: Add "lexical_declaration" to the type check.
"""

from swesmith.bug_gen.procedural.base import ProceduralModifier

from swesmith.bug_gen.procedural.javascript.operations import (
    OperationChangeModifier,
    OperationFlipOperatorModifier,
    OperationSwapOperandsModifier,
    OperationChangeConstantsModifier,
    OperationBreakChainsModifier,
    AugmentedAssignmentSwapModifier,
    TernaryOperatorSwapModifier,
    FunctionArgumentSwapModifier,
)
from swesmith.bug_gen.procedural.javascript.control_flow import (
    ControlIfElseInvertModifier,
    ControlShuffleLinesModifier,
)
from swesmith.bug_gen.procedural.javascript.remove import (
    RemoveLoopModifier,
    RemoveConditionalModifier,
    RemoveAssignmentModifier,
    RemoveTernaryModifier,
)

MODIFIERS_JAVASCRIPT: list[ProceduralModifier] = [
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
