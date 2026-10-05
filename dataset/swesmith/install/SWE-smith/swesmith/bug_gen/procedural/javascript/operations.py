"""
JavaScript operation modifiers for procedural bug generation using tree-sitter.
"""

import sys
import tree_sitter_javascript as tsjs
from swesmith.bug_gen.procedural.javascript.base import JavaScriptProceduralModifier
from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.constants import CodeProperty, BugRewrite, CodeEntity
from tree_sitter import Language, Parser

JS_LANGUAGE = Language(tsjs.language())


def _safe_decode(bytes_obj, fallback=""):
    """Safely decode bytes to UTF-8, handling potential encoding errors."""
    try:
        return bytes_obj.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"WARNING: UTF-8 decode error: {e}", file=sys.stderr)
        return fallback


class OperationChangeModifier(JavaScriptProceduralModifier):
    """Change operators within similar groups (e.g., +/-, *//%, etc.)"""

    explanation: str = CommonPMs.OPERATION_CHANGE.explanation
    name: str = CommonPMs.OPERATION_CHANGE.name
    conditions: list = CommonPMs.OPERATION_CHANGE.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Change operators to others in their group."""

        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._change_operators(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _change_operators(self, source_code: str, node) -> str:
        """Find and change binary operators within their groups."""
        changes = []

        # Operator groups
        operator_groups = {
            "+": ["+", "-"],
            "-": ["+", "-"],
            "*": ["*", "/", "%"],
            "/": ["*", "/", "%"],
            "%": ["*", "/", "%"],
            "&": ["&", "|", "^"],
            "|": ["&", "|", "^"],
            "^": ["&", "|", "^"],
            "<<": ["<<", ">>"],
            ">>": ["<<", ">>"],
        }

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                # Find the operator child
                for child in n.children:
                    if child.type in operator_groups:
                        operator = child.type
                        group = operator_groups[operator]
                        # Choose a different operator from the group
                        other_ops = [op for op in group if op != operator]
                        if other_ops and self.flip():
                            new_op = self.rand.choice(other_ops)
                            changes.append({"node": child, "new_op": new_op})
                        break

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_code.encode("utf-8")
        for change in reversed(changes):
            node = change["node"]
            new_op = change["new_op"]

            modified_source = (
                modified_source[: node.start_byte]
                + new_op.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class OperationFlipOperatorModifier(JavaScriptProceduralModifier):
    """Flip operators to their opposites (e.g., == to !=, < to >, etc.)"""

    explanation: str = "The operators in an expression are likely incorrect."
    name: str = "func_pm_op_flip"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_BINARY_OP]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Flip operators to their opposites."""

        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._flip_operators(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _flip_operators(self, source_code: str, node) -> str:
        """Find and flip binary operators to their opposites."""
        changes = []

        operator_flips = {
            "===": "!==",
            "!==": "===",
            "==": "!=",
            "!=": "==",
            "<=": ">",
            ">=": "<",
            "<": ">=",
            ">": "<=",
            "&&": "||",
            "||": "&&",
            "+": "-",
            "-": "+",
            "*": "/",
            "/": "*",
        }

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                # Find the operator child
                for child in n.children:
                    if child.type in operator_flips:
                        if self.flip():
                            changes.append(
                                {"node": child, "new_op": operator_flips[child.type]}
                            )
                        break

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_code.encode("utf-8")
        for change in reversed(changes):
            node = change["node"]
            new_op = change["new_op"]

            modified_source = (
                modified_source[: node.start_byte]
                + new_op.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class OperationSwapOperandsModifier(JavaScriptProceduralModifier):
    """Swap operands in binary operations (e.g., a + b becomes b + a)"""

    explanation: str = CommonPMs.OPERATION_SWAP_OPERANDS.explanation
    name: str = CommonPMs.OPERATION_SWAP_OPERANDS.name
    conditions: list = CommonPMs.OPERATION_SWAP_OPERANDS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Swap left and right operands."""

        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._swap_operands(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_operands(self, source_code: str, node) -> str:
        """Find and swap operands in binary expressions."""
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_binary_ops(n):
            if n.type == "binary_expression" and len(n.children) >= 3:
                # Binary expression has: left, operator, right
                left = n.children[0]
                operator_node = n.children[1]
                right = n.children[2]

                if self.flip():
                    # For comparison operators, we might need to flip them too
                    operator = operator_node.type
                    if operator in ["<", ">", "<=", ">="]:
                        op_flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
                        operator = op_flip.get(operator, operator)

                    changes.append(
                        {
                            "node": n,
                            "left": left,
                            "right": right,
                            "operator": operator,
                        }
                    )

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for change in reversed(changes):
            node = change["node"]
            left = change["left"]
            right = change["right"]
            operator = change["operator"]

            left_text = _safe_decode(source_bytes[left.start_byte : left.end_byte])
            right_text = _safe_decode(source_bytes[right.start_byte : right.end_byte])

            # Swap: left op right -> right op left
            swapped = f"{right_text} {operator} {left_text}"

            modified_source = (
                modified_source[: node.start_byte]
                + swapped.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class OperationChangeConstantsModifier(JavaScriptProceduralModifier):
    """Change numeric constants to introduce off-by-one errors"""

    explanation: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.explanation
    name: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.name
    conditions: list = CommonPMs.OPERATION_CHANGE_CONSTANTS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Change constants by small amounts."""

        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._change_constants(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _change_constants(self, source_code: str, node) -> str:
        """Find and change numeric constants."""
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_numbers(n):
            if n.type == "number":
                if self.flip():
                    try:
                        value_text = _safe_decode(
                            source_bytes[n.start_byte : n.end_byte]
                        )
                        value = int(value_text)
                        # Small off-by-one changes
                        new_value = value + self.rand.choice([-1, 1, -2, 2])
                        changes.append({"node": n, "new_value": str(new_value)})
                    except ValueError:
                        pass  # Skip floats and hex numbers

            for child in n.children:
                collect_numbers(child)

        collect_numbers(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for change in reversed(changes):
            node = change["node"]
            new_value = change["new_value"]

            modified_source = (
                modified_source[: node.start_byte]
                + new_value.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class OperationBreakChainsModifier(JavaScriptProceduralModifier):
    """Break chained operations by removing parts of the chain"""

    explanation: str = CommonPMs.OPERATION_BREAK_CHAINS.explanation
    name: str = CommonPMs.OPERATION_BREAK_CHAINS.name
    conditions: list = CommonPMs.OPERATION_BREAK_CHAINS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Break chained binary operations."""

        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._break_chains(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _break_chains(self, source_code: str, node) -> str:
        """Find and break chained operations."""
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_chains(n):
            if n.type == "binary_expression" and len(n.children) >= 3:
                left = n.children[0]
                operator = n.children[1]
                right = n.children[2]

                # Check if left or right is also a binary expression (chain)
                if left.type == "binary_expression" and self.flip():
                    # Break left chain: keep only the right part
                    # (a + b) + c -> b + c (take right of left chain)
                    if len(left.children) >= 3:
                        left_right = left.children[2]
                        left_right_text = _safe_decode(
                            source_bytes[left_right.start_byte : left_right.end_byte]
                        )
                        operator_text = _safe_decode(
                            source_bytes[operator.start_byte : operator.end_byte]
                        )
                        right_text = _safe_decode(
                            source_bytes[right.start_byte : right.end_byte]
                        )
                        changes.append(
                            {
                                "node": n,
                                "replacement": f"{left_right_text} {operator_text} {right_text}",
                            }
                        )

                elif right.type == "binary_expression" and self.flip():
                    # Break right chain: keep only the left part
                    # a + (b + c) -> a + b (take left of right chain)
                    if len(right.children) >= 3:
                        right_left = right.children[0]
                        left_text = _safe_decode(
                            source_bytes[left.start_byte : left.end_byte]
                        )
                        operator_text = _safe_decode(
                            source_bytes[operator.start_byte : operator.end_byte]
                        )
                        right_left_text = _safe_decode(
                            source_bytes[right_left.start_byte : right_left.end_byte]
                        )
                        changes.append(
                            {
                                "node": n,
                                "replacement": f"{left_text} {operator_text} {right_left_text}",
                            }
                        )

            for child in n.children:
                collect_chains(child)

        collect_chains(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for change in reversed(changes):
            node = change["node"]
            replacement = change["replacement"]

            modified_source = (
                modified_source[: node.start_byte]
                + replacement.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class AugmentedAssignmentSwapModifier(JavaScriptProceduralModifier):
    """Swap augmented assignment operators (+=, -=, *=, /=, etc.) and update expressions (++, --)"""

    explanation: str = (
        "The augmented assignment or update operator is likely incorrect."
    )
    name: str = "func_pm_aug_assign_swap"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_ASSIGNMENT]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Swap augmented assignment operators."""

        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._swap_augmented_assignments(
            code_entity.src_code, tree.root_node
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_augmented_assignments(self, source_code: str, node) -> str:
        """Find and swap augmented assignment operators and update expressions."""
        changes = []
        source_bytes = source_code.encode("utf-8")

        # Augmented assignment operator swap pairs
        aug_assign_swaps = {
            # Arithmetic
            "+=": "-=",
            "-=": "+=",
            "*=": "/=",
            "/=": "*=",
            "%=": "/=",
            # Bitwise
            "&=": "|=",
            "|=": "&=",
            "^=": "&=",
            # Shift
            "<<=": ">>=",
            ">>=": "<<=",
            ">>>=": "<<=",  # Unsigned right shift
            # Logical (ES2021)
            "&&=": "||=",
            "||=": "&&=",
            "??=": "||=",  # Nullish coalescing assignment
            # Exponentiation
            "**=": "*=",
        }

        # Update expression swaps (++, --)
        update_swaps = {
            "++": "--",
            "--": "++",
        }

        def collect_augmented_assignments(n):
            # Handle augmented assignment expressions (+=, -=, etc.)
            if n.type == "augmented_assignment_expression":
                # Find the operator child
                for child in n.children:
                    op_text = source_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    if op_text in aug_assign_swaps and self.flip():
                        changes.append(
                            {"node": child, "new_op": aug_assign_swaps[op_text]}
                        )
                        break

            # Handle update expressions (++, --)
            elif n.type == "update_expression":
                for child in n.children:
                    op_text = source_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    if op_text in update_swaps and self.flip():
                        changes.append({"node": child, "new_op": update_swaps[op_text]})
                        break

            for child in n.children:
                collect_augmented_assignments(child)

        collect_augmented_assignments(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for change in reversed(changes):
            node = change["node"]
            new_op = change["new_op"]

            modified_source = (
                modified_source[: node.start_byte]
                + new_op.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class TernaryOperatorSwapModifier(JavaScriptProceduralModifier):
    """Modify ternary operators (condition ? consequent : alternative)"""

    explanation: str = "The ternary operator branches may be swapped or the condition may be incorrect."
    name: str = "func_pm_ternary_swap"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_TERNARY]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Modify ternary operators by swapping branches or negating conditions."""
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._modify_ternary(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _modify_ternary(self, source_code: str, node) -> str:
        """Find and modify ternary (conditional) expressions."""
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_ternary_ops(n):
            # In tree-sitter-javascript, ternary is "ternary_expression"
            if n.type == "ternary_expression" and len(n.children) >= 5:
                # Structure: condition ? consequent : alternative
                # Children: [condition, "?", consequent, ":", alternative]
                condition = None
                consequent = None
                alternative = None

                # Parse children - skip operators
                content_children = [c for c in n.children if c.type not in ["?", ":"]]
                if len(content_children) >= 3:
                    condition = content_children[0]
                    consequent = content_children[1]
                    alternative = content_children[2]

                    if condition and consequent and alternative and self.flip():
                        # Choose modification type randomly
                        mod_type = self.rand.choice(
                            ["swap_branches", "negate_condition"]
                        )
                        changes.append(
                            {
                                "node": n,
                                "condition": condition,
                                "consequent": consequent,
                                "alternative": alternative,
                                "mod_type": mod_type,
                            }
                        )

            for child in n.children:
                collect_ternary_ops(child)

        collect_ternary_ops(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for change in reversed(changes):
            node = change["node"]
            condition = change["condition"]
            consequent = change["consequent"]
            alternative = change["alternative"]
            mod_type = change["mod_type"]

            condition_text = _safe_decode(
                source_bytes[condition.start_byte : condition.end_byte]
            )
            consequent_text = _safe_decode(
                source_bytes[consequent.start_byte : consequent.end_byte]
            )
            alternative_text = _safe_decode(
                source_bytes[alternative.start_byte : alternative.end_byte]
            )

            if mod_type == "swap_branches":
                # Swap consequent and alternative: a ? b : c -> a ? c : b
                new_ternary = (
                    f"{condition_text} ? {alternative_text} : {consequent_text}"
                )
            else:  # negate_condition
                # Negate condition: a ? b : c -> !a ? b : c  (but keep branches, so effectively swaps logic)
                # Actually, negating and keeping same branches is same as swapping, so:
                # a ? b : c -> !(a) ? b : c  which equals a ? c : b
                # Let's do: negate condition AND swap branches for different bug pattern
                new_ternary = (
                    f"!({condition_text}) ? {consequent_text} : {alternative_text}"
                )

            modified_source = (
                modified_source[: node.start_byte]
                + new_ternary.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)


class FunctionArgumentSwapModifier(JavaScriptProceduralModifier):
    """Swap adjacent arguments in function calls."""

    explanation: str = "The function arguments may be in the wrong order."
    name: str = "func_pm_arg_swap"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_FUNCTION_CALL]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Swap adjacent arguments in function calls."""
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        modified_code = self._swap_arguments(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_arguments(self, source_code: str, node) -> str:
        """Find function calls and swap adjacent arguments."""
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_function_calls(n):
            if n.type == "call_expression":
                # Find the arguments node
                args_node = None
                for child in n.children:
                    if child.type == "arguments":
                        args_node = child
                        break

                if args_node:
                    # Get actual arguments (skip parentheses and commas)
                    args = [
                        c for c in args_node.children if c.type not in ["(", ")", ","]
                    ]

                    # Need at least 2 arguments to swap
                    if len(args) >= 2 and self.flip():
                        # Choose which pair to swap
                        swap_idx = self.rand.randint(0, len(args) - 2)
                        changes.append(
                            {
                                "args_node": args_node,
                                "args": args,
                                "swap_idx": swap_idx,
                            }
                        )

            for child in n.children:
                collect_function_calls(child)

        collect_function_calls(node)

        if not changes:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for change in reversed(changes):
            args_node = change["args_node"]
            args = change["args"]
            swap_idx = change["swap_idx"]

            # Get the two arguments to swap
            arg1 = args[swap_idx]
            arg2 = args[swap_idx + 1]

            arg1_text = _safe_decode(source_bytes[arg1.start_byte : arg1.end_byte])
            arg2_text = _safe_decode(source_bytes[arg2.start_byte : arg2.end_byte])

            # Reconstruct the arguments list with swapped args
            new_args_parts = []
            for i, arg in enumerate(args):
                if i == swap_idx:
                    new_args_parts.append(arg2_text)
                elif i == swap_idx + 1:
                    new_args_parts.append(arg1_text)
                else:
                    new_args_parts.append(
                        _safe_decode(source_bytes[arg.start_byte : arg.end_byte])
                    )

            new_args = "(" + ", ".join(new_args_parts) + ")"

            modified_source = (
                modified_source[: args_node.start_byte]
                + new_args.encode("utf-8")
                + modified_source[args_node.end_byte :]
            )

        return _safe_decode(modified_source, source_code)
