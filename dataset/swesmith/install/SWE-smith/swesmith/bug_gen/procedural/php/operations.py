"""
PHP operation modifiers for procedural bug generation using tree-sitter.
"""

import sys
from swesmith.bug_gen.procedural.php.base import PhpProceduralModifier, php_parse
from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.constants import CodeProperty, BugRewrite, CodeEntity


def _safe_decode(bytes_obj, fallback=""):
    """Safely decode bytes to UTF-8, handling potential encoding errors."""
    try:
        return bytes_obj.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"WARNING: UTF-8 decode error: {e}", file=sys.stderr)
        return fallback


class OperationChangeModifier(PhpProceduralModifier):
    """Change operators within similar groups (e.g., +/-, *//%, etc.)"""

    explanation: str = CommonPMs.OPERATION_CHANGE.explanation
    name: str = CommonPMs.OPERATION_CHANGE.name
    conditions: list = CommonPMs.OPERATION_CHANGE.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._change_operators(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _change_operators(self, source_code: str, node, offset: int) -> str:
        changes = []

        operator_swaps = {
            "+": ["-"],
            "-": ["+"],
            "*": ["/", "%", "**"],
            "/": ["*", "%", "**"],
            "%": ["*", "/", "**"],
            "**": ["*", "/", "%"],
            "&": ["|", "^"],
            "|": ["&", "^"],
            "^": ["&", "|"],
            "<<": [">>"],
            ">>": ["<<"],
            ".": ["+"],  # PHP string concatenation
        }

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                for child in n.children:
                    if child.type in operator_swaps:
                        operator = child.type
                        candidates = operator_swaps[operator]
                        if self.flip():
                            new_op = self.rand.choice(candidates)
                            changes.append(
                                {
                                    "start": child.start_byte - offset,
                                    "end": child.end_byte - offset,
                                    "new_op": new_op,
                                }
                            )
                        break

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        if not changes:
            return source_code

        modified_source = source_code.encode("utf-8")
        for change in reversed(changes):
            modified_source = (
                modified_source[: change["start"]]
                + change["new_op"].encode("utf-8")
                + modified_source[change["end"] :]
            )

        return _safe_decode(modified_source, source_code)


class OperationFlipOperatorModifier(PhpProceduralModifier):
    """Flip operators to their opposites (e.g., == to !=, < to >, etc.)"""

    explanation: str = "The operators in an expression are likely incorrect."
    name: str = "func_pm_op_flip"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_BINARY_OP]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._flip_operators(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _flip_operators(self, source_code: str, node, offset: int) -> str:
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
            "and": "or",
            "or": "and",
        }

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                for child in n.children:
                    if child.type in operator_flips:
                        if self.flip():
                            changes.append(
                                {
                                    "start": child.start_byte - offset,
                                    "end": child.end_byte - offset,
                                    "new_op": operator_flips[child.type],
                                }
                            )
                        break

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        if not changes:
            return source_code

        modified_source = source_code.encode("utf-8")
        for change in reversed(changes):
            modified_source = (
                modified_source[: change["start"]]
                + change["new_op"].encode("utf-8")
                + modified_source[change["end"] :]
            )

        return _safe_decode(modified_source, source_code)


class OperationSwapOperandsModifier(PhpProceduralModifier):
    """Swap operands in binary operations (e.g., a + b becomes b + a)"""

    explanation: str = CommonPMs.OPERATION_SWAP_OPERANDS.explanation
    name: str = CommonPMs.OPERATION_SWAP_OPERANDS.name
    conditions: list = CommonPMs.OPERATION_SWAP_OPERANDS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._swap_operands(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_operands(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_binary_ops(n):
            if n.type == "binary_expression" and len(n.children) >= 3:
                left = n.children[0]
                operator_node = n.children[1]
                right = n.children[2]

                if self.flip():
                    changes.append(
                        {
                            "node_start": n.start_byte - offset,
                            "node_end": n.end_byte - offset,
                            "left_start": left.start_byte - offset,
                            "left_end": left.end_byte - offset,
                            "right_start": right.start_byte - offset,
                            "right_end": right.end_byte - offset,
                            "operator": operator_node.type,
                        }
                    )

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            left_text = _safe_decode(
                source_bytes[change["left_start"] : change["left_end"]]
            )
            right_text = _safe_decode(
                source_bytes[change["right_start"] : change["right_end"]]
            )

            swapped = f"{right_text} {change['operator']} {left_text}"

            modified_source = (
                modified_source[: change["node_start"]]
                + swapped.encode("utf-8")
                + modified_source[change["node_end"] :]
            )

        return _safe_decode(modified_source, source_code)


class OperationChangeConstantsModifier(PhpProceduralModifier):
    """Change numeric constants to introduce off-by-one errors"""

    explanation: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.explanation
    name: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.name
    conditions: list = CommonPMs.OPERATION_CHANGE_CONSTANTS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._change_constants(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    @staticmethod
    def _parse_php_int(value_text: str) -> int:
        """Parse a PHP integer literal, returning its integer value."""
        lower = value_text.lower()
        for prefix, base in [("0x", 16), ("0b", 2), ("0o", 8)]:
            if lower.startswith(prefix):
                return int(value_text, base)
        if len(value_text) > 1 and value_text.startswith("0") and value_text.isdigit():
            return int(value_text, 8)
        return int(value_text)

    def _change_constants(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_numbers(n):
            if n.type == "integer" and self.flip():
                try:
                    start = n.start_byte - offset
                    end = n.end_byte - offset
                    value_text = _safe_decode(source_bytes[start:end])
                    value = self._parse_php_int(value_text)
                    new_value = value + self.rand.choice([-1, 1, -2, 2])
                    changes.append(
                        {"start": start, "end": end, "new_value": str(new_value)}
                    )
                except ValueError:
                    pass
            elif n.type == "float" and self.flip():
                try:
                    start = n.start_byte - offset
                    end = n.end_byte - offset
                    value_text = _safe_decode(source_bytes[start:end])
                    value = float(value_text)
                    new_value = value + self.rand.choice([-1.0, 1.0, -0.5, 0.5])
                    changes.append(
                        {"start": start, "end": end, "new_value": str(new_value)}
                    )
                except ValueError:
                    pass

            for child in n.children:
                collect_numbers(child)

        collect_numbers(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            modified_source = (
                modified_source[: change["start"]]
                + change["new_value"].encode("utf-8")
                + modified_source[change["end"] :]
            )

        return _safe_decode(modified_source, source_code)


class OperationBreakChainsModifier(PhpProceduralModifier):
    """Break chained operations by removing parts of the chain"""

    explanation: str = CommonPMs.OPERATION_BREAK_CHAINS.explanation
    name: str = CommonPMs.OPERATION_BREAK_CHAINS.name
    conditions: list = CommonPMs.OPERATION_BREAK_CHAINS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._break_chains(code_entity.src_code, tree.root_node, offset)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _break_chains(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_chains(n):
            matched = False
            if n.type == "binary_expression" and len(n.children) >= 3:
                left = n.children[0]
                operator = n.children[1]
                right = n.children[2]

                if left.type == "binary_expression" and self.flip():
                    if len(left.children) >= 3:
                        left_right = left.children[2]
                        lr_text = _safe_decode(
                            source_bytes[
                                left_right.start_byte - offset : left_right.end_byte
                                - offset
                            ]
                        )
                        op_text = _safe_decode(
                            source_bytes[
                                operator.start_byte - offset : operator.end_byte
                                - offset
                            ]
                        )
                        r_text = _safe_decode(
                            source_bytes[
                                right.start_byte - offset : right.end_byte - offset
                            ]
                        )
                        changes.append(
                            {
                                "start": n.start_byte - offset,
                                "end": n.end_byte - offset,
                                "replacement": f"{lr_text} {op_text} {r_text}",
                            }
                        )
                        matched = True

                elif right.type == "binary_expression" and self.flip():
                    if len(right.children) >= 3:
                        right_left = right.children[0]
                        l_text = _safe_decode(
                            source_bytes[
                                left.start_byte - offset : left.end_byte - offset
                            ]
                        )
                        op_text = _safe_decode(
                            source_bytes[
                                operator.start_byte - offset : operator.end_byte
                                - offset
                            ]
                        )
                        rl_text = _safe_decode(
                            source_bytes[
                                right_left.start_byte - offset : right_left.end_byte
                                - offset
                            ]
                        )
                        changes.append(
                            {
                                "start": n.start_byte - offset,
                                "end": n.end_byte - offset,
                                "replacement": f"{l_text} {op_text} {rl_text}",
                            }
                        )
                        matched = True

            # Don't recurse into children of a matched node to avoid
            # overlapping changes that would corrupt byte offsets
            if not matched:
                for child in n.children:
                    collect_chains(child)

        collect_chains(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            modified_source = (
                modified_source[: change["start"]]
                + change["replacement"].encode("utf-8")
                + modified_source[change["end"] :]
            )

        return _safe_decode(modified_source, source_code)


class AugmentedAssignmentSwapModifier(PhpProceduralModifier):
    """Swap augmented assignment operators (+=, -=, *=, /=, etc.) and update expressions (++, --)"""

    explanation: str = (
        "The augmented assignment or update operator is likely incorrect."
    )
    name: str = "func_pm_aug_assign_swap"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_ASSIGNMENT]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._swap_augmented_assignments(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_augmented_assignments(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        aug_assign_swaps = {
            "+=": "-=",
            "-=": "+=",
            "*=": "/=",
            "/=": "*=",
            "%=": "/=",
            "&=": "|=",
            "|=": "&=",
            "^=": "&=",
            "<<=": ">>=",
            ">>=": "<<=",
            ".=": "+=",  # PHP string concat assignment
            "**=": "*=",
            "??=": ".=",
        }

        update_swaps = {
            "++": "--",
            "--": "++",
        }

        def collect_augmented_assignments(n):
            if n.type == "augmented_assignment_expression":
                for child in n.children:
                    op_text = source_bytes[
                        child.start_byte - offset : child.end_byte - offset
                    ].decode("utf-8")
                    if op_text in aug_assign_swaps and self.flip():
                        changes.append(
                            {
                                "start": child.start_byte - offset,
                                "end": child.end_byte - offset,
                                "new_op": aug_assign_swaps[op_text],
                            }
                        )
                        break

            elif n.type == "update_expression":
                for child in n.children:
                    op_text = source_bytes[
                        child.start_byte - offset : child.end_byte - offset
                    ].decode("utf-8")
                    if op_text in update_swaps and self.flip():
                        changes.append(
                            {
                                "start": child.start_byte - offset,
                                "end": child.end_byte - offset,
                                "new_op": update_swaps[op_text],
                            }
                        )
                        break

            for child in n.children:
                collect_augmented_assignments(child)

        collect_augmented_assignments(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            modified_source = (
                modified_source[: change["start"]]
                + change["new_op"].encode("utf-8")
                + modified_source[change["end"] :]
            )

        return _safe_decode(modified_source, source_code)


class TernaryOperatorSwapModifier(PhpProceduralModifier):
    """Modify ternary operators (condition ? consequent : alternative)"""

    explanation: str = "The ternary operator branches may be swapped or the condition may be incorrect."
    name: str = "func_pm_ternary_swap"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_TERNARY]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._modify_ternary(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _modify_ternary(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_ternary_ops(n):
            # PHP uses "conditional_expression" instead of "ternary_expression"
            if n.type == "conditional_expression" and len(n.children) >= 5:
                content_children = [c for c in n.children if c.type not in ["?", ":"]]
                if len(content_children) >= 3:
                    condition = content_children[0]
                    consequent = content_children[1]
                    alternative = content_children[2]

                    if self.flip():
                        mod_type = self.rand.choice(
                            ["swap_branches", "negate_condition"]
                        )
                        changes.append(
                            {
                                "start": n.start_byte - offset,
                                "end": n.end_byte - offset,
                                "cond_start": condition.start_byte - offset,
                                "cond_end": condition.end_byte - offset,
                                "cons_start": consequent.start_byte - offset,
                                "cons_end": consequent.end_byte - offset,
                                "alt_start": alternative.start_byte - offset,
                                "alt_end": alternative.end_byte - offset,
                                "mod_type": mod_type,
                            }
                        )

            for child in n.children:
                collect_ternary_ops(child)

        collect_ternary_ops(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            cond_text = _safe_decode(
                source_bytes[change["cond_start"] : change["cond_end"]]
            )
            cons_text = _safe_decode(
                source_bytes[change["cons_start"] : change["cons_end"]]
            )
            alt_text = _safe_decode(
                source_bytes[change["alt_start"] : change["alt_end"]]
            )

            if change["mod_type"] == "swap_branches":
                new_ternary = f"{cond_text} ? {alt_text} : {cons_text}"
            else:
                new_ternary = f"!({cond_text}) ? {cons_text} : {alt_text}"

            modified_source = (
                modified_source[: change["start"]]
                + new_ternary.encode("utf-8")
                + modified_source[change["end"] :]
            )

        return _safe_decode(modified_source, source_code)


class FunctionArgumentSwapModifier(PhpProceduralModifier):
    """Swap adjacent arguments in function calls."""

    explanation: str = "The function arguments may be in the wrong order."
    name: str = "func_pm_arg_swap"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_FUNCTION_CALL]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._swap_arguments(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_arguments(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_function_calls(n):
            if n.type in [
                "function_call_expression",
                "member_call_expression",
                "scoped_call_expression",
                "nullsafe_member_call_expression",
            ]:
                args_node = None
                for child in n.children:
                    if child.type == "arguments":
                        args_node = child
                        break

                if args_node:
                    args = [
                        c for c in args_node.children if c.type not in ["(", ")", ","]
                    ]

                    if len(args) >= 2 and self.flip():
                        swap_idx = self.rand.randint(0, len(args) - 2)
                        changes.append(
                            {
                                "args_start": args_node.start_byte - offset,
                                "args_end": args_node.end_byte - offset,
                                "args": [
                                    (a.start_byte - offset, a.end_byte - offset)
                                    for a in args
                                ],
                                "swap_idx": swap_idx,
                            }
                        )

            for child in n.children:
                collect_function_calls(child)

        collect_function_calls(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            args = change["args"]
            swap_idx = change["swap_idx"]

            new_args_parts = []
            for i, (start, end) in enumerate(args):
                if i == swap_idx:
                    s2, e2 = args[swap_idx + 1]
                    new_args_parts.append(_safe_decode(source_bytes[s2:e2]))
                elif i == swap_idx + 1:
                    s1, e1 = args[swap_idx]
                    new_args_parts.append(_safe_decode(source_bytes[s1:e1]))
                else:
                    new_args_parts.append(_safe_decode(source_bytes[start:end]))

            new_args = "(" + ", ".join(new_args_parts) + ")"

            modified_source = (
                modified_source[: change["args_start"]]
                + new_args.encode("utf-8")
                + modified_source[change["args_end"] :]
            )

        return _safe_decode(modified_source, source_code)
