"""
PHP remove modifiers for procedural bug generation using tree-sitter.
"""

from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.php.base import PhpProceduralModifier, php_parse
from swesmith.constants import BugRewrite, CodeEntity, CodeProperty


class RemoveLoopModifier(PhpProceduralModifier):
    """Remove loop statements (for, foreach, while, do-while)"""

    explanation: str = CommonPMs.REMOVE_LOOP.explanation
    name: str = CommonPMs.REMOVE_LOOP.name
    conditions: list = CommonPMs.REMOVE_LOOP.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._remove_loops(code_entity.src_code, tree.root_node, offset)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_loops(self, source_code: str, node, offset: int) -> str:
        removals = []

        def collect_loops(n):
            if n.type in [
                "for_statement",
                "foreach_statement",
                "while_statement",
                "do_statement",
            ]:
                if self.flip():
                    removals.append((n.start_byte - offset, n.end_byte - offset))
            for child in n.children:
                collect_loops(child)

        collect_loops(node)

        if not removals:
            return source_code

        modified_source = source_code
        for start, end in reversed(removals):
            modified_source = modified_source[:start] + modified_source[end:]

        return modified_source


class RemoveConditionalModifier(PhpProceduralModifier):
    """Remove conditional statements (if statements)"""

    explanation: str = CommonPMs.REMOVE_CONDITIONAL.explanation
    name: str = CommonPMs.REMOVE_CONDITIONAL.name
    conditions: list = CommonPMs.REMOVE_CONDITIONAL.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._remove_conditionals(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_conditionals(self, source_code: str, node, offset: int) -> str:
        removals = []

        def collect_conditionals(n):
            if n.type == "if_statement":
                if self.flip():
                    removals.append((n.start_byte - offset, n.end_byte - offset))
            for child in n.children:
                collect_conditionals(child)

        collect_conditionals(node)

        if not removals:
            return source_code

        modified_source = source_code
        for start, end in reversed(removals):
            modified_source = modified_source[:start] + modified_source[end:]

        return modified_source


class RemoveAssignmentModifier(PhpProceduralModifier):
    """Remove assignment statements"""

    explanation: str = CommonPMs.REMOVE_ASSIGNMENT.explanation
    name: str = CommonPMs.REMOVE_ASSIGNMENT.name
    conditions: list = CommonPMs.REMOVE_ASSIGNMENT.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._remove_assignments(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_assignments(self, source_code: str, node, offset: int) -> str:
        removals = []

        def collect_assignments(n):
            matched = False
            if n.type in [
                "assignment_expression",
                "augmented_assignment_expression",
                "reference_assignment_expression",
            ]:
                if self.flip():
                    matched = True
                    if n.parent and n.parent.type == "expression_statement":
                        removals.append(
                            (n.parent.start_byte - offset, n.parent.end_byte - offset)
                        )
                    else:
                        removals.append((n.start_byte - offset, n.end_byte - offset))
            if not matched:
                for child in n.children:
                    collect_assignments(child)

        collect_assignments(node)

        if not removals:
            return source_code

        modified_source = source_code
        for start, end in reversed(removals):
            while end < len(modified_source) and modified_source[end] in [
                " ",
                "\t",
                ";",
            ]:
                end += 1
            if end < len(modified_source) and modified_source[end] == "\n":
                end += 1

            modified_source = modified_source[:start] + modified_source[end:]

        return modified_source


class RemoveTernaryModifier(PhpProceduralModifier):
    """Remove ternary expressions by replacing with just one branch."""

    explanation: str = "A ternary conditional expression may be missing - only one branch is being used."
    name: str = "func_pm_remove_ternary"
    conditions: list = [CodeProperty.IS_FUNCTION, CodeProperty.HAS_TERNARY]

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._remove_ternary(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_ternary(self, source_code: str, node, offset: int) -> str:
        changes = []
        source_bytes = source_code.encode("utf-8")

        def collect_ternary_ops(n):
            matched = False
            if n.type == "conditional_expression":
                content_children = [c for c in n.children if c.type not in ["?", ":"]]
                if len(content_children) >= 3:
                    # Standard ternary: $cond ? $then : $else
                    consequent = content_children[1]
                    alternative = content_children[2]

                    if self.flip():
                        matched = True
                        keep_consequent = self.rand.choice([True, False])
                        replacement = consequent if keep_consequent else alternative
                        changes.append(
                            {
                                "start": n.start_byte - offset,
                                "end": n.end_byte - offset,
                                "rep_start": replacement.start_byte - offset,
                                "rep_end": replacement.end_byte - offset,
                            }
                        )
                elif len(content_children) == 2:
                    # Elvis operator: $cond ?: $else
                    condition = content_children[0]
                    alternative = content_children[1]

                    if self.flip():
                        matched = True
                        keep_condition = self.rand.choice([True, False])
                        replacement = condition if keep_condition else alternative
                        changes.append(
                            {
                                "start": n.start_byte - offset,
                                "end": n.end_byte - offset,
                                "rep_start": replacement.start_byte - offset,
                                "rep_end": replacement.end_byte - offset,
                            }
                        )

            if not matched:
                for child in n.children:
                    collect_ternary_ops(child)

        collect_ternary_ops(node)

        if not changes:
            return source_code

        modified_source = source_bytes
        for change in reversed(changes):
            replacement_text = source_bytes[change["rep_start"] : change["rep_end"]]

            modified_source = (
                modified_source[: change["start"]]
                + replacement_text
                + modified_source[change["end"] :]
            )

        return modified_source.decode("utf-8")
