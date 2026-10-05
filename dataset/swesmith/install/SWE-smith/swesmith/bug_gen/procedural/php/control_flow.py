"""
PHP control flow modifiers for procedural bug generation using tree-sitter.
"""

from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.php.base import PhpProceduralModifier, php_parse
from swesmith.constants import BugRewrite, CodeEntity


class ControlIfElseInvertModifier(PhpProceduralModifier):
    """Invert if-else blocks by swapping their bodies"""

    explanation: str = CommonPMs.CONTROL_IF_ELSE_INVERT.explanation
    name: str = CommonPMs.CONTROL_IF_ELSE_INVERT.name
    conditions: list = CommonPMs.CONTROL_IF_ELSE_INVERT.conditions
    min_complexity: int = 5

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        changed = False
        for _ in range(self.max_attempts):
            modified_code = self._invert_if_else_statements(
                code_entity.src_code, tree.root_node, offset
            )

            if modified_code != code_entity.src_code:
                changed = True
                break

        if not changed:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _invert_if_else_statements(self, source_code: str, node, offset: int) -> str:
        modifications = []
        source_bytes = source_code.encode("utf-8")

        def collect_if_statements(n):
            if n.type == "if_statement":
                condition = None
                consequence = None
                alternative = None
                has_elseif = False

                for child in n.children:
                    if child.type == "if":
                        continue
                    elif child.type == "parenthesized_expression":
                        condition = child
                    elif child.type == "compound_statement" and consequence is None:
                        consequence = child
                    elif child.type == "else_if_clause":
                        has_elseif = True
                    elif child.type == "else_clause":
                        for else_child in child.children:
                            if else_child.type == "compound_statement":
                                alternative = else_child
                                break
                        break

                # Skip if/elseif/else chains to avoid dropping elseif branches
                if (
                    condition
                    and consequence
                    and alternative
                    and not has_elseif
                    and self.flip()
                ):
                    modifications.append(
                        {
                            "node_start": n.start_byte - offset,
                            "node_end": n.end_byte - offset,
                            "cond_start": condition.start_byte - offset,
                            "cond_end": condition.end_byte - offset,
                            "cons_start": consequence.start_byte - offset,
                            "cons_end": consequence.end_byte - offset,
                            "alt_start": alternative.start_byte - offset,
                            "alt_end": alternative.end_byte - offset,
                        }
                    )

            for child in n.children:
                collect_if_statements(child)

        collect_if_statements(node)

        if not modifications:
            return source_code

        modified_source = source_bytes
        for mod in reversed(modifications):
            condition_text = source_bytes[mod["cond_start"] : mod["cond_end"]].decode(
                "utf-8"
            )
            consequence_text = source_bytes[mod["cons_start"] : mod["cons_end"]].decode(
                "utf-8"
            )
            alternative_text = source_bytes[mod["alt_start"] : mod["alt_end"]].decode(
                "utf-8"
            )

            inverted = f"if {condition_text} {alternative_text} else {consequence_text}"

            modified_source = (
                modified_source[: mod["node_start"]]
                + inverted.encode("utf-8")
                + modified_source[mod["node_end"] :]
            )

        return modified_source.decode("utf-8")


class ControlShuffleLinesModifier(PhpProceduralModifier):
    """Shuffle independent statements within a function body"""

    explanation: str = CommonPMs.CONTROL_SHUFFLE_LINES.explanation
    name: str = CommonPMs.CONTROL_SHUFFLE_LINES.name
    conditions: list = CommonPMs.CONTROL_SHUFFLE_LINES.conditions
    max_complexity: int = 10

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        tree, offset = php_parse(code_entity.src_code)

        modified_code = self._shuffle_statements(
            code_entity.src_code, tree.root_node, offset
        )

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _shuffle_statements(self, source_code: str, node, offset: int) -> str:
        shuffles = []
        source_bytes = source_code.encode("utf-8")

        def collect_function_bodies(n):
            if n.type in [
                "function_definition",
                "method_declaration",
            ]:
                for child in n.children:
                    if child.type == "compound_statement":
                        statements = [
                            c
                            for c in child.children
                            if c.type not in ["{", "}", "\n"]
                            and c.type.endswith("statement")
                        ]

                        if len(statements) >= 2 and self.flip():
                            shuffles.append(
                                {
                                    "block_start": child.start_byte - offset,
                                    "block_end": child.end_byte - offset,
                                    "statements": [
                                        (s.start_byte - offset, s.end_byte - offset)
                                        for s in statements
                                    ],
                                }
                            )
                        return

            for child in n.children:
                collect_function_bodies(child)

        collect_function_bodies(node)

        if not shuffles:
            return source_code

        modified_source = source_bytes
        for shuffle_info in reversed(shuffles):
            stmts = shuffle_info["statements"]

            stmt_texts = [source_bytes[s:e].decode("utf-8") for s, e in stmts]
            self.rand.shuffle(stmt_texts)

            block_start = shuffle_info["block_start"]
            block_end = shuffle_info["block_end"]

            first_stmt_start = stmts[0][0]
            indent_start = first_stmt_start
            while indent_start > block_start and source_bytes[indent_start - 1] in [
                ord(" "),
                ord("\t"),
            ]:
                indent_start -= 1

            indent = source_bytes[indent_start:first_stmt_start].decode("utf-8")

            new_block = "{\n" + indent + f"\n{indent}".join(stmt_texts) + "\n}"

            modified_source = (
                modified_source[:block_start]
                + new_block.encode("utf-8")
                + modified_source[block_end:]
            )

        return modified_source.decode("utf-8")
