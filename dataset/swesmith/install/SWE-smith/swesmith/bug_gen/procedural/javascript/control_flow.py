"""
JavaScript control flow modifiers for procedural bug generation using tree-sitter.
"""

import tree_sitter_javascript as tsjs
from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.javascript.base import JavaScriptProceduralModifier
from swesmith.constants import BugRewrite, CodeEntity
from tree_sitter import Language, Parser

JS_LANGUAGE = Language(tsjs.language())


class ControlIfElseInvertModifier(JavaScriptProceduralModifier):
    """Invert if-else blocks by swapping their bodies"""

    explanation: str = CommonPMs.CONTROL_IF_ELSE_INVERT.explanation
    name: str = CommonPMs.CONTROL_IF_ELSE_INVERT.name
    conditions: list = CommonPMs.CONTROL_IF_ELSE_INVERT.conditions
    min_complexity: int = 5

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Swap if and else blocks."""
        # Parse the code
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        changed = False
        for _ in range(self.max_attempts):
            modified_code = self._invert_if_else_statements(
                code_entity.src_code, tree.root_node
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

    def _invert_if_else_statements(self, source_code: str, node) -> str:
        """Recursively find and invert if-else statements by swapping the bodies."""
        modifications = []
        source_bytes = source_code.encode("utf-8")

        def collect_if_statements(n):
            if n.type == "if_statement":
                # Parse the if statement structure
                # JavaScript if statement: if (condition) consequence [else alternative]
                condition = None
                consequence = None
                alternative = None

                for i, child in enumerate(n.children):
                    if child.type == "if":
                        continue  # Skip the "if" keyword
                    elif child.type == "parenthesized_expression":
                        condition = child
                    elif child.type == "statement_block" and consequence is None:
                        consequence = child  # First block is the if body
                    elif child.type == "else_clause":
                        # The else clause contains the alternative
                        for else_child in child.children:
                            if else_child.type == "statement_block":
                                alternative = else_child
                                break
                        break

                # Only modify if we have a complete if-else structure
                if condition and consequence and alternative and self.flip():
                    modifications.append(
                        {
                            "node": n,
                            "condition": condition,
                            "consequence": consequence,
                            "alternative": alternative,
                        }
                    )

            for child in n.children:
                collect_if_statements(child)

        collect_if_statements(node)

        if not modifications:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for mod in reversed(modifications):
            node = mod["node"]
            condition = mod["condition"]
            consequence = mod["consequence"]
            alternative = mod["alternative"]

            # Get the text of each part
            condition_text = source_bytes[
                condition.start_byte : condition.end_byte
            ].decode("utf-8")
            consequence_text = source_bytes[
                consequence.start_byte : consequence.end_byte
            ].decode("utf-8")
            alternative_text = source_bytes[
                alternative.start_byte : alternative.end_byte
            ].decode("utf-8")

            # Build the inverted if-else statement
            inverted = f"if {condition_text} {alternative_text} else {consequence_text}"

            # Replace the entire if statement
            modified_source = (
                modified_source[: node.start_byte]
                + inverted.encode("utf-8")
                + modified_source[node.end_byte :]
            )

        return modified_source.decode("utf-8")


class ControlShuffleLinesModifier(JavaScriptProceduralModifier):
    """Shuffle independent statements within a function body"""

    explanation: str = CommonPMs.CONTROL_SHUFFLE_LINES.explanation
    name: str = CommonPMs.CONTROL_SHUFFLE_LINES.name
    conditions: list = CommonPMs.CONTROL_SHUFFLE_LINES.conditions
    max_complexity: int = 10

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Shuffle statements within a function body."""
        # Parse the code
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find function body and shuffle statements
        modified_code = self._shuffle_statements(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _shuffle_statements(self, source_code: str, node) -> str:
        """Find function bodies and shuffle their statements."""
        shuffles = []
        source_bytes = source_code.encode("utf-8")

        def collect_function_bodies(n):
            # Look for function bodies (statement blocks inside functions)
            if n.type in [
                "function_declaration",
                "function_expression",
                "arrow_function",
                "method_definition",
            ]:
                # Find the body
                for child in n.children:
                    if child.type == "statement_block":
                        # Get all direct statement children
                        statements = [
                            c
                            for c in child.children
                            if c.type not in ["{", "}", "\n"]
                            and c.type.endswith("statement")
                        ]

                        # Only shuffle if we have multiple statements
                        if len(statements) >= 2 and self.flip():
                            shuffles.append(
                                {"block": child, "statements": statements.copy()}
                            )
                        return  # Don't recurse into nested functions

            for child in n.children:
                collect_function_bodies(child)

        collect_function_bodies(node)

        if not shuffles:
            return source_code

        # Work with bytes for modifications
        modified_source = source_bytes
        for shuffle_info in reversed(shuffles):
            block = shuffle_info["block"]
            statements = shuffle_info["statements"]

            # Shuffle the statements
            self.rand.shuffle(statements)

            # Build new block content
            new_statements = []
            for stmt in statements:
                stmt_text = source_bytes[stmt.start_byte : stmt.end_byte].decode(
                    "utf-8"
                )
                new_statements.append(stmt_text)

            # Find the opening and closing braces
            block_start = block.start_byte
            block_end = block.end_byte

            # Get indentation from first statement
            first_stmt_start = statements[0].start_byte
            indent_start = first_stmt_start
            while indent_start > block_start and source_bytes[indent_start - 1] in [
                ord(" "),
                ord("\t"),
            ]:
                indent_start -= 1

            indent = source_bytes[indent_start:first_stmt_start].decode("utf-8")

            # Build new block
            new_block = "{\n" + indent + f"\n{indent}".join(new_statements) + "\n}"

            # Replace the block
            modified_source = (
                modified_source[:block_start]
                + new_block.encode("utf-8")
                + modified_source[block_end:]
            )

        return modified_source.decode("utf-8")
