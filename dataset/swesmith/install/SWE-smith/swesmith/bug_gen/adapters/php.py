import re

from swesmith.constants import TODO_REWRITE, CodeEntity, CodeProperty
from tree_sitter import Language, Parser
import tree_sitter_php as tsphp
from swesmith.bug_gen.adapters.utils import build_entity

PHP_LANGUAGE = Language(tsphp.language_php())


class PhpEntity(CodeEntity):
    @property
    def name(self) -> str:
        if self.node.type == "function_definition":
            for child in self.node.children:
                if child.type == "name":
                    return child.text.decode("utf-8")
        elif self.node.type == "method_declaration":
            for child in self.node.children:
                if child.type == "name":
                    func_name = child.text.decode("utf-8")
                    # Find the class this method belongs to
                    class_node = self._find_parent_class()
                    if class_node:
                        class_name = self._get_class_name(class_node)
                        return f"{class_name}::{func_name}" if class_name else func_name
                    return func_name
        elif self.node.type == "class_declaration":
            for child in self.node.children:
                if child.type == "name":
                    return child.text.decode("utf-8")
        return "unknown"

    def _find_parent_class(self):
        """Find the parent class node for a method."""
        current = self.node.parent
        while current:
            if current.type == "class_declaration":
                return current
            current = current.parent
        return None

    def _get_class_name(self, class_node):
        """Extract class name from a class node."""
        for child in class_node.children:
            if child.type == "name":
                return child.text.decode("utf-8")
        return None

    def _analyze_properties(self):
        """Analyze PHP code properties for procedural modifiers."""
        node = self.node

        # Core entity types
        if node.type in ["function_definition", "method_declaration"]:
            self._tags.add(CodeProperty.IS_FUNCTION)
        elif node.type == "class_declaration":
            self._tags.add(CodeProperty.IS_CLASS)

        self._walk_for_properties(node)

    def _walk_for_properties(self, n):
        """Walk the AST and analyze properties."""
        self._check_control_flow(n)
        self._check_operations(n)
        self._check_expressions(n)
        for child in n.children:
            self._walk_for_properties(child)

    def _check_control_flow(self, n):
        """Check for control flow patterns."""
        if n.type in [
            "for_statement",
            "foreach_statement",
            "while_statement",
            "do_statement",
        ]:
            self._tags.add(CodeProperty.HAS_LOOP)
        if n.type == "if_statement":
            self._tags.add(CodeProperty.HAS_IF)
            if any(child.type == "else_clause" for child in n.children):
                self._tags.add(CodeProperty.HAS_IF_ELSE)
        if n.type == "switch_statement":
            self._tags.add(CodeProperty.HAS_SWITCH)
        if n.type in ["try_statement", "catch_clause", "throw_expression"]:
            self._tags.add(CodeProperty.HAS_EXCEPTION)

    def _check_operations(self, n):
        """Check for various operations."""
        if n.type in ["subscript_expression", "member_access_expression"]:
            self._tags.add(CodeProperty.HAS_LIST_INDEXING)
        if n.type in [
            "function_call_expression",
            "member_call_expression",
            "scoped_call_expression",
        ]:
            self._tags.add(CodeProperty.HAS_FUNCTION_CALL)
        if n.type == "return_statement":
            self._tags.add(CodeProperty.HAS_RETURN)
        if n.type in ["namespace_use_declaration"]:
            self._tags.add(CodeProperty.HAS_IMPORT)
        if n.type in ["assignment_expression", "augmented_assignment_expression"]:
            self._tags.add(CodeProperty.HAS_ASSIGNMENT)
        if n.type == "arrow_function":
            self._tags.add(CodeProperty.HAS_LAMBDA)
        if n.type == "anonymous_function":
            self._tags.add(CodeProperty.HAS_LAMBDA)
        if n.type in ["binary_expression", "unary_op_expression", "update_expression"]:
            self._tags.add(CodeProperty.HAS_ARITHMETIC)
        if n.type in ["try_statement"]:
            self._tags.add(CodeProperty.HAS_WRAPPER)
        if n.type == "class_declaration" and any(
            child.type == "base_clause" for child in n.children
        ):
            self._tags.add(CodeProperty.HAS_PARENT)
        if n.type in ["unary_op_expression", "update_expression"]:
            self._tags.add(CodeProperty.HAS_UNARY_OP)
        if n.type == "conditional_expression":
            self._tags.add(CodeProperty.HAS_TERNARY)

    def _check_expressions(self, n):
        """Check binary expression patterns."""
        if n.type == "binary_expression":
            self._tags.add(CodeProperty.HAS_BINARY_OP)
            for child in n.children:
                if hasattr(child, "text"):
                    text = child.text.decode("utf-8")
                    if text in ["&&", "||", "and", "or"]:
                        self._tags.add(CodeProperty.HAS_BOOL_OP)
                    elif text in ["<", ">", "<=", ">="]:
                        self._tags.add(CodeProperty.HAS_OFF_BY_ONE)

    @property
    def complexity(self) -> int:
        """Calculate cyclomatic complexity for PHP code."""

        def walk(node):
            score = 0
            if node.type in [
                "if_statement",
                "else_clause",
                "for_statement",
                "foreach_statement",
                "while_statement",
                "do_statement",
                "switch_statement",
                "case_statement",
                "catch_clause",
                "conditional_expression",
            ]:
                score += 1
            if node.type == "binary_expression":
                for child in node.children:
                    if hasattr(child, "text") and child.text.decode("utf-8") in [
                        "&&",
                        "||",
                        "and",
                        "or",
                    ]:
                        score += 1
            for child in node.children:
                score += walk(child)
            return score

        return 1 + walk(self.node)

    @property
    def signature(self) -> str:
        # Find the opening brace '{' and remove everything after it
        return self.src_code.split("{", 1)[0].strip()

    @property
    def stub(self) -> str:
        # Find the opening brace '{' and remove everything after it
        match = re.search(r"\{", self.src_code)
        if match:
            body_start = match.start()
            return (
                self.src_code[:body_start].rstrip() + " {\n\t// " + TODO_REWRITE + "\n}"
            )
        else:
            # If no body found, return the original code
            return self.src_code


def get_entities_from_file_php(
    entities: list[PhpEntity],
    file_path: str,
    max_entities: int = -1,
) -> list[PhpEntity]:
    """
    Parse a .php file and return up to max_entities top-level functions, methods, and classes.
    If max_entities < 0, collects them all.
    """
    parser = Parser(PHP_LANGUAGE)

    try:
        file_content = open(file_path, "r", encoding="utf8").read()
        tree = parser.parse(bytes(file_content, "utf8"))
        root = tree.root_node
        lines = file_content.splitlines()

        def walk(node):
            # stop if we've hit the limit
            if 0 <= max_entities == len(entities):
                return

            if node.type in [
                "function_definition",
                "method_declaration",
                "class_declaration",
            ]:
                entities.append(build_entity(node, lines, file_path, PhpEntity))
                if 0 <= max_entities == len(entities):
                    return

            for child in node.children:
                walk(child)

        walk(root)
        return entities
    except Exception:
        return entities
