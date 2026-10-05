"""
TypeScript adapter for entity extraction.
"""

import warnings
from pathlib import Path

import tree_sitter_typescript as tsts

from swesmith.constants import CodeEntity, CodeProperty, TODO_REWRITE
from swesmith.bug_gen.adapters.utils import build_entity
from tree_sitter import Language, Parser

TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())


class TypeScriptEntity(CodeEntity):
    def _analyze_properties(self):
        node = self.node

        if node.type in [
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "generator_function_declaration",
        ]:
            self._tags.add(CodeProperty.IS_FUNCTION)
        elif node.type in ["class_declaration", "class"]:
            self._tags.add(CodeProperty.IS_CLASS)

        self._walk_for_properties(node)

    def _walk_for_properties(self, n):
        self._check_control_flow(n)
        self._check_operations(n)
        self._check_binary_expressions(n)
        for child in n.children:
            self._walk_for_properties(child)

    def _check_control_flow(self, n):
        if n.type in [
            "for_statement",
            "for_in_statement",
            "for_of_statement",
            "while_statement",
            "do_statement",
        ]:
            self._tags.add(CodeProperty.HAS_LOOP)
        if n.type == "if_statement":
            self._tags.add(CodeProperty.HAS_IF)
            if any(child.type == "else_clause" for child in n.children):
                self._tags.add(CodeProperty.HAS_IF_ELSE)
        if n.type in ["try_statement", "catch_clause", "throw_statement"]:
            self._tags.add(CodeProperty.HAS_EXCEPTION)

    def _check_operations(self, n):
        if n.type in ["subscript_expression", "member_expression"]:
            self._tags.add(CodeProperty.HAS_LIST_INDEXING)
        if n.type == "call_expression":
            self._tags.add(CodeProperty.HAS_FUNCTION_CALL)
        if n.type == "return_statement":
            self._tags.add(CodeProperty.HAS_RETURN)
        if n.type in ["import_statement", "import_clause"]:
            self._tags.add(CodeProperty.HAS_IMPORT)
        if n.type in ["assignment_expression", "variable_declaration"]:
            self._tags.add(CodeProperty.HAS_ASSIGNMENT)
        if n.type == "arrow_function":
            self._tags.add(CodeProperty.HAS_LAMBDA)
        if n.type in ["binary_expression", "unary_expression", "update_expression"]:
            self._tags.add(CodeProperty.HAS_ARITHMETIC)
        if n.type == "decorator":
            self._tags.add(CodeProperty.HAS_DECORATOR)
        if n.type in ["try_statement", "with_statement"]:
            self._tags.add(CodeProperty.HAS_WRAPPER)
        if n.type == "class_declaration" and any(
            child.type == "class_heritage" for child in n.children
        ):
            self._tags.add(CodeProperty.HAS_PARENT)
        if n.type in ["unary_expression", "update_expression"]:
            self._tags.add(CodeProperty.HAS_UNARY_OP)
        if n.type == "ternary_expression":
            self._tags.add(CodeProperty.HAS_TERNARY)

    def _check_binary_expressions(self, n):
        if n.type == "binary_expression":
            self._tags.add(CodeProperty.HAS_BINARY_OP)
            for child in n.children:
                if hasattr(child, "text"):
                    text = child.text.decode("utf-8")
                    if text in ["&&", "||"]:
                        self._tags.add(CodeProperty.HAS_BOOL_OP)
                    if text in ["<", ">", "<=", ">="]:
                        self._tags.add(CodeProperty.HAS_OFF_BY_ONE)

    @property
    def name(self) -> str:
        if self.node.type in ["function_declaration", "generator_function_declaration"]:
            return self._find_child_text("identifier")
        if self.node.type == "method_definition":
            return self._find_child_text("property_identifier")
        if self.node.type == "class_declaration":
            return self._find_child_text("type_identifier") or self._find_child_text(
                "identifier"
            )
        if self.node.type == "variable_declarator":
            return self._find_child_text("identifier")
        if self.node.type == "assignment_expression":
            return self._find_child_text("identifier")
        return ""

    def _find_child_text(self, child_type: str) -> str:
        for child in self.node.children:
            if child.type == child_type:
                return child.text.decode("utf-8")
        return ""

    @property
    def signature(self) -> str:
        # Use node.text (raw bytes) instead of src_code (dedented string) for accurate byte offsets
        node_text = self.node.text.decode("utf-8")

        for child in self.node.children:
            if child.type in ["statement_block", "class_body"]:
                body_start_byte = child.start_byte - self.node.start_byte
                signature = node_text[:body_start_byte].strip()
                # Remove trailing { if present
                if signature.endswith(" {"):
                    signature = signature[:-2].strip()
                return signature

        # Arrow functions with expression body
        if self.node.type == "arrow_function" and "=>" in node_text:
            return node_text.split("=>")[0].strip() + " =>"

        # Function expressions: var myFunc = function(x, y) { ... }
        if self.node.type == "variable_declarator":
            first_line = node_text.split("\n")[0]
            if " = function" in first_line:
                brace_pos = first_line.find(" {")
                if brace_pos != -1:
                    return first_line[:brace_pos].strip()
                return first_line.rstrip(";").strip()

        return node_text.split("\n")[0].strip()

    @property
    def stub(self) -> str:
        sig = self.signature
        if self.node.type == "arrow_function":
            if "=>" in sig:
                return f"{sig} {{\n\t// {TODO_REWRITE}\n}}"
            return f"{sig} => {{\n\t// {TODO_REWRITE}\n}}"
        return f"{sig} {{\n\t// {TODO_REWRITE}\n}}"

    @property
    def complexity(self) -> int:
        def walk(node):
            score = 0
            if node.type in [
                "if_statement",
                "else_clause",
                "for_statement",
                "for_in_statement",
                "for_of_statement",
                "while_statement",
                "do_statement",
                "switch_statement",
                "case_clause",
                "catch_clause",
                "conditional_expression",
            ]:
                score += 1
            if node.type == "binary_expression":
                for child in node.children:
                    if hasattr(child, "text") and child.text.decode("utf-8") in [
                        "&&",
                        "||",
                    ]:
                        score += 1
            for child in node.children:
                score += walk(child)
            return score

        return 1 + walk(self.node)


def get_entities_from_file_ts(
    entities: list[TypeScriptEntity],
    file_path: str,
    max_entities: int = -1,
) -> list[TypeScriptEntity]:
    file_ext = Path(file_path).suffix
    language = TSX_LANGUAGE if file_ext == ".tsx" else TS_LANGUAGE
    parser = Parser(language)

    try:
        file_content = open(file_path, "r", encoding="utf8").read()
    except UnicodeDecodeError:
        warnings.warn(f"Could not decode file {file_path}", stacklevel=2)
        return entities

    tree = parser.parse(bytes(file_content, "utf8"))
    lines = file_content.splitlines()

    _walk_and_collect(tree.root_node, entities, lines, str(file_path), max_entities)
    return entities


def _walk_and_collect(node, entities, lines, file_path, max_entities):
    if 0 <= max_entities == len(entities):
        return

    if node.type == "ERROR":
        warnings.warn(f"Error encountered parsing {file_path}", stacklevel=2)
        return

    if node.type in [
        "function_declaration",
        "method_definition",
        "class_declaration",
        "generator_function_declaration",
    ]:
        entities.append(
            build_entity(
                node, lines, file_path, TypeScriptEntity, default_indent_size=2
            )
        )
        if 0 <= max_entities == len(entities):
            return

    elif node.type == "variable_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                for grandchild in child.children:
                    if grandchild.type in ["function_expression", "arrow_function"]:
                        entities.append(
                            build_entity(
                                child,
                                lines,
                                file_path,
                                TypeScriptEntity,
                                default_indent_size=2,
                            )
                        )
                        if 0 <= max_entities == len(entities):
                            return

    elif node.type == "assignment_expression":
        for child in node.children:
            if child.type in ["function_expression", "arrow_function"]:
                entities.append(
                    build_entity(
                        node, lines, file_path, TypeScriptEntity, default_indent_size=2
                    )
                )
                if 0 <= max_entities == len(entities):
                    return

    for child in node.children:
        _walk_and_collect(child, entities, lines, file_path, max_entities)
