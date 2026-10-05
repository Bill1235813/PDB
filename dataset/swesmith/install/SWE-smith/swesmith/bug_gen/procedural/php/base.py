"""
Base class for PHP procedural modifications.
"""

from abc import ABC

import tree_sitter_php as tsphp
from tree_sitter import Language, Parser

from swesmith.bug_gen.procedural.base import ProceduralModifier

PHP_LANGUAGE = Language(tsphp.language_php_only())


def php_parse(src_code: str):
    """Parse PHP source code using the php_only grammar.

    Returns (tree, offset) where offset is always 0.
    """
    parser = Parser(PHP_LANGUAGE)
    tree = parser.parse(bytes(src_code, "utf8"))
    return tree, 0


class PhpProceduralModifier(ProceduralModifier, ABC):
    """Base class for PHP-specific procedural modifications using tree-sitter AST."""

    pass
