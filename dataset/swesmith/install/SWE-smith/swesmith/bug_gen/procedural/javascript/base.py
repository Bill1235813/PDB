"""
Base class for JavaScript procedural modifications.
"""

from abc import ABC
from swesmith.bug_gen.procedural.base import ProceduralModifier


class JavaScriptProceduralModifier(ProceduralModifier, ABC):
    """Base class for JavaScript-specific procedural modifications using tree-sitter AST."""

    pass
