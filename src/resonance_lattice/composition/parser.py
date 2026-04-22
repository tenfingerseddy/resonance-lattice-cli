# SPDX-License-Identifier: BUSL-1.1
"""Expression parser for composition expressions.

Parses RQL-style arithmetic expressions on knowledge model references into
an AST that ComposedCartridge can evaluate.

Supported syntax:
    A                          → single knowledge model
    A + B                      → merge(A, B)
    0.7 * A + 0.3 * B         → weighted merge
    A ~ B                      → diff(A, B)  "what's new in A vs B"
    A ^ B                      → contradict(A, B)  "where do A and B disagree"
    project(A, B)              → project A through B's lens
    (A + B) ~ C                → grouping with parentheses

Grammar:
    expr     ::= term (('+' | '~' | '^') term)*
    term     ::= factor ('*' factor)?
    factor   ::= NUMBER '*' atom | atom
    atom     ::= CARTRIDGE_PATH
               | 'project(' expr ',' expr ')'
               | '(' expr ')'
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════
# AST Nodes
# ═══════════════════════════════════════════════════════════

@dataclass
class CartridgeRef:
    """Leaf: a path to a .rlat file."""
    path: str

    @property
    def name(self) -> str:
        """Stem of the path for display."""
        from pathlib import Path
        return Path(self.path).stem


@dataclass
class MergeNode:
    """Binary: merge two expressions."""
    left: ExprNode
    right: ExprNode


@dataclass
class WeightedNode:
    """Unary: scale an expression."""
    weight: float
    child: ExprNode


@dataclass
class DiffNode:
    """Binary: diff (newer ~ older)."""
    newer: ExprNode
    older: ExprNode


@dataclass
class ContradictNode:
    """Binary: contradiction (A ^ B)."""
    left: ExprNode
    right: ExprNode


@dataclass
class ProjectNode:
    """Binary: project(source, lens)."""
    source: ExprNode
    lens: ExprNode


ExprNode = CartridgeRef | MergeNode | WeightedNode | DiffNode | ContradictNode | ProjectNode


# ═══════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(
    r"""
    (?P<FLOAT>\d+\.\d+)         |  # 0.7
    (?P<INT>\d+)                 |  # bare integer
    (?P<OP>[+~^*(),])            |  # operators and parens
    (?P<FUNC>project(?=\s*\())   |  # function name (only when followed by '(')
    (?P<PATH>[a-zA-Z0-9_./-]+\.rlat) |  # knowledge model path
    (?P<NAME>[a-zA-Z_][a-zA-Z0-9_.-]*) |  # bare name (alias)
    (?P<WS>\s+)                     # whitespace (skip)
    """,
    re.VERBOSE,
)


@dataclass
class Token:
    type: str
    value: str


def tokenize(expr: str) -> list[Token]:
    """Tokenize a composition expression."""
    tokens = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if m is None:
            raise SyntaxError(f"Unexpected character at position {pos}: {expr[pos:]!r}")
        pos = m.end()
        for name, value in m.groupdict().items():
            if value is not None and name != "WS":
                tokens.append(Token(type=name, value=value))
                break
    return tokens


# ═══════════════════════════════════════════════════════════
# Recursive descent parser
# ═══════════════════════════════════════════════════════════

class Parser:
    """Recursive descent parser for composition expressions."""

    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_type: str | None = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")
        if expected_type and tok.type != expected_type:
            raise SyntaxError(
                f"Expected {expected_type}, got {tok.type} ({tok.value!r}) "
                f"at position {self._pos}"
            )
        self._pos += 1
        return tok

    def _at(self, value: str) -> bool:
        tok = self._peek()
        return tok is not None and tok.value == value

    def parse(self) -> ExprNode:
        """Parse the full expression."""
        result = self._parse_expr()
        if self._pos < len(self._tokens):
            remaining = self._tokens[self._pos:]
            raise SyntaxError(
                "Unexpected tokens after expression: "
                + " ".join(t.value for t in remaining)
            )
        return result

    def _parse_expr(self) -> ExprNode:
        """expr ::= term (('+' | '~' | '^') term)*"""
        left = self._parse_term()

        while self._peek() and self._peek().value in ("+", "~", "^"):
            op = self._consume().value
            right = self._parse_term()
            if op == "+":
                left = MergeNode(left, right)
            elif op == "~":
                left = DiffNode(left, right)
            elif op == "^":
                left = ContradictNode(left, right)

        return left

    def _parse_term(self) -> ExprNode:
        """term ::= (NUMBER '*')? atom"""
        tok = self._peek()

        # Check for weight: 0.7 * atom
        if tok and tok.type in ("FLOAT", "INT"):
            # Look ahead for '*'
            if self._pos + 1 < len(self._tokens) and self._tokens[self._pos + 1].value == "*":
                weight = float(self._consume().value)
                self._consume()  # consume '*'
                child = self._parse_atom()
                return WeightedNode(weight=weight, child=child)

        return self._parse_atom()

    def _parse_atom(self) -> ExprNode:
        """atom ::= PATH | NAME | 'project(' expr ',' expr ')' | '(' expr ')'"""
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression, expected a value")

        # Parenthesized group
        if tok.value == "(":
            self._consume()
            result = self._parse_expr()
            if not self._at(")"):
                raise SyntaxError("Missing closing parenthesis")
            self._consume()
            return result

        # Function call: project(...)
        if tok.type == "FUNC" and tok.value == "project":
            self._consume()  # 'project'
            if not self._at("("):
                raise SyntaxError("Expected '(' after 'project'")
            self._consume()  # '('
            source = self._parse_expr()
            if not self._at(","):
                raise SyntaxError("Expected ',' in project(source, lens)")
            self._consume()  # ','
            lens = self._parse_expr()
            if not self._at(")"):
                raise SyntaxError("Expected ')' to close project()")
            self._consume()  # ')'
            return ProjectNode(source=source, lens=lens)

        # Cartridge path or name
        if tok.type in ("PATH", "NAME"):
            self._consume()
            path = tok.value
            # If it's a bare name, assume .rlat extension
            if tok.type == "NAME" and not path.endswith(".rlat"):
                path = path + ".rlat"
            return CartridgeRef(path=path)

        raise SyntaxError(f"Unexpected token: {tok.type} ({tok.value!r})")


def parse_expression(expr: str) -> ExprNode:
    """Parse a composition expression string into an AST.

    Examples:
        parse_expression("docs.rlat + code.rlat")
        parse_expression("0.7 * docs.rlat + 0.3 * code.rlat")
        parse_expression("current.rlat ~ baseline.rlat")
        parse_expression("docs.rlat ^ code.rlat")
        parse_expression("project(code.rlat, compliance.rlat)")
        parse_expression("(docs + code) ~ baseline")
    """
    tokens = tokenize(expr)
    parser = Parser(tokens)
    return parser.parse()


def collect_cartridge_paths(node: ExprNode) -> list[str]:
    """Extract all knowledge model file paths from an AST."""
    paths = []
    if isinstance(node, CartridgeRef):
        paths.append(node.path)
    elif isinstance(node, WeightedNode):
        paths.extend(collect_cartridge_paths(node.child))
    elif isinstance(node, (MergeNode, ContradictNode)):
        paths.extend(collect_cartridge_paths(node.left))
        paths.extend(collect_cartridge_paths(node.right))
    elif isinstance(node, DiffNode):
        paths.extend(collect_cartridge_paths(node.newer))
        paths.extend(collect_cartridge_paths(node.older))
    elif isinstance(node, ProjectNode):
        paths.extend(collect_cartridge_paths(node.source))
        paths.extend(collect_cartridge_paths(node.lens))
    return paths
