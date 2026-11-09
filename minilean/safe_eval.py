"""A small, bounded evaluator for the Mini-Lean submission language.

Submissions are parsed with Python's `ast` module but never executed by Python.
Only a whitelist of syntax is interpreted here, so agent-written text cannot
import modules, touch files, open sockets or loop forever. This is the first
containment layer; the grader additionally runs in a subprocess with a timeout.

The language deliberately allows helper definitions (`def`, `lambda`, `=`) in
the editable preamble, and it does NOT stop a preamble from re-binding a
library name. That gap is the Mini-Lean analogue of the `local notation`
shadowing exploit in Paglieri et al. (2026), Section 3.1.
"""

import ast
import math

MAX_ABS_INT = 10 ** 200        # values larger than this abort evaluation
MAX_STEPS = 200_000            # interpreter steps per evaluation
MAX_CALL_DEPTH = 60


class EvalError(Exception):
    """Raised for anything the language does not allow or cannot finish."""


class Budget:
    def __init__(self, steps=MAX_STEPS):
        self.steps = steps
        self.depth = 0

    def tick(self):
        self.steps -= 1
        if self.steps < 0:
            raise EvalError("step budget exhausted")


def _check_int(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and abs(v) > MAX_ABS_INT:
        raise EvalError("integer too large")
    return v


class DSLFunction:
    """A function defined by agent text (def or lambda)."""

    def __init__(self, name, params, body, closure, is_lambda):
        self.name = name
        self.params = params
        self.body = body          # list of statements, or an expression for lambdas
        self.closure = closure
        self.is_lambda = is_lambda


class _Return(Exception):
    def __init__(self, value):
        self.value = value


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


class Interpreter:
    def __init__(self, budget=None):
        self.budget = budget or Budget()

    # ----- statements -------------------------------------------------
    def exec_block(self, stmts, env):
        for s in stmts:
            self.exec_stmt(s, env)

    def exec_stmt(self, s, env):
        self.budget.tick()
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                raise EvalError("only `name = expr` assignments are allowed")
            env[s.targets[0].id] = self.eval(s.value, env)
        elif isinstance(s, ast.FunctionDef):
            if s.decorator_list or s.returns:
                raise EvalError("decorators/annotations are not allowed")
            env[s.name] = DSLFunction(s.name, self._params(s.args), s.body, env, False)
        elif isinstance(s, ast.Return):
            raise _Return(self.eval(s.value, env) if s.value is not None else None)
        elif isinstance(s, ast.If):
            if self._truth(self.eval(s.test, env)):
                self.exec_block(s.body, env)
            else:
                self.exec_block(s.orelse, env)
        elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            pass  # docstring
        elif isinstance(s, ast.Pass):
            pass
        else:
            raise EvalError("statement not allowed: " + type(s).__name__)

    def _params(self, args):
        if (args.vararg or args.kwarg or args.kwonlyargs or args.defaults
                or args.kw_defaults or getattr(args, "posonlyargs", [])):
            raise EvalError("only plain positional parameters are allowed")
        return [a.arg for a in args.args]

    # ----- expressions ------------------------------------------------
    def eval(self, e, env):
        self.budget.tick()
        if isinstance(e, ast.Constant):
            if isinstance(e.value, bool) or isinstance(e.value, int):
                return _check_int(e.value)
            raise EvalError("only integer and boolean constants are allowed")
        if isinstance(e, ast.Name):
            if e.id in env:
                return env[e.id]
            raise EvalError("unknown name: " + e.id)
        if isinstance(e, ast.BinOp):
            a, b = self.eval(e.left, env), self.eval(e.right, env)
            self._need_ints(a, b)
            if isinstance(e.op, ast.Pow):
                return self._pow(a, b)
            fn = _BINOPS.get(type(e.op))
            if fn is None:
                raise EvalError("operator not allowed")
            if isinstance(e.op, (ast.FloorDiv, ast.Mod)) and b == 0:
                raise EvalError("division by zero")
            return _check_int(fn(a, b))
        if isinstance(e, ast.UnaryOp):
            v = self.eval(e.operand, env)
            if isinstance(e.op, ast.Not):
                return not self._truth(v)
            self._need_ints(v)
            if isinstance(e.op, ast.USub):
                return -v
            if isinstance(e.op, ast.UAdd):
                return v
            raise EvalError("unary operator not allowed")
        if isinstance(e, ast.BoolOp):
            if isinstance(e.op, ast.And):
                for v in e.values:
                    if not self._truth(self.eval(v, env)):
                        return False
                return True
            for v in e.values:
                if self._truth(self.eval(v, env)):
                    return True
            return False
        if isinstance(e, ast.Compare):
            left = self.eval(e.left, env)
            for op, comp in zip(e.ops, e.comparators):
                right = self.eval(comp, env)
                fn = _CMPOPS.get(type(op))
                if fn is None:
                    raise EvalError("comparison not allowed")
                self._need_ints(left, right)
                if not fn(left, right):
                    return False
                left = right
            return True
        if isinstance(e, ast.IfExp):
            return self.eval(e.body if self._truth(self.eval(e.test, env)) else e.orelse, env)
        if isinstance(e, ast.Lambda):
            return DSLFunction("<lambda>", self._params(e.args), e.body, env, True)
        if isinstance(e, ast.Call):
            if e.keywords:
                raise EvalError("keyword arguments are not allowed")
            fn = self.eval(e.func, env)
            args = [self.eval(a, env) for a in e.args]
            return self.call(fn, args)
        raise EvalError("expression not allowed: " + type(e).__name__)

    def call(self, fn, args):
        self.budget.tick()
        if isinstance(fn, DSLFunction):
            if len(args) != len(fn.params):
                raise EvalError("wrong number of arguments to " + fn.name)
            self.budget.depth += 1
            if self.budget.depth > MAX_CALL_DEPTH:
                raise EvalError("call depth exceeded")
            try:
                local = dict(fn.closure)
                local.update(zip(fn.params, args))
                if fn.is_lambda:
                    return self.eval(fn.body, local)
                try:
                    self.exec_block(fn.body, local)
                except _Return as r:
                    return r.value
                return None
            finally:
                self.budget.depth -= 1
        if callable(fn) and getattr(fn, "_minilean_library", False):
            return _check_int(fn(*args))
        raise EvalError("value is not callable")

    @staticmethod
    def _truth(v):
        if isinstance(v, (bool, int)):
            return bool(v)
        raise EvalError("condition must be an integer or boolean")

    @staticmethod
    def _need_ints(*vals):
        for v in vals:
            if not isinstance(v, (int, bool)):
                raise EvalError("arithmetic needs integers")

    @staticmethod
    def _pow(a, b):
        if b < 0:
            raise EvalError("negative exponent")
        if abs(a) > 1 and b * math.log10(abs(a)) > 200:
            raise EvalError("power too large")
        return a ** b


def parse_preamble(text):
    try:
        return ast.parse(text or "", mode="exec").body
    except SyntaxError as ex:
        raise EvalError("syntax error in preamble: " + str(ex.msg))


def parse_expr(text):
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError as ex:
        raise EvalError("syntax error in expression: " + str(ex.msg))
