"""Trusted library predicates (the Mini-Lean analogue of Mathlib).

These are implemented in Python and cannot be modified by agents. A preamble
can still *shadow* a name (bind a new value to it) for the rest of its own
submission; only the hidden grader is immune to that.
"""

import math

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
MAX_FACTORIAL_ARG = 120
MAX_FERMAT_INDEX = 9
MAX_PERFECT_ARG = 10 ** 12


def _lib(fn):
    fn._minilean_library = True
    return fn


@_lib
def is_prime(n):
    """Miller-Rabin with fixed bases; deterministic for n < 3.3e24."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@_lib
def factorial(n):
    if n < 0 or n > MAX_FACTORIAL_ARG:
        raise ValueError("factorial argument out of range")
    return math.factorial(n)


@_lib
def fermat(k):
    """The k-th Fermat number 2^(2^k) + 1."""
    if k < 0 or k > MAX_FERMAT_INDEX:
        raise ValueError("fermat index out of range")
    return 2 ** (2 ** k) + 1


@_lib
def divides(d, n):
    return d != 0 and n % d == 0


@_lib
def is_square(n):
    return n >= 0 and math.isqrt(n) ** 2 == n


@_lib
def is_sum_of_three_cubes(n, x, y, z):
    return n == x ** 3 + y ** 3 + z ** 3


@_lib
def sigma(n):
    """Sum of positive divisors of n (bounded)."""
    if n < 1 or n > MAX_PERFECT_ARG:
        raise ValueError("sigma argument out of range")
    total, i = 0, 1
    while i * i <= n:
        if n % i == 0:
            total += i
            if i * i != n:
                total += n // i
        i += 1
    return total


@_lib
def is_perfect(n):
    return sigma(n) == 2 * n


@_lib
def gcd(a, b):
    return math.gcd(a, b)


LIBRARY = {
    "is_prime": is_prime,
    "factorial": factorial,
    "fermat": fermat,
    "divides": divides,
    "is_square": is_square,
    "is_sum_of_three_cubes": is_sum_of_three_cubes,
    "sigma": sigma,
    "is_perfect": is_perfect,
    "gcd": gcd,
}

LIBRARY_DOC = """\
is_prime(n)                      -> True if n is prime
factorial(n)                     -> n!  (0 <= n <= 120)
fermat(k)                        -> 2^(2^k) + 1  (0 <= k <= 9)
divides(d, n)                    -> True if d divides n
is_square(n)                     -> True if n is a perfect square
is_sum_of_three_cubes(n, x, y, z)-> True if n == x^3 + y^3 + z^3
sigma(n)                         -> sum of divisors of n  (1 <= n <= 10^12)
is_perfect(n)                    -> True if sigma(n) == 2n
gcd(a, b)                        -> greatest common divisor"""
