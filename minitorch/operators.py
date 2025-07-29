"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Iterable, Callable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies two numbers"""
    return a * b


def id(a: float) -> float:
    """Return the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -a


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another"""
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if two numbers are equal"""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of the two numbers"""
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value"""
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function"""
    if a >= 0:
        result = 1.0 / (1.0 + math.exp(-a))
    else:
        result = math.exp(a) / (1 + math.exp(a))

    epsilon = 1e-15

    return max(epsilon, min(1 - epsilon, result))


def relu(a: float) -> float:
    """Applies the ReLU activation function"""
    return max(0, a)


def log(a: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(a)


def exp(a: float) -> float:
    """Calculates the exponential function"""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal function"""
    return 1.0 / a


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second arg"""
    return (1.0 / a) * b


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return (1.0 / (a**2)) * b


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of a ReLU times a second arg"""
    derivative = 1 if a > 0 else 0
    return derivative * b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(a: Iterable[float], fn: Callable[[float], float]) -> Iterable[float]:
    """Higher-order function that applies a
    given function to each element of an iterable
    """
    return [fn(num) for num in a]


def zipWith(
    a: Iterable[float], b: Iterable[float], fn: Callable[[float, float], float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    res: list = []
    a = list(a)
    b = list(b)

    for i in range(min(len(a), len(b))):
        res.append(fn(a[i], b[i]))

    return res


def reduce(
    a: Iterable[float], initial: float, fn: Callable[[float, float], float]
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    for val in a:
        initial = fn(initial, val)

    return initial


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negates a list"""

    def neg(x: float) -> float:
        return -x

    return map(a, neg)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add two lists together"""
    return zipWith(a, b, add)


def sum(list: Iterable[float]) -> float:
    """Sum lists"""
    return reduce(list, 0.0, add)


def prod(list: Iterable[float]) -> float:
    """Take the product of lists"""
    return reduce(list, 1.0, mul)
