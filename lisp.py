#!/usr/bin/env python
# coding: utf-8
# # VSA Lisp with Residue Numbers
#
# The following is an interpreter based on Tomkins-Flanagan & Kelly (2024),
# encoding integers as Residue Numbers, and changing the underlying algebra
# to Fourier Holographic Reduced Representations.

# In[]:
import numpy as np

from typing import Dict, Callable, Tuple, List, Any
import math
import cmath

from abc import ABCMeta, abstractmethod

dim = 600
np.random.seed(0)


# In[]:

# # Vocabularies
#
# In order to capture the differing operations on the underlying representations,
# we will be encoding the two representations using the `Vocabulary` abstraction,
# which allows for modular definition of the algebraic operations.

class Vocabulary(metaclass=ABCMeta):
    """
    `Vocabulary`

    Abstract base class for vector symbolic algebras.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def symbols(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def vector_gen(self) -> np.ndarray:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> np.ndarray:
        pass


#  # FHHR and RHC
#
#  Fourier Holographic Reduced Representations are Holographic Reduced
#  Representations (Plate, 1988) in the fourier domain. Hypervectors are
#  vectors $\mathbb{C}^D$, and a set of angles $\theta_1, \theta_2, \ldots, \theta_D$,
#  $$
#  [e^{i \theta_1}, e^{i \theta_2}, \ldots, e^{i \theta_D}]
#  $$

# In[]:
class FHRR(Vocabulary):
    """
    `FHRR`

    Vocabulary and associated methods for Fourier Holographic Reduced
    Representations.
    """

    _dim: int
    _symbols: Dict[str, np.ndarray]

    def __init__(
        self, dim: int, symbols: Dict[str, np.ndarray] | List[str] = {}
    ) -> None:
        self._dim = dim

        if isinstance(symbols, dict):
            self._symbols = symbols
        elif isinstance(symbols, list):
            self._symbols = {}
            for symbol in symbols:
                self._symbols[symbol] = self.vector_gen()
        else:
            raise TypeError(
                "expected `Dict[str, np.ndarray] | List[str]`", symbols
            )

    @property
    def dim(self) -> int:
        """
        `dim`

        The dimensionality of the vocabulary.
        """
        return self._dim

    @property
    def symbols(self) -> Dict[str, np.ndarray]:
        """
        `symbols`

        Associative array between strings and vectors in the vocabulary.
        """
        return self._symbols

    @symbols.setter
    def symbols(self, value: Dict[str, np.ndarray] | List[str]) -> None:
        if isinstance(value, dict):
            self._symbols = value
        elif isinstance(value, list):
            self._symbols = {}
            for symbol in value:
                self._symbols[symbol] = self.vector_gen()
        else:
            raise TypeError(
                "expected `Dict[str, np.ndarray] | List[str]`", value
            )

    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        `bind`

        Element-wise multiplication between vectors `x` and `y`.
        """
        assert x.dtype == y.dtype, "dtypes must match"
        assert x.size == y.size, "sizes must match"
        return np.multiply(x, y)

    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        `superpose`

        Element-wise addition between vectors `x` and `y`.
        """
        assert x.dtype == y.dtype, "dtypes must match"
        assert x.size == y.size, "sizes must match"
        res = x + y
        # return res / np.linalg.norm(res)
        return res

    def vector_gen(self) -> np.ndarray:
        """
        `vector_gen`

        Initialize a new vector.
        """
        thetas = np.random.uniform(high=math.pi * 2, size=dim)
        return np.exp(thetas * cmath.sqrt(-1))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._symbols[key]

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        `sim`

        Vector similarity between vectors `x` and `y`.
        """
        assert x.dtype == y.dtype, "dtypes must match"
        assert x.size == y.size, "sizes must match"
        return abs(np.dot(np.conjugate(x.T), y).real / x.size)

    def inv(self, x: np.ndarray) -> np.ndarray:
        """
        `inv`

        Vector inverse.
        """
        return np.conjugate(x)

    def unbind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        `unbind`

        Element-wise multiplication between vector `x` and the
        complex conjugate of `y`.
        """
        return self.bind(x, self.inv(y))

    def reverse(self) -> Dict[Tuple[Any, ...], str]:
        return {hash_array(v): k for (k, v) in self._symbols.items()}

    def set_symbols(self, symbols: List[str]) -> None:
        for symbol in symbols:
            self._symbols[symbol] = self.vector_gen()

    def add_symbol(self, symbol: str) -> None:
        self._symbols[symbol] = self.vector_gen()


fhrr = FHRR(dim=dim)


# In[]:
red = fhrr.vector_gen()
ball = fhrr.vector_gen()

red_ball = fhrr.bind(red, ball)

red_and_ball = fhrr.superpose(red, ball)

fhrr.sim(red_and_ball, ball)

# In[]:

#
# For Residue Hyperdimensional computing, we define the vectors as something
# similar; except for the fact that we must specify: moduli (to make
# roots of unity of the unit circle).

# In[]:
# TODO: implement residue hyperdimensional computing
class RHC(Vocabulary):
    """
    `RHC`

    Residue Hyperdimensional Computing vocabulary with associated
    algebraic operations.
    """

    _dim: int
    _symbols: Dict[str, np.ndarray]
    _moduli: list[int]

    def __init__(self, dim: int, moduli: list[int]) -> None:
        self._dim = dim
        self._moduli = moduli
        self._symbols = {}

    @property
    def dim(self) -> int:
        """
        `dim`

        Dimensionality of the vocabulary.
        """
        return self._dim

    @property
    def symbols(self) -> Dict[str, np.ndarray]:
        return self._symbols

    def _roots_of_unity(self, m: int) -> np.ndarray:
        incr = 2 * math.pi / m
        points = [2 * math.pi]
        curr_val = incr
        while curr_val < 2 * math.pi:
            points.append(curr_val)
            curr_val += incr
        v = np.zeros(self._dim)
        choose = np.vectorize(lambda _: np.random.choice(points))
        return choose(v)

    def _gen_moduli(self, modulo: int) -> np.ndarray:
        phis = self._roots_of_unity(modulo)
        return np.exp(phis * cmath.sqrt(-1))

    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("TODO")

    def superpose(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("TODO")

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        raise Exception("todo")

    def vector_gen(self) -> np.ndarray:
        raise Exception("TODO")

    def __getitem__(self, symbol: str) -> np.ndarray:
        raise Exception("TODO")

    def v(self, x: int) -> np.ndarray:
        """
        `v`

        Encode a number `x` into the residue arithmetic code.
        """
        raise Exception("todo")

    def add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("todo")

    def mul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("todo")

    def sub(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("todo")

    def div(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("todo")


rhc = RHC(dim=dim, moduli=[3, 5, 7])


# In[]:


# # Resonator Networks
#
# Let us have codebooks $X = [x_1, x_2, \ldots, x_D]$,
# $Y = [y_1, y_2, \ldots, y_D]$, and $Z = [z_1, z_2, \ldots, z_D]$. Given some
# $s := x_{i^*} \odot y_{j^*} \odot z_{k^*}$, and codebooks $X, Y$ and $Z$,
# the goal is to find $x_{i^*}, y_{j^*}, z_{k^*}$.
#
# Let $\hat x, \hat y$, and $\hat z$ represent the estimate for each factor.
# These vectors can be initialized to the superposition of all possible factors,
# $$
# \begin{align*}
# \hat x (0) &= \sum^D_i x_i,\\
# \hat y (0) &= \sum^D_j y_j, \\
# \hat z (0) &= \sum^D_r z_r
# \end{align*}
# $$
#
# A factor can be inferred on the basis of the other two; for example,
# $$
# \hat z (1) = s \odot \hat x (0) \odot \hat y (0)
# $$
# Because the binding of $\hat x (0) \odot \hat y (0)$ is the superposition
# of all possible codes in the codebook, it represents, for example if $D = 100$,
# then $D^2 = 10,000$.
#
# The results of inference can be improved using clean-up memory which helps
# reduce noise, and the process is built on cross-talk. We can, through
# iterative application, arrive at good enough estimates.
# $$
# \begin{align*}
# \hat x (t + 1) &= g (X X^T (s \odot \hat y (t) \odot \hat z (t))), \\
# \hat y (t + 1) &= g (Y Y^T (s \odot \hat x (t) \odot \hat z (t))), \\
# \hat z (t + 1) &= g (Z Z^T (s \odot \hat x (t) \odot \hat y (t)))
# \end{align*}
# $$
# where $g$ is a function preventing run-away feedback, holding the values
# of each vector at $\pm 1$.
#
# The clean-up memory for $\hat x$ which is the matrix multiplication $XX^T$
# with threshold function $g$, then this operation is equivalent to outer-product
# Hebbian learning (Hopfield, 1982); except here, rather than directing feeding
# the network back to itself, the result of the clean-up is sent to other
# parts of the network.
#

# TODO: implement general resonator network
def resonator(
    *,
    label: np.ndarray,
    expr: np.ndarray,
    codebooks: np.ndarray,
    iterations: int = 20,
) -> np.ndarray:
    """
    `resonator`

    Args:
        label : np.ndarray, D-dim vec
        expr : np.ndarray, D-dim vec
        codebooks : np.ndarray, n x D matrix
        iterations : int, no. of iterations

    Returns:
        estimated value
    """
    raise Exception("todo")


# # Memory

# In[]:
floor = 0.2
"""
Maximum value for two vectors to be the equivalent.
"""


def hash_array(x: np.ndarray) -> Tuple[Any, ...]:
    return (*x,)


# TODO: implemenent more effective cleanup memory
class SimpleCleanup:
    """
    `SimpleCleanup`

    Simple clean-up, auto-associative memory used for attracting representations
    to their closest stored representation.
    """

    memory_matrix: np.ndarray
    dim: int
    max_trace: int
    incr: int
    size: int

    def __init__(
        self, dim: int = dim, max_trace: int = 100
    ):
        self.memory_matrix = np.zeros((max_trace, dim), dtype=complex)
        self.dim = dim
        self.max_trace = max_trace
        self.incr = max_trace
        self.size = 0

    def memorize(self, v: np.ndarray, name: str | None = None) -> np.ndarray:
        if self.size >= self.max_trace:
            self.memory_matrix = np.concatenate(
                [self.memory_matrix, np.zeros((self.incr, self.dim))], axis=0
            )
            self.max_trace += self.incr
        self.memory_matrix[self.size, :] = v
        self.size += 1

        if name is not None:
            print(f"stored {name} at row {self.size-1}")

        return v

    def recall(self, v: np.ndarray) -> np.ndarray:
        activations = [fhrr.sim(v, m) for m in self.memory_matrix]
        return self.memory_matrix[np.argmax(activations), :]


class AssocMem:
    _theta: float = floor
    _assoc: Dict[Tuple[Any, ...], np.ndarray]

    def __init__(self) -> None:
        self._assoc = {}

    def alloc(
        self, probe: np.ndarray, trace: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._assoc[hash_array(probe)] = trace
        return probe, trace

    def deref(self, probe: np.ndarray) -> np.ndarray:
        if len(self._assoc) == 0:
            return probe

        trace = self._assoc.get(hash_array(probe))
        if trace is not None:
            return trace

        keys = list(self._assoc.keys())
        acts = [fhrr.sim(probe, np.array(k)) for k in keys]
        nearest = np.argmax(acts)

        if (acts[nearest]) > self._theta:
            return self._assoc[keys[nearest]]

        return probe


def closest(value: np.ndarray) -> str:
    max_sim = 0.0
    max_word = "NONE"
    for word in fhrr.symbols:
        sim = fhrr.sim(fhrr[word], value)
        if sim > max_sim:
            max_sim = sim
            max_word = word
    return max_word


mem = SimpleCleanup()
define_mem = AssocMem()
assoc_mem = AssocMem()
basic_functions = [
    "car",
    "cdr",
    "eq?",
    "cons",
    "atom?",
    "int?",
    "quote",
    "if",
    "+",
    "*",
    "-",
    "/",
]
keywords = [
    "lambda",
    "define",
]
constants = [
    "#t",
    "#f",
    "nil",
]


def _clear_symbols_and_reset() -> None:
    global mem, assoc_mem, define_mem, basic_functions, keywords, constants
    global fhrr, rhc
    fhrr.symbols = {}

    kws = (
        basic_functions
        + keywords
        + constants
        + [
            # private values, if a user uses these, they die
            "__lhs",
            "__rhs",
            "__phi",
            "__args",
            "__body",
            "__func",
        ]
    )

    mem = SimpleCleanup()
    assoc_mem = AssocMem()
    define_mem = AssocMem()
    fhrr = FHRR(dim=dim)
    # rhc = RHC(dim=dim)

    fhrr.set_symbols(kws)
    for key, value in fhrr.symbols.items():
        mem.memorize(value, name=key)


# In[]:
# # The LISP Interpreter and Environment
#
# ## Encoding

# In[]:
def encode_number(x: int) -> np.ndarray:
    """
    `encode_number`

    Encode an integer `x`, adding it to cleanup memory.
    """
    vx = rhc.v(x)
    return mem.memorize(fhrr.superpose(vx, fhrr["__int"]))


def encode_atom(name: str) -> np.ndarray:
    """
    `encode_atom`

    Encode an atom `name`.
    """
    print(f"encode_atom: {name}")
    if name not in fhrr.symbols:
        fhrr.add_symbol(name)
    trace = mem.memorize(fhrr[name])
    return trace


def _cons(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    `_cons`

    Return a semantic pointer to a tuple of `x` and `y`.
    """
    ptr = fhrr.vector_gen()
    x = mem.memorize(x)
    y = mem.memorize(y)
    ptr, _ = assoc_mem.alloc(
        ptr,
        # (x * __lhs) + (y * __rhs) + __phi
        fhrr.superpose(
            fhrr.bind(x, fhrr["__lhs"]),
            fhrr.superpose(fhrr.bind(y, fhrr["__rhs"]), fhrr["__phi"]),
        ),
    )
    return ptr


def encode_list(xs: List[Any]) -> np.ndarray:
    """
    `encode_list`

    Encode a list into a vector.
    """
    print(f"encode_list: {xs}")
    if len(xs) == 0:
        return fhrr["nil"]

    head, *tail = xs
    print(f"{head =}")
    print(f"{tail =}")
    headv = encode(head)

    if len(tail) == 0:
        return _cons(headv, fhrr["nil"])
    else:
        return _cons(headv, encode_list(tail))


def encode_tuple(xs: Tuple[Any, ...]) -> np.ndarray:
    """
    `encode_tuple`

    Encode a tuple into a vector.
    """
    print(f"encode_tuple: {xs}")
    print(f"encode_tuple: {xs}")
    if len(xs) != 2:
        raise ValueError("only 2-tuples are able to be encoded")

    left, right = xs
    return _cons(encode(left), encode(right))


def encode(item: Any) -> np.ndarray:
    """
    `encode`

    Encode a `str`, `tuple`, `list`, or `np.ndarray` into VSA.
    """
    print(f"encode: {item}")
    if isinstance(item, int):
        return encode_number(item)
    elif isinstance(item, np.ndarray):
        return item
    elif isinstance(item, str):
        return encode_atom(item)
    elif isinstance(item, list):
        return encode_list(item)
    elif isinstance(item, tuple):
        return encode_tuple(item)
    else:
        raise TypeError(
            "can only handle strings, 2-tuples, lists, and np.ndarrays", item
        )


# In[]:
# ## Evaluation

# In[]:
def is_aeq(x: np.ndarray, y: np.ndarray) -> bool:
    """
    `is_aeq`

    Is Approximately EQual
    """
    simi = fhrr.sim(x, y)
    return simi > floor


def check_symbol(code: np.ndarray) -> bool:
    """
    `check_symbol`

    Compare encoded representation to the current lexicon.
    """
    is_symb = any(fhrr.sim(code, x) > floor for x in fhrr.symbols.values())
    return is_symb


def check_atomic(code: np.ndarray) -> np.ndarray:
    """
    `check_atomic`

    Determine whether a value is atomic or not, or it is a list.
    """
    # if check_symbol(code):
    #     return fhrr["#t"]
    # dereference the value
    value = assoc_mem.deref(code)
    close_to_phi = fhrr.sim(value, fhrr["__phi"]) * fhrr["#f"]
    far_from_phi = (
        max((2 * floor) - fhrr.sim(value, fhrr["__phi"]), 0.0) * fhrr["#t"]
    )
    return mem.recall(fhrr.superpose(close_to_phi, far_from_phi))


def check_function(code: np.ndarray) -> np.ndarray:
    """
    `check_atomic`

    Determine whether a value is atomic or not, or it is a list.
    """
    # if check_symbol(code):
    #     return fhrr["#t"]
    # dereference the value
    value = assoc_mem.deref(code)
    close_to_func = fhrr.sim(value, fhrr["__func"]) * fhrr["#f"]
    far_from_func = (
        max((2 * floor) - fhrr.sim(value, fhrr["__func"]), 0.0) * fhrr["#t"]
    )
    return mem.recall(fhrr.superpose(close_to_func, far_from_func))


def car(
    x: np.ndarray, locals_: AssocMem | None = None, eval_: bool = False
) -> np.ndarray:
    """
    `car`

    Dereference and return the first element of a tuple.
    """
    xdref = assoc_mem.deref(x)
    # print(f"{lisp_to_str(xdref)=}")
    # print(f"{lisp_to_str(fhrr.unbind(xdref, fhrr["__lhs"]))=}")
    xev = evaluate(xdref, locals_=locals_) if eval_ else xdref
    return mem.recall(fhrr.unbind(xev, fhrr["__lhs"]))


def cdr(
    x: np.ndarray, locals_: AssocMem | None = None, eval_: bool = False
) -> np.ndarray:
    """
    `cdr`

    Dereference and return the second element of a tuple.
    """
    xdref = assoc_mem.deref(x)
    xev = evaluate(xdref, locals_=locals_) if eval_ else xdref
    return mem.recall(fhrr.unbind(xev, fhrr["__rhs"]))


def cons(x: np.ndarray, *, locals_: AssocMem | None) -> np.ndarray:
    """
    `cons`

    Create a tuple.
    """
    car_ = evaluate(car(x), locals_=locals_)
    cdr_ = evaluate(car(cdr(x)), locals_=locals_)
    return _cons(car_, cdr_)


def eq(x: np.ndarray, locals_: AssocMem | None) -> np.ndarray:
    """
    `eq`

    Test object equality. Implements `eq?`.
    """
    lhs = evaluate(car(x), locals_=locals_)
    rhs = evaluate(car(cdr(x)), locals_=locals_)
    return mem.recall(
        fhrr.superpose(
            fhrr.sim(lhs, rhs) * fhrr["#t"],
            (1 - fhrr.sim(lhs, rhs)) * fhrr["#f"],
        )
    )


def atom(x: np.ndarray, locals_: AssocMem | None) -> np.ndarray:
    """
    `atom`

    Implements the builtin function `atom?`
    """
    x = evaluate(x, locals_=locals_)
    lhs = fhrr.sim(cdr(x), fhrr["nil"]) * check_atomic(car(x))
    rhs = max((2 * floor) - fhrr.sim(x, fhrr["nil"]), 0.0) * fhrr["#f"]
    return fhrr.superpose(lhs, rhs)


def quote(x: np.ndarray) -> np.ndarray:
    """
    `quote`

    Implements the built-in `quote` function.
    """
    return _cons(fhrr["quote"], _cons(car(x), fhrr["nil"]))


def if_(x: np.ndarray, *, locals_: AssocMem | None) -> np.ndarray:
    condition = evaluate(car(x), locals_)
    fst_branch = mem.recall(car(cdr(x)))
    snd_branch = mem.recall(car(cdr(cdr(x))))
    if is_aeq(condition, fhrr["#t"]):
        return evaluate(fst_branch, locals_=locals_)
    else:
        return evaluate(snd_branch, locals_=locals_)


def add(x: np.ndarray, locals_: AssocMem | None) -> np.ndarray:
    lhs = mem.recall(evaluate(car(x), locals_=locals_))
    rhs = mem.recall(evaluate(car(cdr(x)), locals_=locals_))

    lhs_i = lhs - fhrr["__int"]
    rhs_i = rhs - fhrr["__int"]
    return fhrr.bind(rhc.add(lhs_i, rhs_i), fhrr["__int"])


def mul(x: np.ndarray, locals_: AssocMem | None) -> np.ndarray:
    lhs = mem.recall(evaluate(car(x), locals_=locals_))
    rhs = mem.recall(evaluate(car(cdr(x)), locals_=locals_))

    lhs_i = lhs - fhrr["__int"]
    rhs_i = rhs - fhrr["__int"]
    return fhrr.bind(rhc.mul(lhs_i, rhs_i), fhrr["__int"])


def div(x: np.ndarray, locals_: AssocMem | None) -> np.ndarray:
    lhs = mem.recall(evaluate(car(x), locals_=locals_))
    rhs = mem.recall(evaluate(car(cdr(x)), locals_=locals_))

    lhs_i = lhs - fhrr["__int"]
    rhs_i = rhs - fhrr["__int"]
    return fhrr.bind(rhc.div(lhs_i, rhs_i), fhrr["__int"])


def sub(x: np.ndarray, locals_: AssocMem | None) -> np.ndarray:
    lhs = mem.recall(evaluate(car(x), locals_=locals_))
    rhs = mem.recall(evaluate(car(cdr(x)), locals_=locals_))

    lhs_i = lhs - fhrr["__int"]
    rhs_i = rhs - fhrr["__int"]
    return fhrr.bind(rhc.sub(lhs_i, rhs_i), fhrr["__int"])


def evaluate_lambda(lam: np.ndarray) -> np.ndarray:
    """
    `evaluate_lambda`

    Return a pointer to a lambda function.
    """
    ptr = fhrr.vector_gen()
    params = mem.recall(car(lam))
    body = mem.recall(car(cdr(lam)))
    ptr, _ = assoc_mem.alloc(
        ptr,
        fhrr.superpose(
            fhrr.bind(params, fhrr["__args"]),
            fhrr.superpose(fhrr.bind(body, fhrr["__body"]), fhrr["__func"]),
        ),
    )
    return ptr


def evaluate_define(x: np.ndarray) -> np.ndarray:
    """
    `evaluate_define`

    Evaluate a `define` expression, adding it to the memory.
    """
    name, body = car(x), car(cdr(x))
    define_mem.alloc(name, body)
    return fhrr["nil"]


# TODO
def get_params(code: np.ndarray) -> np.ndarray:
    """
    `get_params`

    Retrieve the parameters from a function pointer.
    """
    raise Exception("TODO")


# TODO
def get_body(code: np.ndarray) -> np.ndarray:
    """
    `get_body`

    Retrieve the body from a function pointer.
    """
    raise Exception("TODO")


# TODO
def associate(
    params: np.ndarray, args: np.ndarray, locals_: AssocMem | None
) -> AssocMem:
    """
    `associate`

    Associate parameters `params` with  `arguments`, creating a new
    local associative memory. Note that names in `locals_` are rebound
    to their new local values, if present in parameters.
    """
    raise Exception("TODO")


def evaluate_application(
    operator: np.ndarray, operands: np.ndarray, *, locals_: AssocMem | None
) -> np.ndarray:
    """
    `evaluate_application`

    Dereference operator `operator` and apply it to operands `operands`.
    """
    opx = (
        locals_.deref(mem.recall(operator))
        if locals_
        else mem.recall(operator)
    )

    basic_fn = fhrr.reverse().get(hash_array(opx))
    if basic_fn is not None and basic_fn in basic_functions:
        if basic_fn == "car":
            return car(operands, locals_=locals_, eval_=True)
        elif basic_fn == "cdr":
            return cdr(operands, locals_=locals_, eval_=True)
        elif basic_fn == "cons":
            return cons(operands, locals_=locals_)
        elif basic_fn == "eq?":
            return eq(operands, locals_=locals_)
        elif basic_fn == "atom?":
            return atom(operands, locals_=locals_)
        elif basic_fn == "quote":
            return quote(operands)
        elif basic_fn == "if":
            return if_(operands, locals_=locals_)
        elif basic_fn == "+":
            return add(operands, locals_=locals_)
        elif basic_fn == "*":
            return mul(operands, locals_=locals_)
        elif basic_fn == "/":
            return div(operands, locals_=locals_)
        elif basic_fn == "-":
            return sub(operands, locals_=locals_)
        else:
            raise Exception()
    elif is_aeq(check_function(opx), fhrr["#t"]):
        params, body = get_params(operator), get_body(operator)
        lambda_locals = associate(params, operands, locals_)
        return evaluate(body, locals_=lambda_locals)
    else:
        # If `opx` isn't a function, then we return it unevaluated,
        # and continue.
        return _cons(opx, evaluate(operands, locals_=locals_))


def evaluate(code: np.ndarray, locals_: AssocMem | None = None) -> np.ndarray:
    """
    `evaluate`

    'One-step' evaluation of encoded syntax `code`. Called 'one-step'
    as we use the novel lambda encoding.
    """

    # If it's just atomic, then we check to see if the name is bound, and
    # replace it if so
    if is_aeq(check_atomic(code), fhrr["#t"]):
        print('is an atom')
        if locals_ is None:
            return define_mem.deref(code)
        else:
            return locals_.deref(code)

    head, tail = car(code), cdr(code)

    print(lisp_to_str(head))

    # Make a function pointer
    if is_aeq(head, fhrr["lambda"]):
        return evaluate_lambda(tail)
    elif is_aeq(head, fhrr["define"]):
        return evaluate_define(tail)
    else:
        # Otherwise, we're going to treat this as a function application.
        return evaluate_application(head, tail, locals_=locals_)


# In[]:
# # Main functions

# In[]:
def interpret(syntax: List[Any], clear_env: bool = True) -> List[np.ndarray]:
    """
    `interpet`

    Interpret a source-level syntax into VSA coded lisp expressions, and return
    the raw result.
    """
    if clear_env:
        _clear_symbols_and_reset()

    codes = []
    for item in syntax:
        codes.append(encode(item))

    results = []
    for code in codes:
        results.append(evaluate(code))

    return results


def parse_expr(tokens: List[str]) -> Any:
    """
    `parse_expr`

    Parse an expression.
    """
    print(f"parse_expr: {tokens}")
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF while reading")
    token = tokens.pop(0)
    if token == "(":
        print("entering list")
        expr = []
        while tokens[0] != ")":
            expr.append(parse_expr(tokens))

        assert tokens[0] == ")", "expected closing parenthesis"
        tokens.pop(0)

        if len(expr) == 3 and expr[1] == ".":
            return (expr[0], expr[1])

        return expr
    elif token == ")":
        raise SyntaxError("unexpected closing parenthesis")
    elif token.isnumeric():
        return int(token)
    else:
        return str(token)


def tokenize(txt: str) -> List[str]:
    return txt.lower().replace("(", " ( ").replace(")", " ) ").split()


def parse(txt: str) -> List[str]:
    """
    `parse`

    Convert raw string `txt` into a list of tokens.
    """
    syntax = []
    tokens = tokenize(txt)
    while tokens:
        syntax.append(parse_expr(tokens))
    return syntax


def lisp_to_str(value: np.ndarray) -> str | List[Any] | Tuple[Any, ...]:
    """
    `lisp_to_str`

    Convert a lisp expression encoded in VSA into a `np.ndarray`.
    """
    if is_aeq(check_atomic(value), fhrr["#t"]):
        return closest(value)

    left, right = lisp_to_str(car(value)), lisp_to_str(cdr(value))

    if isinstance(right, list):
        return [left, *right]
    elif right == "NIL":
        return [left]
    else:
        return (left, right)


def run(txt: str) -> None:
    """
    `run`

    Interpret a `txt` as a source-level representation of lisp, and
    print the results of the evalutation.
    """
    syntax = parse(txt)
    values = interpret(syntax)
    for value in values:
        print(lisp_to_str(value))


# In[]:

# Sample Program

conts = """
(cons (quote foo) (quote bar))
"""

# # print(conts)
run(conts)

# run("(car (cons (quote foo) (quote bar)))")
# In[]:

# _clear_symbols_and_reset()
# e = encode(parse("(cons (quote foo) (quote bar))")[0])
# lisp_to_str(e)


# %%
