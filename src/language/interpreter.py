"""Module containing functions for interpreting and manipulating vector
symbols.
"""

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from syntax import KEYWORDS, OPERATORS, lex, parse
from vsa import VSA

from .encoding import (AssociativeMemory, EncodingEnvironment,
                       IntegerEncodingScheme, encode, make_cons)

BASIC_FUNCTIONS = [
    word
    for word in list(KEYWORDS.keys()) + list(OPERATORS.keys())
    if word not in ["define", "lambda"]
]
"""List of reserved, basic function symbols."""


@dataclass
class InterpreterError(Exception):
    """Exception thrown by the interpeter."""

    msg: str


@dataclass
class EvalEnvironment[T: (VSA[np.complex128], VSA[np.float64])]:
    """The evaluation environment, which contains a local associative memory
    for evaluation contexts, as well as a global definition memory, which
    remains constant throughout the evaluation of the program.
    """

    define_mem: AssociativeMemory[T]
    locals_: AssociativeMemory[T] | None


def is_approx_eq[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](x: T, y: T, env: EncodingEnvironment[T], floor: float = 0.2) -> bool:
    """Approximate equality between two vector symbols.

    Args:
    -   x (VSA): The left-hand side of the comparison.
    -   y (VSA): The right-hand side of the comparison.
    -   env (EncodingEnv): The encoding environment.
    -   floor (float): Defaults to `0.2`, the comparison floor.

    Returns:
        `True` iff the similarity between the `x` and `y` is greater than
        `floor`, `False` otherwise.
    """
    print("is_approx_eq call", file=sys.stderr)
    similarity = abs(env.vsa.similarity(x.data, y.data))
    print(f"\tsimilarity = {similarity}", file=sys.stderr)
    return similarity > floor


def is_nil[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](value: T, enc_env: EncodingEnvironment[T], floor: float = 0.2) -> bool:
    """Test whether or not a value is `nil`.

    Args
    -   value (VSA): A vector-symbol, which will be tested.
    -   enc_env (VSA): The encoding environment which contains the `nil` symbol.

    Returns:
        A boolean value about whether or not the value given is `nil`.
    """
    return is_approx_eq(value, enc_env.codebook["nil"], enc_env)


def is_false[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](value: T, enc_env: EncodingEnvironment[T], floor: float = 0.2) -> bool:
    """Test whether or not a value is `#f`.

    Args
    -   value (VSA): A vector-symbol, which will be tested.
    -   enc_env (VSA): The encoding environment which contains the `#f` symbol.

    Returns:
        A boolean value about whether or not the value given is `#f`.
    """
    return is_approx_eq(value, enc_env.codebook["#f"], enc_env)


def is_true[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](value: T, enc_env: EncodingEnvironment[T], floor: float = 0.2) -> bool:
    """Test whether or not a value is `#t`.

    Args
    -   value (VSA): A vector-symbol, which will be tested.
    -   enc_env (VSA): The encoding environment which contains the `#t` symbol.

    Returns:
        A boolean value about whether or not the value given is `#t`.
    """
    return is_approx_eq(value, enc_env.codebook["#t"], enc_env)


def check_atomic[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], floor: float = 0.2,) -> T:
    """Determine whether or not a value is an atomic value or a list."""
    print("check_atomic_call", file=sys.stderr)
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        return enc_env.codebook["#t"]

    close_to_phi = (
        enc_env.vsa.similarity(value.data, enc_env.codebook["__phi"].data)
        * enc_env.codebook["#f"].data
    )
    far_from_phi = (
        max(
            (2 * floor)
            - enc_env.vsa.similarity(value.data, enc_env.codebook["__phi"].data),
            0.0,
        )
        * enc_env.codebook["#t"].data
    )

    return enc_env.cleanup_memory.recall(
        enc_env.vsa.from_array(enc_env.vsa.bundle(close_to_phi, far_from_phi))
    )


def check_function[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], floor: float = 0.2,) -> T:
    """Determine whether or not a value is a function by checking for the
    special function symbol. This involves dereferencing the function as a
    semantic function pointer, and inspecting the contents at the memory slot.

    Args:
    -   expr (VSA): An expression that is treated as a semantic pointer.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   floor (float): Comparison floor, defaults to `0.2`.

    Returns:
        The vector-symbol for `#t` if the value contains the special function
        vector-symbol.
    """
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        return enc_env.codebook["#f"]

    close_to___func = (
        enc_env.vsa.similarity(value.data, enc_env.codebook["__func"].data)
        * enc_env.codebook["#t"].data
    )
    far_from___func = (
        max(
            (2 * floor)
            - enc_env.vsa.similarity(value.data, enc_env.codebook["__func"].data),
            0.0,
        )
        * enc_env.codebook["#f"].data
    )

    bundled_result = enc_env.vsa.bundle(close_to___func, far_from___func)
    cleanuped = enc_env.cleanup_memory.recall(enc_env.vsa.from_array(bundled_result))
    return cleanuped


def car[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Treat the expression as a semantic pointer and dereference it, returning
    the second element of the underlying tuple chunk.

    Args:
    -   expr (VSA): A vector-symbol semantic pointer.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnv): The encoding environment.

    Returns:
        The second element in the underlying tuple chunk.

    Raises:
        `InterpreterError`.
    """
    print("car call", file=sys.stderr)
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        raise InterpreterError("Segmentation fault in car hehe")

    slot = enc_env.vsa.unbind(value.data, enc_env.codebook["__lhs"].data)
    # return enc_env.cleanup_memory.recall(enc_env.vsa.from_array(slot))
    recalled_value = enc_env.cleanup_memory.recall(enc_env.vsa.from_array(slot))
    return recalled_value


def cdr[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Treat the expression as a semantic pointer and dereference it,
    returning the first element of an underlying tuple chunk.

    Args:
    -   expr (VSA): A vector-symbol semantic pointer.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The encoding environment.

    Returns:
        A vector-symbol corresponding to the first element.

    Raises:
        `InterpreterError`.
    """
    print("cdr call", file=sys.stderr)
    value = enc_env.associative_memory.deref(expr)
    if value is None:
        raise InterpreterError("Segmentation fault in cdr hehe")

    slot = enc_env.vsa.unbind(value.data, enc_env.codebook["__rhs"].data)
    recalled_value = enc_env.cleanup_memory.recall(enc_env.vsa.from_array(slot))
    return recalled_value


def cons[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Create a new tuple out of the operands. For more information about how
    this works, see `.language.interpreter.make_cons`.

    Args:
    -   rand (VSA): A vector-symbol list representing the arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A pointer to the tuple of the operand lists.

    Raises:
        `InterpreterError`.
    """
    car_ = car(rand, enc_env, eval_env)
    ecar = evaluate(car_, enc_env, eval_env)

    cdr_ = cdr(rand, enc_env, eval_env)
    cadr = car(cdr_, enc_env, eval_env)
    ecadr = evaluate(cadr, enc_env, eval_env)

    ptr = make_cons(ecar, ecadr, enc_env)
    return ptr


def eq[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Compare two values to see if they are the same.

    Args:
    -   rand (VSA): A vector-symbol list representing the arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A pointer to the tuple of the operand lists.

    Raises:
        `InterpreterError`.
    """
    raise Exception("TODO")


def atom[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Test whether the operand is an atomic value or not.

    Args:
    -   rand (VSA): A vector-symbol list representing the arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        True if the value is indeed atomic, false otherwise.

    Raises:
        `InterpreterError`.
    """
    arg = car(rand, enc_env, eval_env)
    return check_atomic(arg, enc_env)


def if_[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Conditional evaluation of the branches. Tests the condition as to
    whether or not it is true, and the exectutes the second argument if the
    result is true. Otherwise, executes the third argument.

    Args:
    -   rand (VSA): A vector-symbol list representing the arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        The result of the branch taken.

    Raises:
        `InterpreterError`.
    """

    # expected syntax:
    # (if (condition (consequent (alternate nil))))

    condition = car(rand, enc_env, eval_env)
    branches = cdr(rand, enc_env, eval_env)
    if is_nil(branches, enc_env):
        raise InterpreterError(
            "ERROR: Conditional expression requires `CONSEQUENT` and `ALTERNATE`, see documentation: [https://conservatory.scheme.org/schemers/Documents/Standards/R5RS/HTML/]"
        )

    consequent = car(branches, enc_env, eval_env)
    final_branch = cdr(branches, enc_env, eval_env)
    alternate = car(final_branch, enc_env, eval_env)

    if is_nil(condition, enc_env) or is_false(condition, enc_env):
        return evaluate(alternate, enc_env, eval_env)
    else:
        return evaluate(consequent, enc_env, eval_env)

    
# TODO: add support for quoting
def quote[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Capture the following code as raw syntax which can be later evaluated.
    Syntax which is quoted can be later interpreted using the function 
    `unquote`.

    The format of a quote chunk is a pointer to some raw syntax which is
    annotated as being quoted.

    Args:
    -   rand (VSA): A vector-symbol representing some syntax.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A quote chunk.
    """
    raise Exception("TODO")


def add[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Addition of operands. 
    
    The form of the function depends on `.encoding.IntegerEncodingScheme`. If
    it is `.encoding.IntegerEncodingScheme.ListIntegers`, then we will use an
    encoding scheme reminiscent of the [Peano definition](https://en.wikipedia.org/wiki/Peano_axioms),
    or, the [Church encoding](https://en.wikipedia.org/wiki/Church_encoding) of
    the natural numbers.
    
    If the the encoding scheme is `.encoding.IntegerEncodingScheme.RHCIntegers`,
    then we will use [Residue Hyperdimensional Computing](https://direct.mit.edu/neco/article/37/1/1/125267/Computing-With-Residue-Numbers-in-High-Dimensional),
    or RHC.

    Args:
    -   rand (VSA): A vector-symbol of arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        The sum of the two numbers, using the encoding scheme provided.
    """
    raise Exception("TODO")


def sub[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Subtraction of operands. 
    
    The form of the function depends on `.encoding.IntegerEncodingScheme`. If
    it is `.encoding.IntegerEncodingScheme.ListIntegers`, then we will use an
    encoding scheme reminiscent of the [Peano definition](https://en.wikipedia.org/wiki/Peano_axioms),
    or, the [Church encoding](https://en.wikipedia.org/wiki/Church_encoding) of
    the natural numbers.
    
    If the the encoding scheme is `.encoding.IntegerEncodingScheme.RHCIntegers`,
    then we will use [Residue Hyperdimensional Computing](https://direct.mit.edu/neco/article/37/1/1/125267/Computing-With-Residue-Numbers-in-High-Dimensional),
    or RHC.

    Args:
    -   rand (VSA): A vector-symbol of arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        The difference of the two numbers, using the encoding scheme provided.
    """
    raise Exception("TODO")


def mul[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Product of operands. 
    
    The form of the function depends on `.encoding.IntegerEncodingScheme`. If
    it is `.encoding.IntegerEncodingScheme.ListIntegers`, then we will use an
    encoding scheme reminiscent of the [Peano definition](https://en.wikipedia.org/wiki/Peano_axioms),
    or, the [Church encoding](https://en.wikipedia.org/wiki/Church_encoding) of
    the natural numbers.
    
    If the the encoding scheme is `.encoding.IntegerEncodingScheme.RHCIntegers`,
    then we will use [Residue Hyperdimensional Computing](https://direct.mit.edu/neco/article/37/1/1/125267/Computing-With-Residue-Numbers-in-High-Dimensional),
    or RHC.

    Args:
    -   rand (VSA): A vector-symbol of arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        The product of the two numbers, using the encoding scheme provided.
    """
    raise Exception("TODO")


def div[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Division of operands. 
    
    The form of the function depends on `.encoding.IntegerEncodingScheme`. If
    it is `.encoding.IntegerEncodingScheme.ListIntegers`, then we will use an
    encoding scheme reminiscent of the [Peano definition](https://en.wikipedia.org/wiki/Peano_axioms),
    or, the [Church encoding](https://en.wikipedia.org/wiki/Church_encoding) of
    the natural numbers.
    
    If the the encoding scheme is `.encoding.IntegerEncodingScheme.RHCIntegers`,
    then we will use [Residue Hyperdimensional Computing](https://direct.mit.edu/neco/article/37/1/1/125267/Computing-With-Residue-Numbers-in-High-Dimensional),
    or RHC.

    Args:
    -   rand (VSA): A vector-symbol of arguments.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        The fraction of the two numbers, using the encoding scheme provided.
    """
    raise Exception("TODO")


def make_function_pointer[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    args: T, body: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]
) -> T:
    """Allocate a semantic function pointer for the arguments and the body.

    The structure of the chunk to which the semantic pointer points to is
    similar to the structure of a tuple chunk. We bind together
    `args` with the reserved symbol, `__args`
    (see `language.encoding.EncodingEnvironment.initial_codebook` for the
    other reserved symbols). We the bundle this with the function body,
    where the function body is binded to the symbol `__body`. Finally,
    the symbols are bundled with the `__func` function symbol.

    Args:
    -   args (VSA): A vector-symbol representing the arguments as a list.
    -   body (VSA): A vector-symbol representing the unevaluated function body.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A semantic pointer in the encoding environment that points to a
        function body chunk.
    """
    print("make_function_pointer call", file=sys.stderr)
    tagged_args = enc_env.vsa.bind(args.data, enc_env.codebook["__args"].data)
    tagged_body = enc_env.vsa.bind(body.data, enc_env.codebook["__body"].data)
    function_tag = enc_env.codebook["__func"].data
    chunk = enc_env.vsa.bundle(
        tagged_args, enc_env.vsa.bundle(tagged_body, function_tag)
    )
    ptr = enc_env.associative_memory.alloc(enc_env.vsa.from_array(chunk))
    return ptr


def evaluate_lambda[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    function_body: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]
) -> T:
    """Evaluate a lambda expression, converting it into a semantic function
    pointer. For more information about the layout of a semantic function
    pointer, see `.interpreter.make_function_pointer`.

    Args:
    -   function_body (VSA): The function body of the lambda.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A semantic function pointer in `enc_env.associative_memory` which
        points to a function chunk.
    """
    print("evaluate_lambda call", file=sys.stderr)
    args = car(function_body, enc_env, eval_env)
    body = cdr(function_body, enc_env, eval_env)
    return make_function_pointer(args, body, enc_env, eval_env)


def evaluate_define[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](define_body: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Evaluate a definition, associates a value to a name in `eval_env`.

    Args:
    -   define_body (VSA): A vector-symbol representing the definition body.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A `nil` value, as per the specification.

    Raises:
        `InterpreterError`.
    """
    print("evaluate_define call", file=sys.stderr)
    name = car(define_body, enc_env, eval_env)
    body = car(cdr(define_body, enc_env, eval_env), enc_env, eval_env)
    eval_env.define_mem.associate(name, body)
    return enc_env.codebook["nil"]


def evaluate_application[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    rator: T, rand: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]
) -> T:
    """Evaluate an application of either a built-in function or a user-defined
    semantic function pointer.

    Args:
    -   rator (VSA): The ope*rator* of the application.
    -   rand (VSA): A, possibly empty, list of arguments to pass to the operator.
            stands for ope*rand*.
    -   enc_env (EncodingEnvironment): The encoding environment.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A vector symbol which is the result of the application of the operator
        to the operands.

    Raises:
        `InterpreterError`.
    """
    print("evaluate_application call", file=sys.stderr)
    operator_v: T
    if (eval_env.locals_ is not None) and (
        mayb_local_val := eval_env.locals_.deref(rator)
    ) is not None:
        operator_v = mayb_local_val
    elif (mayb_define_val := eval_env.define_mem.deref(rator)) is not None:
        operator_v = mayb_define_val
    else:
        operator_v = rator
    operator_v = enc_env.cleanup_memory.recall(operator_v)

    if is_approx_eq(operator_v, enc_env.codebook["car"], enc_env):
        print("closest to `car`", file=sys.stderr)
        # we evaluate the result of the operation, as we are not doing
        # syntactic manipulation.
        car_ = car(rand, enc_env, eval_env)

        if not is_nil(cdr(rand, enc_env, eval_env), enc_env):
            print(
                "WARNING: car takes only one argument, ignoring the rest!",
                file=sys.stderr,
            )

        value = evaluate(car_, enc_env, eval_env)
        return car(value, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["cdr"], enc_env):
        print("closest to `cdr`", file=sys.stderr)
        # ditto here
        car_ = car(rand, enc_env, eval_env)

        if not is_nil(cdr(rand, enc_env, eval_env), enc_env):
            print(
                "WARNING: cdr takes only one argument, ignoring the rest!",
                file=sys.stderr,
            )

        value = evaluate(car_, enc_env, eval_env)
        return cdr(value, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["cons"], enc_env):
        print("closest to `cons`", file=sys.stderr)
        return cons(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["eq?"], enc_env):
        print("closest to `eq?`", file=sys.stderr)
        return eq(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["atom?"], enc_env):
        print("closest to `atom?`", file=sys.stderr)
        return atom(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["if"], enc_env):
        print("closest to `if`", file=sys.stderr)
        return if_(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["quote"], enc_env):
        print("closest to `quote`", file=sys.stderr)
        return quote(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["+"], enc_env):
        print("closest to `+`", file=sys.stderr)
        return add(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["-"], enc_env):
        print("closest to `-`", file=sys.stderr)
        return sub(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["*"], enc_env):
        print("closest to `*`", file=sys.stderr)
        return mul(rand, enc_env, eval_env)
    elif is_approx_eq(operator_v, enc_env.codebook["/"], enc_env):
        print("closest to `/`", file=sys.stderr)
        return div(rand, enc_env, eval_env)
    elif is_approx_eq(
        check_function(operator_v, enc_env), enc_env.codebook["#t"], enc_env
    ):
        print("is a function!", file=sys.stderr)
    else:
        eval_rand = evaluate(rand, enc_env, eval_env)
        return make_cons(operator_v, eval_rand, enc_env)

    raise Exception("TODO")


def evaluate[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> T:
    """Evalute an encoded vector-symbol.

    Args:
    -   expr (VSA): A vector-symbol representing an expression in the lisp.
    -   enc_env (EncodingEnv): The encoding environment used in encoding.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A vector-symbol representing the result of the evaluation.

    Raises:
        `InterpreterError`.
    """
    print("evaluate call", file=sys.stderr)
    if is_approx_eq(check_atomic(expr, enc_env), enc_env.codebook["#t"], enc_env):
        print("\texpression is an atom", file=sys.stderr)

        # TODO: fix this to check first locals if available, then
        # fall through to define
        if eval_env.locals_ is None:
            res = eval_env.define_mem.deref(expr)
            if res is None:
                print("value not found in `define_mem`", file=sys.stderr)
                return expr
            return res
        else:
            res = eval_env.locals_.deref(expr)
            if res is None:
                print("value not found in `locals_`", file=sys.stderr)
                return expr
            return res

    print("\texpression is non-atomic", file=sys.stderr)
    head = car(expr, enc_env, eval_env)
    tail = cdr(expr, enc_env, eval_env)

    if is_approx_eq(head, enc_env.codebook["lambda"], enc_env):
        return evaluate_lambda(tail, enc_env, eval_env)
    elif is_approx_eq(head, enc_env.codebook["define"], enc_env):
        return evaluate_define(tail, enc_env, eval_env)
    else:
        return evaluate_application(head, tail, enc_env, eval_env)

    raise Exception("TODO")


def closest[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](value: T, enc_env: EncodingEnvironment[T]) -> str:
    """Return the closest key for which `value` matches too in the encoding
    environment's codebook.

    Args:
    -   value (VSA): The vector-symbol to search the coding environment for.
    -   enc_env (EncodingEnvironment): The encoding environment.

    Returns:
        A string corresponding to the closest key, the string "NONE" otherwise.
    """
    max_sim = 0.0
    max_word = "NONE"
    for word in enc_env.codebook.keys():
        sim = enc_env.vsa.similarity(enc_env.codebook[word].data, value.data)
        if sim > max_sim:
            max_sim = sim
            max_word = word
    return max_word


def decode[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](expr: T, enc_env: EncodingEnvironment[T], eval_env: EvalEnvironment[T]) -> (
    str | list[Any] | tuple[Any, ...]
):
    """Decode a vector symbol to a Python object.

    Args:
    -   expr (VSA): A vector-symbol, which is the product of `encode`.
    -   enc_env (EncodingEnvironment): The encoding environment used to create
            `expr`.
    -   eval_env (EvalEnvironment): The evaluation environment.

    Returns:
        A Python object corresponding to the encoded value.

    Raises:
        `InterpreterError`.
    """
    if is_approx_eq(check_atomic(expr, enc_env), enc_env.codebook["#t"], enc_env):
        return closest(expr, enc_env)
    else:
        return ""


def interpret[T: (
    VSA[np.complex128],
    VSA[np.float64],
)](
    src: str, vsa: type[T], dim: int, integer_encoding_scheme: IntegerEncodingScheme
) -> None:
    """Interpret a source-level string of the language.

    Args:
    -   src (str): The source level representation of the language.
    -   vsa (type[VSA]): The VSA you want to use in interpretation.
    -   dim (int): The dimensionality of the interpretation.
    -   integer_encoding_scheme (IntegerEncodingScheme): The integer encoding
            scheme used in encoding and interpretation.

    Returns:
        Another vector-symbol that is the result of the evaluation.

    Raises:
        `InterpreterError`.
    """
    tokens = lex(src)
    intr_rep = parse(tokens)

    encoding_env = EncodingEnvironment(
        vsa=vsa, dim=dim, integer_encoding_scheme=integer_encoding_scheme
    )
    encoded_rep = encode(intr_rep, encoding_env)

    eval_env = EvalEnvironment(
        AssociativeMemory(encoding_env.vsa, encoding_env.dim), None
    )

    result = evaluate(encoded_rep, encoding_env, eval_env)
    decoded_result = decode(result, encoding_env, eval_env)
    print(decoded_result)
