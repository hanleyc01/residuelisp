"""Vector-symbolic architecture operations. In particular, FHRR and RHC operations.

Implementation from [`torchhd`](https://torchhd.readthedocs.io/en/stable/_modules/torchhd/tensors/fhrr.html#FHRRTensor).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import jax.tree_util as jtu


def fhrr_random(
    key: jax.Array, num_vectors: int, dim: int, dtype: jnp.dtype = jnp.complex64
) -> jax.Array:
    """Generate a random FHRR vector-symbol in the Fourier domain.

    Args:
    -   key (jax.Array): A random key.
    -   num_vectors (int): The number of vectors to generate.
    -   dim (int): The dimensionality of the vectors.
    -   dtype (jnp.dtype): The data type of the vectors. Defaults to `jnp.complex64`.

    Returns:
        A random vector-symbol in the Fourier domain.
    """
    x = jr.uniform(
        key,
        shape=(num_vectors, dim),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    x_cos = jnp.cos(x)
    x_sin = jnp.sin(x)
    # Assign the real part to `x_cos` and the imaginary part to `x_sin`
    result = x_cos + 1j * x_sin
    return result


def bind(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two vectors in the Fourier domain.

    Args:
    -   x (jax.Array): A vector in the Fourier domain.
    -   y (jax.Array): A vector in the Fourier domain.

    Returns:
        The binding of `x` and `y`.
    """
    return x * y


def bundle(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bundle two vectors in the Fourier domain.

    Args:
    -   x (jax.Array): A vector in the Fourier domain.
    -   y (jax.Array): A vector in the Fourier domain.

    Returns:
        The superposition of `x` and `y`.
    """
    return x + y


def dot_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute the dot product similarity between two vectors in the Fourier domain.

    Args:
    -   x (jax.Array): A vector in the Fourier domain.
    -   y (jax.Array): A vector in the Fourier domain.

    Returns:
        The similarity between `x` and `y`.
    """
    if len(y.shape) >= 2:
        y = y.transpose(-1, -2)
    return jnp.real(x.dot(jnp.conj(y)))


def cosine_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute the cosine similarity between two vectors in the Fourier domain.

    Args:
    -   x (jax.Array): A vector in the Fourier domain.
    -   y (jax.Array): A vector in the Fourier domain.

    Returns:
        The similarity between `x` and `y`.
    """
    self_dot = jnp.sum(x * x.conj(), axis=-1)
    self_mag = jnp.sqrt(self_dot)
    other_dot = jnp.sum(y * y.conj(), axis=-1)
    other_mag = jnp.sqrt(other_dot)

    if len(x.shape) >= 2:
        magnitude = jnp.expand_dims(self_mag, axis=-1) * jnp.expand_dims(
            other_mag, axis=-2
        )
    else:
        magnitude = self_mag * other_mag

    magnitude = jnp.clip(magnitude.real, min=1e-6)
    return (dot_similarity(x, y) / magnitude).real


def inverse(x: jax.Array) -> jax.Array:
    """Compute the inverse of a vector in the Fourier domain.

    Args:
    -   x (jax.Array): A vector in the Fourier domain.

    Returns:
        The inverse of `x`.
    """
    return jnp.conj(x)


def unbind(x: jax.Array, y: jax.Array) -> jax.Array:
    """Unbind two vectors in the Fourier domain.

    Args:
    -   x (jax.Array): A vector in the Fourier domain.
    -   y (jax.Array): A vector in the Fourier domain.

    Returns:
        The unbinding of `x` and `y`.
    """
    return bind(x, inverse(y))
