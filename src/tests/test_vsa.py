import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from language.vsa import fhrr_random, cosine_similarity, bind, bundle, inverse, unbind


# Test whether the FHRR random function generates the correct number of vectors
def test_fhrr_random() -> None:
    dim = 1000
    num_vectors = 100
    key = jr.PRNGKey(0)

    vectors = fhrr_random(key, num_vectors=num_vectors, dim=dim)

    assert vectors.shape == (num_vectors, dim)
    assert vectors.dtype == jnp.complex64


# Test whethere FHRR random generation works correctly. We expect
# that two random FHRR vectors are near orthogonal, and thus,
# their cosine similarity should be close to 0.
@pytest.mark.skip(reason="Testing takes too long")
def test_fhrr_random_cosine_similarity() -> None:
    dim = 1000
    num_vectors = 100
    key = jr.PRNGKey(0)

    vectors = fhrr_random(key, num_vectors=num_vectors, dim=dim)

    for i in range(num_vectors):
        for j in range(num_vectors):
            if i == j:
                assert jnp.isclose(cosine_similarity(vectors[i], vectors[j]), 1.0)
            else:
                assert jnp.isclose(
                    cosine_similarity(vectors[i], vectors[j]), 0.0, atol=0.2
                )


# FHRR binding of two vectors should create a new vector which is near-orthogonal
# with respect to the original vectors.
def test_binding() -> None:
    dim = 1000
    key = jr.PRNGKey(0)
    x = fhrr_random(key, num_vectors=1, dim=dim)
    y = fhrr_random(key, num_vectors=1, dim=dim)

    bound = bind(x, y)

    assert jnp.isclose(cosine_similarity(x, bound), 0.0, atol=0.2)
    assert jnp.isclose(cosine_similarity(y, bound), 0.0, atol=0.2)
    assert jnp.isclose(cosine_similarity(bound, x), 0.0, atol=0.2)
    assert jnp.isclose(cosine_similarity(bound, y), 0.0, atol=0.2)


# FHRR bundling, or superposition, of two vectors should create a new vector which
# is near parallel with respect to the original vectors.
def test_bundling() -> None:
    dim = 1000
    keyx, keyy = jr.split(jr.PRNGKey(0), 2)
    x = fhrr_random(keyx, num_vectors=1, dim=dim)
    y = fhrr_random(keyy, num_vectors=1, dim=dim)

    bound = bundle(x, y)

    assert jnp.isclose(jnp.abs(cosine_similarity(x, bound)), 1.0, atol=0.3)
    assert jnp.isclose(jnp.abs(cosine_similarity(y, bound)), 1.0, atol=0.3)
    assert jnp.isclose(jnp.abs(cosine_similarity(bound, x)), 1.0, atol=0.3)
    assert jnp.isclose(jnp.abs(cosine_similarity(bound, y)), 1.0, atol=0.3)


# FHRR inverse of a vector should create a new vector which is nearly orthogonal
# with respect to the original vector.
def test_inverse() -> None:
    dim = 1000
    key = jr.PRNGKey(0)
    x = fhrr_random(key, num_vectors=1, dim=dim)

    bound = inverse(x)

    assert jnp.isclose(cosine_similarity(x, bound), 0.0, atol=0.2)
    assert jnp.isclose(cosine_similarity(bound, x), 0.0, atol=0.2)


# FHRR unbinding of the binding of two vectors should create a new vector
# which is nearly parallel with respect to the other vector.
def test_unbinding() -> None:
    dim = 1000
    keyx, keyy = jr.split(jr.PRNGKey(0), 2)
    x = fhrr_random(keyx, num_vectors=1, dim=dim)
    y = fhrr_random(keyy, num_vectors=1, dim=dim)

    bound = bind(x, y)
    unbound = unbind(bound, x)

    assert jnp.isclose(jnp.abs(cosine_similarity(y, unbound)), 1.0, atol=0.3)
    assert jnp.isclose(jnp.abs(cosine_similarity(unbound, y)), 1.0, atol=0.3)
    assert jnp.isclose(cosine_similarity(x, unbound), 0.0, atol=0.3)
    assert jnp.isclose(cosine_similarity(unbound, x), 0.0, atol=0.3)

    unbound = unbind(bound, y)

    assert jnp.isclose(cosine_similarity(y, unbound), 0.0, atol=0.3)
    assert jnp.isclose(cosine_similarity(unbound, y), 0.0, atol=0.3)
    assert jnp.isclose(jnp.abs(cosine_similarity(x, unbound)), 1.0, atol=0.3)
    assert jnp.isclose(jnp.abs(cosine_similarity(unbound, x)), 1.0, atol=0.3)
