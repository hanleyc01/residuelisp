import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from language.memory import CleanupMemory, AssociativeMemory
from language.vsa import cosine_similarity, fhrr_random


# Test whether the memory is initialized correctly
def test_cleanup_memory_init() -> None:
    dim = 1000
    init_traces = 100
    memory = CleanupMemory.init(init_traces=init_traces, dim=dim)

    assert memory.num_traces == 0
    assert memory.W.shape == (init_traces, dim)
    assert memory.dtype == jnp.complex64


# Testing whether memorizing works correctly
def test_cleanup_memory_recall_memorize() -> None:
    dim = 1000
    init_traces = 100
    key = jr.PRNGKey(0)

    memory = CleanupMemory.init(init_traces=init_traces, dim=dim)

    patterns = jr.uniform(key, shape=(init_traces, dim), minval=-1, maxval=1)
    for i in range(init_traces):
        _, memory = memory.memorize(patterns[i])
        assert memory.num_traces == i + 1

    assert jnp.allclose(memory.W, patterns)


# Testing whether recall works correctly
def test_cleanup_memory_recall() -> None:
    dim = 1000
    init_traces = 100
    key = jr.PRNGKey(0)

    patterns = fhrr_random(key, num_vectors=init_traces, dim=dim)
    memory = CleanupMemory(init_traces, patterns, dtype=jnp.complex64)

    for i in range(init_traces):
        trace = memory.recall(patterns[i])
        assert jnp.isclose(
            jnp.abs(cosine_similarity(trace, patterns[i])), 1.0, atol=0.2
        )


# Test associative memory initialization
def test_associative_memory_init() -> None:
    dim = 1000
    init_traces = 100
    memory = AssociativeMemory.init(
        init_traces=init_traces, dim=dim, dtype=jnp.complex64
    )

    assert memory.num_traces == 0
    assert memory.address_matrix.shape == (init_traces, dim)
    assert memory.content_matrix.shape == (init_traces, dim)
    assert memory.dtype == jnp.complex64


# Test associative memory association
def associative_memory_alloc() -> None:
    dim = 1000
    init_traces = 100
    key = jr.PRNGKey(0)

    memory = AssociativeMemory.init(
        init_traces=init_traces, dim=dim, dtype=jnp.complex64
    )

    for i in range(init_traces):
        ptr, memory = memory.alloc(key, fhrr_random(key, num_vectors=1, dim=dim))
        assert memory.num_traces == i + 1

    assert jnp.allclose(memory.address_matrix, ptr)


# Test associative memory direct association
def associative_memory_associate() -> None:
    dim = 1000
    init_traces = 100
    key_address = jr.PRNGKey(0)
    key_content = jr.PRNGKey(1)

    memory = AssociativeMemory.init(
        init_traces=init_traces, dim=dim, dtype=jnp.complex64
    )

    for i in range(init_traces):
        ptr, memory = memory.associate(
            address=fhrr_random(key_address, num_vectors=1, dim=dim),
            content=fhrr_random(key_content, num_vectors=1, dim=dim),
        )
        assert memory.num_traces == i + 1


# Test associative memory dereferencing
def associative_memory_deref() -> None:
    dim = 1000
    init_traces = 100
    key1 = jr.PRNGKey(0)
    key2 = jr.PRNGKey(1)

    memory = AssociativeMemory.init(
        init_traces=init_traces, dim=dim, dtype=jnp.complex64
    )

    trace = fhrr_random(key2, num_vectors=1, dim=dim)
    ptr, memory = memory.alloc(key1, trace)
    assert memory.deref(ptr) is not None
    assert jnp.allclose(
        memory.deref(ptr),
    )
