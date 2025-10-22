"""Module containing the clean-up memory and associative memory."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import numpy as np

from .vsa import cosine_similarity, fhrr_random


class CleanupMemory(eqx.Module):
    """Simple clean-up memory that uses a Hopfield network in the complex domain.

    Attributes:
    -   num_traces (int): The number of traces currently in the memory.
    -   W (jax.Array): The weight matrix of the Hopfield network.
    -   dtype (jnp.dtype): The data type of the memory.
    """

    num_traces: int
    W: jax.Array
    dtype: jnp.dtype = jnp.complex64

    @classmethod
    def init(
        cls,
        init_traces: int,
        dim: int,
        dtype: jnp.dtype = jnp.complex64,
    ) -> "CleanupMemory":
        num_traces = 0
        W = jnp.zeros((init_traces, dim), dtype=dtype)
        dtype = dtype

        return cls(num_traces=num_traces, W=W, dtype=dtype)

    def memorize(self, trace: jax.Array) -> tuple[jax.Array, "CleanupMemory"]:
        """Memorize a trace into cleanup memory, and return the trace
        and updated memory.

        Args:
        -   trace (jax.Array): A vector-symbol.

        Returns:
            The vector symbol `trace` passed, and the updated cleanup memory.
        """
        N, _ = self.W.shape
        num_traces = self.num_traces
        W = self.W
        if num_traces >= N:
            W = jnp.concatenate(
                [self.W, jnp.zeros_like(self.W, dtype=self.dtype)], axis=0
            )
        W = W.at[num_traces, :].set(trace)
        num_traces += 1
        return trace, CleanupMemory(
            num_traces=num_traces,
            W=W,
            dtype=self.dtype,
        )

    def recall(self, query: jax.Array) -> jax.Array:
        """Recall a query pattern based on the memory network.

        Args:
        -   query (jax.Array): A vector-symbol to recall.

        Returns:
            The recalled trace.
        """
        sims = cosine_similarity(query, self.W)
        return self.W[jnp.argmax(jnp.abs(sims))]


class AssociativeMemory(eqx.Module):
    """Associative memory class for storing semantic pointers.

    Attributes:
    -   address_matrix (jax.Array): The address matrix.
    -   content_matrix (jax.Array): The content matrix.
    -   theta (float): The theta parameter, or tolerance, for the associative memory recall.
    -   dtype (jnp.dtype): The data type of the memory.
    """

    address_matrix: jax.Array
    content_matrix: jax.Array
    num_traces: int
    theta: float = 0.2
    dtype: jnp.dtype = jnp.complex64

    @classmethod
    def init(
        cls,
        init_traces: int,
        dim: int,
        dtype: jnp.dtype = jnp.complex64,
        theta: float = 0.2,
    ) -> "AssociativeMemory":
        return cls(
            address_matrix=jnp.zeros((init_traces, dim), dtype=dtype),
            content_matrix=jnp.zeros((init_traces, dim), dtype=dtype),
            theta=theta,
            dtype=dtype,
            num_traces=0,
        )

    def alloc(
        self, key: jax.Array, trace: jax.Array
    ) -> tuple[jax.Array, "AssociativeMemory"]:
        """Allocate a semantic pointer for the trace.

        Args:
        -   key (jax.Array): A random key.
        -   trace (jax.Array): A vector-symbol.

        Returns:
            The semantic pointer, and the updated associative memory.
        """
        ptr = fhrr_random(key, num_vectors=1, dim=self.dim)
        return self.associate(ptr, trace)

    def associate(
        self, address: jax.Array, content: jax.Array
    ) -> tuple[jax.Array, "AssociativeMemory"]:
        """Directly associate an arbitrary address with a content pattern.

        Args:
        -   address (jax.Array): An address.
        -   content (jax.Array): A content pattern.

        Returns:
            The updated associative memory.
        """
        if self.num_traces >= self.address_matrix.shape[0]:
            address_matrix = jnp.concatenate(
                self.address_matrix,
                jnp.zeros_like(self.address_matrix, dtype=self.dtype),
                axis=0,
            )
            content_matrix = jnp.concatenate(
                self.content_matrix,
                jnp.zeros_like(self.content_matrix, dtype=self.dtype),
                axis=0,
            )
        else:
            address_matrix = self.address_matrix
            content_matrix = self.content_matrix

        address_matrix = address_matrix.at[self.num_traces].set(address)
        content_matrix = content_matrix.at[self.num_traces].set(content)

        return address, AssociativeMemory(
            dim=self.dim,
            address_matrix=address_matrix,
            content_matrix=content_matrix,
            theta=self.theta,
            dtype=self.dtype,
            num_traces=self.num_traces + 1,
        )

    def deref(self, ptr: jax.Array) -> jax.Array | None:
        """Dereference a semantic pointer, returning the content pattern.

        Args:
        -   ptr (jax.Array): A semantic pointer.

        Returns:
            The content pattern associated with the semantic pointer.
        """
        if len(self.address_matrix) == 0:
            return None

        sims = cosine_similarity(ptr, self.address_matrix)
        nearest_index = jnp.argmax(jnp.abs(sims))
        if sims[nearest_index] < self.theta:
            return None
        else:
            return self.content_matrix[nearest_index]
