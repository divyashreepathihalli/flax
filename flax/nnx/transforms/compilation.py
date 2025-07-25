# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pytype: skip-file
from __future__ import annotations

import dataclasses
import functools
import typing as tp

import jax
import jax.experimental
import jax.experimental.shard_map
from jax.sharding import AbstractMesh, Mesh, PartitionSpec

from flax.nnx import (
  extract,
  filterlib,
  graph,
  statelib,
  variablelib,
)
from flax.typing import Missing

F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
Specs = tp.Any
AxisName = tp.Hashable

# -------------------------------
# jit
# -------------------------------


class StateSharding(extract.PrefixMapping):
  def __init__(
    self,
    filter_sharding: statelib.State
    | tp.Mapping[filterlib.Filter, tp.Any]
    | tp.Iterable[tuple[filterlib.Filter, tp.Any]],
    /,
  ):
    if isinstance(filter_sharding, statelib.State):
      filter_sharding = statelib.create_path_filters(filter_sharding)  # type: ignore

    iterable = tuple(
      filter_sharding.items()
      if isinstance(filter_sharding, tp.Mapping)
      else filter_sharding
    )
    self._filters = tuple(filter for filter, _ in iterable)
    self._shardings = tuple(axis for _, axis in iterable)

  @property
  def filters(self) -> tuple[filterlib.Filter, ...]:
    return self._filters

  @property
  def shardings(self) -> tuple[tp.Any, ...]:
    return self._shardings

  def map_prefix(
    self, path: variablelib.PathParts, variable: variablelib.Variable
  ) -> tp.Any:
    for filter, sharding in zip(self.filters, self.shardings):
      predicate = filterlib.to_predicate(filter)
      if predicate(path, variable):
        return sharding
    raise ValueError(f'No axis found for {path=}, {variable=}')

  def __repr__(self):
    return f'StateSharding({dict(zip(self.filters, self.shardings))})'

  def __eq__(self, other):
    return (
      isinstance(other, StateSharding)
      and self.filters == other.filters
      and self.shardings == other.shardings
    )

  def __hash__(self):
    return hash((self.filters, self.shardings))


def _jit_split_fn(ctx: graph.SplitContext, path, prefix, x):
  if isinstance(prefix, StateSharding):
    graphdef, *states = ctx.flatten(x, *prefix.filters)
    return extract.NodeStates.from_split(graphdef, *states, metadata=prefix)
  return extract.NodeStates.from_split(*ctx.flatten(x, with_paths=False))


def _jit_merge_fn(ctx: graph.MergeContext, path, prefix, leaf) -> tp.Any:
  if not isinstance(leaf, extract.NodeStates):
    raise ValueError(f'Expected TreeNode, got {type(leaf)} at path {path}')
  return ctx.unflatten(leaf.graphdef, *leaf.states)


@dataclasses.dataclass(eq=False)
class JitFn:
  f: tp.Callable[..., tp.Any]
  in_shardings: tp.Any
  out_shardings: tp.Any
  kwarg_shardings: tp.Any
  ctxtag: tp.Hashable

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args, **pure_kwargs):
    args, kwargs = extract.from_tree(
      (pure_args, pure_kwargs),
      merge_fn=_jit_merge_fn,
      ctxtag=self.ctxtag,
      is_inner=True,
    )

    out = self.f(*args, **kwargs)

    args_out, kwargs_out = extract.clear_non_graph_nodes((args, kwargs))
    pure_args_out, pure_kwargs_out, pure_out = extract.to_tree(
      (args_out, kwargs_out, out),
      prefix=(self.in_shardings, self.kwarg_shardings, self.out_shardings),
      ctxtag=self.ctxtag,
      split_fn=_jit_split_fn,
    )

    return pure_args_out, pure_kwargs_out, pure_out


@tp.overload
def jit(
  *,
  in_shardings: tp.Any = None,
  out_shardings: tp.Any = None,
  static_argnums: int | tp.Sequence[int] | None = None,
  static_argnames: str | tp.Iterable[str] | None = None,
  donate_argnums: int | tp.Sequence[int] | None = None,
  donate_argnames: str | tp.Iterable[str] | None = None,
  keep_unused: bool = False,
  device: tp.Optional[jax.Device] = None,
  backend: tp.Optional[str] = None,
  inline: bool = False,
  abstracted_axes: tp.Optional[tp.Any] = None,
) -> tp.Callable[[tp.Callable[..., tp.Any]], JitWrapped]: ...
@tp.overload
def jit(
  fun: tp.Callable[..., tp.Any],
  *,
  in_shardings: tp.Any = None,
  out_shardings: tp.Any = None,
  static_argnums: int | tp.Sequence[int] | None = None,
  static_argnames: str | tp.Iterable[str] | None = None,
  donate_argnums: int | tp.Sequence[int] | None = None,
  donate_argnames: str | tp.Iterable[str] | None = None,
  keep_unused: bool = False,
  device: tp.Optional[jax.Device] = None,
  backend: tp.Optional[str] = None,
  inline: bool = False,
  abstracted_axes: tp.Optional[tp.Any] = None,
) -> JitWrapped: ...
def jit(
  fun: tp.Callable[..., tp.Any] | type[Missing] = Missing,
  *,
  in_shardings: tp.Any = None,
  out_shardings: tp.Any = None,
  static_argnums: int | tp.Sequence[int] | None = None,
  static_argnames: str | tp.Iterable[str] | None = None,
  donate_argnums: int | tp.Sequence[int] | None = None,
  donate_argnames: str | tp.Iterable[str] | None = None,
  keep_unused: bool = False,
  device: tp.Optional[jax.Device] = None,
  backend: tp.Optional[str] = None,
  inline: bool = False,
  abstracted_axes: tp.Optional[tp.Any] = None,
) -> JitWrapped | tp.Callable[[tp.Callable[..., tp.Any]], JitWrapped]:
  """
  Lifted version of ``jax.jit`` that can handle Modules / graph nodes as
  arguments.

  Args:
    fun: Function to be jitted. ``fun`` should be a pure function, as
      side-effects may only be executed once.

      The arguments and return value of ``fun`` should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.

      JAX keeps a weak reference to ``fun`` for use as a compilation cache key,
      so the object ``fun`` must be weakly-referenceable. Most :class:`Callable`
      objects will already satisfy this requirement.
    in_shardings: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The ``in_shardings`` argument is optional. JAX will infer the shardings
      from the input :py:class:`jax.Array`'s and defaults to replicating the input
      if the sharding cannot be inferred.

      The valid resource assignment specifications are:
        - :py:class:`Sharding`, which will decide how the value
            will be partitioned. With this, using a mesh context manager is not
            required.
        - :py:obj:`None`, will give JAX the freedom to choose whatever sharding
          it wants.
          For in_shardings, JAX will mark is as replicated but this behavior
          can change in the future.
          For out_shardings, we will rely on the XLA GSPMD partitioner to
          determine the output shardings.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it. This is similar to pjit's in_shardings.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs. This is similar to pjit's
      out_shardings.

      The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
      will use GSPMD's sharding propagation to figure out what the sharding of the
      output(s) should be.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If neither ``static_argnums`` nor ``static_argnames`` is provided, no
      arguments are treated as static. If ``static_argnums`` is not provided but
      ``static_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``static_argnames``
      (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``static_argnums`` or ``static_argnames`` will
      be treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: An optional string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    inline: Specify whether this function should be inlined into enclosing
      jaxprs (rather than being represented as an application of the xla_call
      primitive with its own subjaxpr). Default False.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.
  """

  if fun is Missing:
    return functools.partial(
      jit,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      donate_argnames=donate_argnames,
      keep_unused=keep_unused,
      device=device,
      backend=backend,
      inline=inline,
      abstracted_axes=abstracted_axes,
    )  # type: ignore[return-value]

  return JitWrapped(
    fun,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    static_argnums=static_argnums,
    static_argnames=static_argnames,
    donate_argnums=donate_argnums,
    donate_argnames=donate_argnames,
    keep_unused=keep_unused,
    device=device,
    backend=backend,
    inline=inline,
    abstracted_axes=abstracted_axes,
  )


class JitWrapped:
  """A function ready to be traced, lowered, and compiled.

  This protocol reflects the output of functions such as
  ``jax.jit``. Calling it results in JIT (just-in-time) lowering,
  compilation, and execution. It can also be explicitly lowered prior
  to compilation, and the result compiled prior to execution.
  """

  def __init__(
    self,
    fun: tp.Callable[..., tp.Any],
    in_shardings: tp.Any,
    out_shardings: tp.Any,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
  ):
    functools.update_wrapper(self, fun)
    kwarg_shardings = None
    self.jax_in_shardings = jax.tree.map(
      lambda x: extract.NodeStates.from_prefixes(x.shardings, metadata=x)
      if isinstance(x, StateSharding)
      else x,
      in_shardings,
    )
    self.jax_out_shardings = jax.tree.map(
      lambda x: extract.NodeStates.from_prefixes(x.shardings, metadata=x)
      if isinstance(x, StateSharding)
      else x,
      out_shardings,
    )

    self.jitted_fn = jax.jit(
      JitFn(fun, in_shardings, out_shardings, kwarg_shardings, self),
      in_shardings=self.jax_in_shardings,
      out_shardings=(
        self.jax_in_shardings,
        kwarg_shardings,
        self.jax_out_shardings,
      ),
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      donate_argnames=donate_argnames,
      keep_unused=keep_unused,
      device=device,
      backend=backend,
      inline=inline,
      abstracted_axes=abstracted_axes,
    )
    self.in_shardings = in_shardings
    self.out_shardings = out_shardings
    self.kwarg_shardings = kwarg_shardings
    self.static_argnums = static_argnums

  # implement descriptor protocol so that we can use this as a method
  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    return functools.partial(self, obj)

  def _get_pure_args_kwargs(self, args, kwargs):
    pure_args, pure_kwargs = extract.to_tree(
      (args, kwargs),
      prefix=(self.in_shardings, self.kwarg_shardings)
      if self.in_shardings is not None or self.kwarg_shardings is not None
      else None,
      split_fn=_jit_split_fn,
      check_aliasing=self.in_shardings is not None
      or self.kwarg_shardings is not None,
      ctxtag=self,
    )
    return pure_args, pure_kwargs

  def _get_non_pure_out(self, pure_args_out, pure_kwargs_out, pure_out, /):
    _args_out, _kwargs_out, out = extract.from_tree(
      (pure_args_out, pure_kwargs_out, pure_out),
      merge_fn=_jit_merge_fn,
      is_inner=False,
      ctxtag=self,
    )
    return out

  def __call__(self, *args, **kwargs):
    # run dynamic_cache_context before update_context
    with graph.update_context(self):
      pure_args, pure_kwargs = self._get_pure_args_kwargs(args, kwargs)
      pure_args_out, pure_kwargs_out, pure_out = self.jitted_fn(
        *pure_args, **pure_kwargs
      )
      out = self._get_non_pure_out(pure_args_out, pure_kwargs_out, pure_out)
    return out

  def eval_shape(self, *args, **kwargs):
    """See ``jax.eval_shape``."""
    args, kwargs = graph.clone((args, kwargs))
    with graph.update_context(self):
      pure_args, pure_kwargs = self._get_pure_args_kwargs(args, kwargs)
      pure_args_out, pure_kwargs_out, pure_out = self.jitted_fn.eval_shape(
        *pure_args, **pure_kwargs
      )
      out = self._get_non_pure_out(pure_args_out, pure_kwargs_out, pure_out)
    return out

  def trace(self, *args, **kwargs) -> Traced:
    """Trace this function explicitly for the given arguments.

    A traced function is staged out of Python and translated to a jaxpr. It is
    ready for lowering but not yet lowered.

    Returns:
      A ``Traced`` instance representing the tracing.
    """
    with graph.update_context(self):
      pure_args, pure_kwargs = self._get_pure_args_kwargs(args, kwargs)
      traced = self.jitted_fn.trace(*pure_args, **pure_kwargs)
    return Traced(traced, self)

  def lower(self, *args, **kwargs) -> Lowered:
    """Lower this function explicitly for the given arguments.

    This is a shortcut for ``self.trace(*args, **kwargs).lower()``.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    with graph.update_context(self):
      pure_args, pure_kwargs = self._get_pure_args_kwargs(args, kwargs)
      lowered = self.jitted_fn.lower(*pure_args, **pure_kwargs)
    return Lowered(lowered, self)


class Stage:
  args_info: tp.Any  # PyTree of ArgInfo

  @property
  def _inner_obj(self) -> tp.Any:
    raise NotImplementedError

  @property
  def in_tree(self) -> jax.tree_util.PyTreeDef:
    return self._inner_obj.in_tree

  @property
  def in_avals(self):
    return self._inner_obj.in_avals

  @property
  def donate_argnums(self):
    return self._inner_obj.donate_argnums

@dataclasses.dataclass(frozen=True, slots=True)
class Compiled(Stage):
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """

  compiled: jax.stages.Compiled
  jit_wrapped: JitWrapped

  @property
  def _inner_obj(self):
    return self.compiled

  @property
  def args_info(self) -> tp.Any:  # PyTree of ArgInfo
    raise self.compiled.args_info

  @staticmethod
  def call(*args, **kwargs):
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    with graph.update_context(self.jit_wrapped):
      pure_args, pure_kwargs = self.jit_wrapped._get_pure_args_kwargs(
        args, kwargs
      )
      pure_args_out, pure_kwargs_out, pure_out = self.compiled(
        *pure_args, **pure_kwargs
      )
      out = self.jit_wrapped._get_non_pure_out(
        pure_args_out, pure_kwargs_out, pure_out
      )
    return out

  @property
  def out_tree(self) -> jax.tree_util.PyTreeDef:
    return self.compiled.out_tree

  def as_text(self) -> str | None:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This is not a valid nor
    reliable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self.compiled.as_text()

  def cost_analysis(self) -> tp.Any | None:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self.compiled.cost_analysis()

  def memory_analysis(self) -> tp.Any | None:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self.compiled.memory_analysis()

  def runtime_executable(self) -> tp.Any | None:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This is not valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self.compiled.runtime_executable()

  @property
  def input_shardings(self):  # PyTree[sharding.Sharding]
    return self.compiled.input_shardings

  @property
  def output_shardings(self):  # PyTree[sharding.Sharding]
    return self.compiled.output_shardings

  @property
  def input_layouts(self):
    return self.compiled.input_formats


@dataclasses.dataclass(frozen=True, slots=True)
class Lowered(Stage):
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (:func:`~jax.jit`, :func:`~jax.pmap`, etc.).
  """

  lowered: jax.stages.Lowered
  jit_wrapped: JitWrapped

  @property
  def _inner_obj(self):
    return self.lowered

  @property
  def args_info(self) -> tp.Any:  # PyTree of ArgInfo
    return self.lowered.args_info

  @property
  def out_tree(self):
    return self.lowered.out_tree

  @classmethod
  def from_flat_info(
    cls,
    lowering: tp.Any,  # type: ignore[name-defined]
    in_tree: jax.tree_util.PyTreeDef,
    in_avals,
    donate_argnums: tuple[int, ...],
    out_tree: jax.tree_util.PyTreeDef,
    no_kwargs: bool = False,
  ):
    raise NotImplementedError

  def compile(
    self, compiler_options: jax.stages.CompilerOptions | None = None
  ) -> Compiled:
    """Compile, returning a corresponding ``Compiled`` instance."""
    compiled = self.lowered.compile(compiler_options)
    return Compiled(compiled, self.jit_wrapped)

  def as_text(
    self, dialect: str | None = None, *, debug_info: bool = False
  ) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization.
    Use `jax.export` if you want reliable and portable serialization.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo",
        or "hlo").
      debug_info: Whether to include debugging information,
        e.g., source location.
    """
    return self.lowered.as_text(dialect=dialect, debug_info=debug_info)

  def compiler_ir(self, dialect: str | None = None) -> tp.Any | None:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This is not a valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.
    Use `jax.export` if you want reliable and portable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo",
        or "hlo").
    """
    return self.lowered.compiler_ir(dialect=dialect)

  def cost_analysis(self) -> tp.Any | None:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self.lowered.cost_analysis()

@dataclasses.dataclass(frozen=True, slots=True)
class Traced(Stage):
  """Traced form of a function specialized to argument types and values.

  A traced computation is ready for lowering. This class carries the
  traced representation with the remaining information needed to later
  lower, compile, and execute it.
  """

  traced: jax.stages.Traced
  jit_wrapped: JitWrapped

  @property
  def _inner_obj(self):
    return self.traced

  @property
  def out_info(self):
    return self.traced.out_info

  def lower(
    self, *, lowering_platforms: tuple[str, ...] | None = None
  ) -> Lowered:
    """Lower to compiler input, returning a ``Lowered`` instance."""
    lowered = self.traced.lower(lowering_platforms=lowering_platforms)
    return Lowered(lowered, self.jit_wrapped)


# -------------------------------
# shard_map
# -------------------------------

# TODO: create StateSpec and consider enabling a mode that does
# not use filters during split for performance. Overall there might
# be performance limitations for using shard_map at a top-level


@dataclasses.dataclass(eq=False)
class ShardMapFn:
  f: tp.Callable[..., tp.Any]
  in_specs: tp.Any
  out_specs: tp.Any
  kwarg_specs: tp.Any
  ctxtag: tp.Hashable

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args, **pure_kwargs):
    args, kwargs = extract.from_tree(
      (pure_args, pure_kwargs),
      merge_fn=_jit_merge_fn,
      ctxtag=self.ctxtag,
      is_inner=True,
    )

    out = self.f(*args, **kwargs)

    args_out, kwargs_out = extract.clear_non_graph_nodes((args, kwargs))
    pure_args_out, pure_kwargs_out, pure_out = extract.to_tree(
      (args_out, kwargs_out, out),
      prefix=(self.in_specs, self.kwarg_specs, self.out_specs),
      ctxtag=self.ctxtag,
      split_fn=_jit_split_fn,
    )

    return pure_args_out, pure_kwargs_out, pure_out


@tp.overload
def shard_map(
  f: F,
  *,
  mesh: Mesh | AbstractMesh,
  in_specs: Specs,
  out_specs: Specs,
  check_rep: bool = True,
  auto: frozenset[AxisName] = frozenset(),
) -> F: ...
@tp.overload
def shard_map(
  *,
  mesh: Mesh | AbstractMesh,
  in_specs: Specs,
  out_specs: Specs,
  check_rep: bool = True,
  auto: frozenset[AxisName] = frozenset(),
) -> tp.Callable[[F], F]: ...
def shard_map(
  f: F | type[Missing] = Missing,
  *,
  mesh: Mesh | AbstractMesh,
  in_specs: Specs,
  out_specs: Specs,
  check_rep: bool = True,
  auto: frozenset[AxisName] = frozenset(),
) -> F | tp.Callable[[F], F]:
  """
  Lifted version of
  `jax.experimental.shard_map.shard_map <https://docs.jax.dev/en/latest/_autosummary/jax.experimental.shard_map.shard_map.html>`_
  that can handle Modules / graph nodes as arguments.

  Simple data parallel example::

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import PartitionSpec as P

    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    x = jnp.ones((32, 2))

    @nnx.shard_map(
      mesh=mesh, in_specs=(P(None), P('data')), out_specs=P('data')
    )
    def f(m, x):
      return m(x)

    y = f(m, x)

    jax.debug.visualize_array_sharding(y)

  Notice that here we simply used some ``PartitionSpec`` to define the spec
  the the whole model and data. This works for simple cases but if we need
  to assign different ``PartitionSpec`` to different parts of the model we
  need to use ``StateSharding`` and create some filters that allow us to target
  specific parts of the model. Here's an example of how to do tensor parallelism
  for a simple MLP block using ``StateSharding`` and filters::

    mesh = jax.sharding.Mesh(jax.local_devices(), ('model',))

    class MLP(nnx.Module):
      def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

      def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))

    m = MLP(2, 64, 3, rngs=nnx.Rngs(0))
    x = jnp.ones((32, 2))

    def path_ends_with(*path_suffix): # custom filter
      return lambda path, value: path[-len(path_suffix):] == path_suffix

    model_spec = nnx.StateSharding({
      path_ends_with('linear1', 'kernel'): P(None, 'model'),
      path_ends_with('linear2', 'kernel'): P('model', None),
    })

    @nnx.shard_map(mesh=mesh, in_specs=(model_spec, P(None)), out_specs=P(None))
    def f(m, x):
      y = m(x)
      return jax.lax.psum(y, 'model')

    y = f(m, x)

    jax.debug.visualize_array_sharding(m.linear1.kernel.value)
    jax.debug.visualize_array_sharding(m.linear2.kernel.value)


  Alternatively, a ``State`` object with the exact PartitionSpec for each
  state then you can be passed to ``StateSharding``::

    mesh = jax.sharding.Mesh(jax.local_devices(), ('model',))

    class MLP(nnx.Module):
      def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

      def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))

    m = MLP(2, 64, 3, rngs=nnx.Rngs(0))
    x = jnp.ones((32, 2))

    model_spec = nnx.State(
      {
        'linear1': {'kernel': P(None, 'model')},
        'linear2': {'kernel': P('model', None)},
      }
    )

    @nnx.shard_map(
      mesh=mesh,
      in_specs=(nnx.StateSharding(model_spec), P(None)),
      out_specs=P(None),
    )
    def f(m, x):
      y = m(x)
      return jax.lax.psum(y, 'model')

    y = f(m, x)

    jax.debug.visualize_array_sharding(m.linear1.kernel.value)
    jax.debug.visualize_array_sharding(m.linear2.kernel.value)

  Here ``model_spec`` was created manually but you can also automate
  this process by using ``nnx.get_partition_spec`` to automatically
  create it for you (see
  `Scale up on multiple devices <https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html>`_
  ).

  Args:
    f: callable to be mapped. Each application of ``f``, or "instance" of ``f``,
      takes as input a shard of the mapped-over arguments and produces a shard
      of the output.
    mesh: a ``jax.sharding.Mesh`` representing the array of devices over which
      to shard the data and on which to execute instances of ``f``. The names of
      the ``Mesh`` can be used in collective communication operations in ``f``.
      This is typically created by a utility function like
      :func:`jax.experimental.mesh_utils.create_device_mesh`.
    in_specs: a pytree with ``jax.sharding.PartitionSpec``or ``nnx.StateSharding``
      (mapping substates to ``PartitionSpec``s) instances as leaves,
      with a tree structure that is a tree prefix of the
      args tuple to be mapped over. Similar to ``jax.sharding.NamedSharding``,
      each ``PartitionSpec`` represents how the corresponding argument (or subtree
      of arguments) should be sharded along the named axes of ``mesh``. In each
      ``PartitionSpec``, mentioning a ``mesh`` axis name at a position expresses sharding
      the corresponding argument array axis along that positional axis; not
      mentioning an axis name expresses replication. If an argument, or argument
      subtree, has a corresponding spec of None, that argument is not sharded.
    out_specs: a pytree with ``jax.sharding.PartitionSpec`` or ``nnx.StateSharding``
      (mapping substates to ``PartitionSpec``s) instances as leaves, with a tree structure
      that is a tree prefix of the output of ``f``.
      Each ``PartitionSpec`` represents how the corresponding output shards should be
      concatenated. In each ``PartitionSpec``, metioning a ``mesh`` axis name at
      a position expresses concatenation of that mesh axis's shards along the
      corresponding positional axis. Not mentioning a ``mesh`` axis name
      expresses a promise that the output values are equal along that mesh axis,
      and that rather than concatenating only a single value should be produced.
    check_rep: If True (default) enable additional validity checks and automatic
      differentiation optimizations. The validity checks concern whether any mesh
      axis names not mentioned in ``out_specs`` are consistent with how the outputs
      of ``f`` are replicated. Must be set False if using a Pallas kernel in ``f``.
    auto: (experimental) an optional set of axis names from ``mesh`` over which we
      do not shard the data or map the function, but rather we allow the
      compiler to control sharding. These names cannot be used in ``in_specs``,
      ``out_specs``, or in communication collectives in ``f``.

  Returns:
    A callable that applies the input function ``f`` across data sharded according to
    the ``mesh`` and ``in_specs``.
  """
  if f is Missing:
    return functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      check_rep=check_rep,
      auto=auto,
    )  # type: ignore[return-value]
  assert not isinstance(f, type)

  kwarg_specs = PartitionSpec()
  jax_in_specs = jax.tree.map(
    lambda x: extract.NodeStates(
      _graphdef=PartitionSpec(),  # type: ignore[arg-type]
      states=x.shardings,
      metadata=x,
    )
    if isinstance(x, StateSharding)
    else x,
    in_specs,
  )
  jax_out_specs = jax.tree.map(
    lambda x: extract.NodeStates(
      _graphdef=PartitionSpec(),  # type: ignore[arg-type]
      states=x.shardings,
      metadata=x,
    )
    if isinstance(x, StateSharding)
    else x,
    out_specs,
  )

  @functools.wraps(f)
  def shard_map_wrapper(*args, **kwargs):
    # run dynamic_cache_context before update_context
    with graph.update_context(shard_map_wrapper):
      pure_args, pure_kwargs = extract.to_tree(
        (args, kwargs),
        prefix=(in_specs, kwarg_specs)
        if in_specs is not None or kwarg_specs is not None
        else None,
        split_fn=_jit_split_fn,
        check_aliasing=in_specs is not None or kwarg_specs is not None,
        ctxtag=shard_map_wrapper,
      )
      pure_args_out, pure_kwargs_out, pure_out = shard_map_fn(
        *pure_args, **pure_kwargs
      )
      _args_out, _kwargs_out, out = extract.from_tree(
        (pure_args_out, pure_kwargs_out, pure_out),
        merge_fn=_jit_merge_fn,
        is_inner=False,
        ctxtag=shard_map_wrapper,
      )
    return out

  shard_map_fn = jax.experimental.shard_map.shard_map(
    ShardMapFn(f, in_specs, out_specs, kwarg_specs, shard_map_wrapper),
    mesh=mesh,
    in_specs=jax_in_specs,
    out_specs=(jax_in_specs, kwarg_specs, jax_out_specs),  # type: ignore
    check_rep=check_rep,
    auto=auto,
  )

  shard_map_wrapper.inner = shard_map_fn  # type: ignore

  return shard_map_wrapper  # type: ignore
