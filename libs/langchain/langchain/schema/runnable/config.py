from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union
from uuid import UUID

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain.callbacks.base import BaseCallbackManager, Callbacks
    from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager


class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable."""

    tags: List[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    metadata: Dict[str, Any]
    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """

    callbacks: Callbacks
    """
    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).
    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
    """

    run_name: str
    """
    Name for the tracer run for this call. Defaults to the name of the class.
    """

    run_id: UUID
    """
    Unique ID for the tracer run for this call. Defaults to uuid4().
    """

    _locals: Dict[str, Any]
    """
    Local variables
    """

    max_concurrency: Optional[int]
    """
    Maximum number of parallel calls to make. If not provided, defaults to 
    ThreadPoolExecutor's default. This is ignored if an executor is provided.
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to 10.
    """


def ensure_config(config: Optional[RunnableConfig] = None) -> RunnableConfig:
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        _locals={},
        recursion_limit=10,
    )
    if config is not None:
        empty.update(config)
    return empty


def get_config_list(
    config: Optional[Union[RunnableConfig, List[RunnableConfig]]], length: int
) -> List[RunnableConfig]:
    """
    Helper method to get a list of configs from a single config or a list of
    configs, useful for subclasses overriding batch() or abatch().
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, but got {length}")
    if isinstance(config, list) and len(config) != length:
        raise ValueError(
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )

    return (
        list(map(ensure_config, config))
        if isinstance(config, list)
        else [patch_config(config, deep_copy_locals=True) for _ in range(length)]
    )


def patch_config(
    config: Optional[RunnableConfig],
    *,
    deep_copy_locals: bool = False,
    callbacks: Optional[BaseCallbackManager] = None,
    recursion_limit: Optional[int] = None,
) -> RunnableConfig:
    config = ensure_config(config)
    if deep_copy_locals:
        config["_locals"] = deepcopy(config["_locals"])
    if callbacks is not None:
        # If we're replacing callbacks we need to unset run_name and run_id
        # As those should apply only to the same run as the original callbacks
        config["callbacks"] = callbacks
        config["run_name"] = None
        config["run_id"] = None
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    from langchain.callbacks.manager import CallbackManager

    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    from langchain.callbacks.manager import AsyncCallbackManager

    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


@contextmanager
def get_executor_for_config(config: RunnableConfig) -> Generator[Executor, None, None]:
    with ThreadPoolExecutor(max_workers=config.get("max_concurrency")) as executor:
        yield executor
