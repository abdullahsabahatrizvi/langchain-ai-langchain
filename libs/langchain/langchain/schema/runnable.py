import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from pydantic import Field

from langchain.callbacks.base import BaseCallbackManager, Callbacks
from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable


async def _gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    async with semaphore:
        return await coro


async def _gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(_gated_coro(semaphore, c) for c in coros))


class RunnableConfig(TypedDict, total=False):
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


Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")
Other = TypeVar("Other")


class Runnable(Generic[Input, Output], ABC):
    def __or__(
        self, __value: "Runnable[Any, Other]"
    ) -> "RunnableSequence[Input, Other]":
        if isinstance(__value, Runnable):
            return RunnableSequence(first=self, last=__value)
        else:
            raise TypeError(f"unsupported type: {type(__value)}")

    def __ror__(
        self,
        __value: Union[
            "Runnable[Other, Any]",
            Dict[str, Union["Runnable[Other, Any]", Callable[[Other], Any]]],
        ],
    ) -> "RunnableSequence[Other, Output]":
        if isinstance(__value, dict):
            runnables = {
                key: r if isinstance(r, Runnable) else RunnableLambda(r)
                for key, r in __value.items()
            }
            return RunnableSequence(
                first=RunnableCombine(runnables=runnables),
                last=self,
            )
        elif isinstance(__value, Runnable):
            return RunnableSequence(first=__value, last=self)
        else:
            raise TypeError(f"unsupported type: {type(__value)}")

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.invoke, input, config
        )

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        configs = self._get_config_list(config, len(inputs))

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = [
                executor.submit(self.invoke, input, config)
                for input, config in zip(inputs, configs)
            ]
            return [future.result() for future in futures]

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        configs = self._get_config_list(config, len(inputs))
        coros = (self.ainvoke(input, config) for input, config in zip(inputs, configs))

        return await _gather_with_concurrency(max_concurrency, *coros)

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        yield self.invoke(input, config)

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        yield await self.ainvoke(input, config)

    def _get_config_list(
        self, config: Optional[Union[RunnableConfig, List[RunnableConfig]]], length: int
    ) -> List[RunnableConfig]:
        if isinstance(config, list) and len(config) != length:
            raise ValueError(
                f"config must be a list of the same length as inputs, "
                f"but got {len(config)} configs for {length} inputs"
            )

        return config if isinstance(config, list) else [config or {}] * length


class RunnableSequence(Serializable, Runnable[Input, Output]):
    first: Runnable[Input, Any]
    middle: List[Runnable[Any, Any]] = Field(default_factory=list)
    last: Runnable[Any, Output]

    @property
    def lc_serializable(self) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True

    def __or__(self, __value: Runnable[Any, Other]) -> "RunnableSequence[Input, Other]":
        if isinstance(__value, RunnableSequence):
            return RunnableSequence(
                first=self.first,
                middle=self.middle + [self.last] + __value.middle,
                last=__value.last,
            )
        elif isinstance(__value, Runnable):
            return RunnableSequence(
                first=self.first, middle=self.middle + [self.last], last=__value
            )
        else:
            raise TypeError(f"unsupported type: {type(__value)}")

    def __ror__(
        self,
        __value: Union[
            Runnable[Other, Any],
            Dict[str, Union["Runnable[Other, Any]", Callable[[Other], Any]]],
        ],
    ) -> "RunnableSequence[Other, Output]":
        if isinstance(__value, dict):
            runnables = {
                key: r if isinstance(r, Runnable) else RunnableLambda(r)
                for key, r in __value.items()
            }
            return RunnableSequence(
                first=RunnableCombine(runnables=runnables),
                middle=[self.first] + self.middle,
                last=self.last,
            )
        elif isinstance(__value, RunnableSequence):
            return RunnableSequence(
                first=__value.first,
                middle=__value.middle + [__value.last] + self.middle,
                last=self.last,
            )
        elif isinstance(__value, Runnable):
            return RunnableSequence(
                first=__value, middle=[self.first] + self.middle, last=self.last
            )
        else:
            raise TypeError(f"unsupported type: {type(__value)}")

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        cm = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = cm.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke
        try:
            for step in [self.first] + self.middle + [self.last]:
                input = step.invoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise
        else:
            rm.on_chain_end(input if isinstance(input, dict) else {"output": input})
            return cast(Output, input)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        cm = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = await cm.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke
        try:
            for step in [self.first] + self.middle + [self.last]:
                input = await step.ainvoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise
        else:
            await rm.on_chain_end(
                input if isinstance(input, dict) else {"output": input}
            )
            return cast(Output, input)

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        configs = self._get_config_list(config, len(inputs))
        cms = [
            CallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        rms = [
            cm.on_chain_start(
                dumpd(self), input if isinstance(input, dict) else {"input": input}
            )
            for cm, input in zip(cms, inputs)
        ]

        # invoke
        try:
            for step in [self.first] + self.middle + [self.last]:
                inputs = step.batch(
                    inputs,
                    [
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(rms, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
        except (KeyboardInterrupt, Exception) as e:
            for rm in rms:
                rm.on_chain_error(e)
            raise
        else:
            for rm, input in zip(rms, inputs):
                rm.on_chain_end(input if isinstance(input, dict) else {"output": input})
            return cast(List[Output], inputs)

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        from langchain.callbacks.manager import (
            AsyncCallbackManager,
            AsyncCallbackManagerForChainRun,
        )

        # setup callbacks
        configs = self._get_config_list(config, len(inputs))
        cms = [
            AsyncCallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        rms: List[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *[
                cm.on_chain_start(
                    dumpd(self), input if isinstance(input, dict) else {"input": input}
                )
                for cm, input in zip(cms, inputs)
            ]
        )

        # invoke batch on each step
        # this uses batching optimizations in subclasses, like LLM
        try:
            for step in [self.first] + self.middle + [self.last]:
                inputs = await step.abatch(
                    inputs,
                    [
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(rms, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
        except (KeyboardInterrupt, Exception) as e:
            await asyncio.gather(*[rm.on_chain_error(e) for rm in rms])
            raise
        else:
            await asyncio.gather(
                *[
                    rm.on_chain_end(
                        input if isinstance(input, dict) else {"output": input}
                    )
                    for rm, input in zip(rms, inputs)
                ]
            )
            return cast(List[Output], inputs)

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        cm = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = cm.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke the first steps
        try:
            for step in [self.first] + self.middle:
                input = step.invoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise

        # stream the last step
        final: Union[Output, None] = None
        final_supported = True
        try:
            for output in self.last.stream(
                input, _patch_config(config, rm.get_child())
            ):
                yield output
                # Accumulate output if possible, otherwise disable accumulation
                if final_supported:
                    if final is None:
                        final = output
                    else:
                        try:
                            final += output  # type: ignore[operator]
                        except TypeError:
                            final = None
                            final_supported = False
                            pass
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise
        else:
            rm.on_chain_end(final if isinstance(final, dict) else {"output": final})

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        cm = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = await cm.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke the first steps
        try:
            for step in [self.first] + self.middle:
                input = await step.ainvoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise

        # stream the last step
        final: Union[Output, None] = None
        final_supported = True
        try:
            async for output in self.last.astream(
                input, _patch_config(config, rm.get_child())
            ):
                yield output
                # Accumulate output if possible, otherwise disable accumulation
                if final_supported:
                    if final is None:
                        final = output
                    else:
                        try:
                            final += output  # type: ignore[operator]
                        except TypeError:
                            final = None
                            final_supported = False
                            pass
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise
        else:
            await rm.on_chain_end(
                final if isinstance(final, dict) else {"output": final}
            )


class RunnableCombine(Serializable, Runnable[Input, Dict[str, Any]]):
    runnables: Dict[str, Runnable[Input, Any]]

    @property
    def lc_serializable(self) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        cm = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = cm.on_chain_start(dumpd(self), {"input": input})

        # invoke
        try:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        step.invoke, input, _patch_config(config, rm.get_child())
                    )
                    for step in self.runnables.values()
                ]
                output = {
                    key: future.result() for key, future in zip(self.runnables, futures)
                }
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise
        else:
            rm.on_chain_end(output)
            return output

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        cm = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = await cm.on_chain_start(dumpd(self), {"input": input})

        # invoke
        try:
            results = await asyncio.gather(
                *[
                    step.ainvoke(input, _patch_config(config, rm.get_child()))
                    for step in self.runnables.values()
                ]
            )
            output = {key: value for key, value in zip(self.runnables, results)}
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise
        else:
            await rm.on_chain_end(output)
            return output


class RunnableLambda(Runnable[Input, Output]):
    def __init__(self, func: Callable[[Input], Output]) -> None:
        if callable(func):
            self.func = func
        else:
            raise TypeError(f"unsupported type: {type(func)}")

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self.func(input)


def _patch_config(
    config: RunnableConfig, callback_manager: BaseCallbackManager
) -> RunnableConfig:
    config = config.copy()
    config["callbacks"] = callback_manager
    return config
