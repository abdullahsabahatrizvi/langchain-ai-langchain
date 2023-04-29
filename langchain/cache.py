"""Beta Feature: base interface for cache."""
import json
import hashlib

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.embeddings.base import Embeddings
from langchain.schema import Generation


RETURN_VAL_TYPE = List[Generation]

def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments."""


class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: Dict[Tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[(prompt, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}


Base = declarative_base()


class FullLLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "full_llm_cache"
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    idx = Column(Integer, primary_key=True)
    response = Column(String)


class SQLAlchemyCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Type[FullLLMCache] = FullLLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if rows:
                return [Generation(text=row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        items = [
            self.cache_schema(prompt=prompt, llm=llm_string, response=gen.text, idx=i)
            for i, gen in enumerate(return_val)
        ]
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        with Session(self.engine) as session:
            session.execute(self.cache_schema.delete())


class SQLiteCache(SQLAlchemyCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str = ".langchain.db"):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)


class RedisCache(BaseCache):
    """Cache that uses Redis as a backend."""
    # TODO - implement a TTL policy in Redis

    def __init__(self, redis_: Any):
        """Initialize by passing in Redis instance."""
        try:
            from redis import Redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass in Redis object.")
        self.redis = redis_

    def _key(self, prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        generations = []
        # Read from a Redis HASH
        results = self.redis.hgetall(self._key(prompt, llm_string))
        if results:
            for _, text in results.items():
                generations.append(Generation(text=text.decode()))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        # Write to a Redis HASH
        key = self._key(prompt, llm_string)
        self.redis.hset(key, mapping = {
            idx: generation.text
            for idx, generation in enumerate(return_val)
        })

class RedisSemanticCache(BaseCache):
    """Cache that uses Redis as a vector-store backend."""
    # TODO - implement a TTL policy in Redis

    def __init__(
        self,
        redis_url: str,
        embedding: Embeddings,
        score_threshold: float = 0.2
    ):
        """Initialize by passing in the `init` GPTCache func

        Args:
            redis_url (str): URL to connect to Redis.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):

        Example:
        .. code-block:: python
            import langchain

            from langchain.cache import RedisSemanticCache
            from langchain.embeddings import OpenAIEmbeddings

            langchain.llm_cache = RedisSemanticCache(
                redis_url="redis://localhost:6379",
                embedding=OpenAIEmbeddings()
            )

        """
        self._cache_dict = {}
        self.redis_url = redis_url
        self.embedding = embedding
        self.score_threshold = score_threshold

    def _index_name(self, llm_string: str):
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str):
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        from langchain.vectorstores.redis import Redis

        # create new vectorstore client for the specific llm string
        try:
            self._cache_dict[index_name] = Redis.from_existing_index(
                embedding=self.embedding,
                index_name=index_name,
                redis_url=self.redis_url
            )
        except ValueError:
            redis = Redis(
                embedding_function=self.embedding.embed_query,
                index_name=index_name,
                redis_url=self.redis_url,
            )
            _embedding = self.embedding.embed_query(text="test")
            redis._create_index(dim=len(_embedding))
            self._cache_dict[index_name] = redis

        return self._cache_dict[index_name]

    def clear(self, llm_string: str):
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(llm_string)
        if index_name in self._cache_dict:
            self._cache_dict[index_name].drop_index(
                index_name=index_name,
                delete_documents=True,
                redis_url=self.redis_url
            )
            del self._cache_dict[index_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations = []
        # Read from a Hash
        results = llm_cache.similarity_search_limit_score(
            query=prompt,
            k=1,
            score_threshold=self.score_threshold,
        )
        if results:
            for document in results:
                for text in document.metadata["return_val"]:
                    generations.append(Generation(text=text))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        # Write to vectorstore
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": [generation.text for generation in return_val]
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        asynchronous = kwargs.get("asynchronous", False)
        self.redis.flushdb(asynchronous=asynchronous, **kwargs)


class GPTCache(BaseCache):
    """Cache that uses GPTCache as a backend."""

    def __init__(self, init_func: Optional[Callable[[Any], None]] = None):
        """Initialize by passing in init function (default: `None`).

        Args:
            init_func (Optional[Callable[[Any], None]]): init `GPTCache` function
            (default: `None`)

        Example:
        .. code-block:: python

            # Initialize GPTCache with a custom init function
            import gptcache
            from gptcache.processor.pre import get_prompt
            from gptcache.manager.factory import get_data_manager

            # Avoid multiple caches using the same file,
            causing different llm model caches to affect each other
            i = 0
            file_prefix = "data_map"

            def init_gptcache_map(cache_obj: gptcache.Cache):
                nonlocal i
                cache_path = f'{file_prefix}_{i}.txt'
                cache_obj.init(
                    pre_embedding_func=get_prompt,
                    data_manager=get_data_manager(data_path=cache_path),
                )
                i += 1

            langchain.llm_cache = GPTCache(init_gptcache_map)

        """
        try:
            import gptcache  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import gptcache python package. "
                "Please install it with `pip install gptcache`."
            )

        self.init_gptcache_func: Optional[Callable[[Any], None]] = init_func
        self.gptcache_dict: Dict[str, Any] = {}

    @staticmethod
    def _update_cache_callback_none(*_: Any, **__: Any) -> None:
        """When updating cached data, do nothing.

        Because currently only cached queries are processed."""
        return None

    @staticmethod
    def _llm_handle_none(*_: Any, **__: Any) -> None:
        """Do nothing on a cache miss"""
        return None

    @staticmethod
    def _cache_data_converter(data: str) -> RETURN_VAL_TYPE:
        """Convert the `data` in the cache to the `RETURN_VAL_TYPE` data format."""
        return [Generation(**generation_dict) for generation_dict in json.loads(data)]

    def _get_gptcache(self, llm_string: str) -> Any:
        """Get a cache object.

        When the corresponding llm model cache does not exist, it will be created."""
        from gptcache import Cache
        from gptcache.manager.factory import get_data_manager
        from gptcache.processor.pre import get_prompt

        _gptcache = self.gptcache_dict.get(llm_string, None)
        if _gptcache is None:
            _gptcache = Cache()
            if self.init_gptcache_func is not None:
                self.init_gptcache_func(_gptcache)
            else:
                _gptcache.init(
                    pre_embedding_func=get_prompt,
                    data_manager=get_data_manager(data_path=llm_string),
                )
            self.gptcache_dict[llm_string] = _gptcache
        return _gptcache

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the cache data.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then retrieve the data from the cache based on the `prompt`.
        """
        from gptcache.adapter.adapter import adapt

        _gptcache = self.gptcache_dict.get(llm_string, None)
        if _gptcache is None:
            return None
        res = adapt(
            GPTCache._llm_handle_none,
            GPTCache._cache_data_converter,
            GPTCache._update_cache_callback_none,
            cache_obj=_gptcache,
            prompt=prompt,
        )
        return res

    @staticmethod
    def _update_cache_callback(
        llm_data: RETURN_VAL_TYPE,
        update_cache_func: Callable[[Any], None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Save the `llm_data` to cache storage"""
        handled_data = json.dumps([generation.dict() for generation in llm_data])
        update_cache_func(handled_data)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then store the `prompt` and `return_val` in the cache object.
        """
        from gptcache.adapter.adapter import adapt

        _gptcache = self._get_gptcache(llm_string)

        def llm_handle(*_: Any, **__: Any) -> RETURN_VAL_TYPE:
            return return_val

        return adapt(
            llm_handle,
            GPTCache._cache_data_converter,
            GPTCache._update_cache_callback,
            cache_obj=_gptcache,
            cache_skip=True,
            prompt=prompt,
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        from gptcache import Cache

        for gptcache_instance in self.gptcache_dict.values():
            gptcache_instance = cast(Cache, gptcache_instance)
            gptcache_instance.flush()

        self.gptcache_dict.clear()
