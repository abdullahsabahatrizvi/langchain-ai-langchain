from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain.chains.query_constructor.schema import VirtualColumnName


class PineconeTranslator(Visitor):
    """Translate `Pinecone` internal query language elements to valid filters."""

    allowed_comparators = (
        Comparator.EQ,
        Comparator.NE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.IN,
        Comparator.NIN,
    )
    """Subset of allowed logical comparators."""
    allowed_operators = (Operator.AND, Operator.OR)
    """Subset of allowed logical operators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return f"${func.value}"

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        if comparison.comparator in (Comparator.IN, Comparator.NIN) and not isinstance(
            comparison.value, list
        ):
            comparison.value = [comparison.value]
        if type(comparison.attribute) is VirtualColumnName:
            attribute = comparison.attribute()
        elif type(comparison.attribute) is str:
            attribute = comparison.attribute
        else:
            raise TypeError(
                f"Unknown type {type(comparison.attribute)} for `comparison.attribute`!"
            )

        return {attribute: {self._format_func(comparison.comparator): comparison.value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
