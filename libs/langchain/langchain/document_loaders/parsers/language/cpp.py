from langchain.document_loaders.parsers.language.tree_sitter_segmenter import (
    TreeSitterSegmenter,
)

CHUNK_QUERY = """
    [
        (class_specifier
            body: (field_declaration_list)) @class
        (struct_specifier
            body: (field_declaration_list)) @struct
        (function_definition) @function
    ]
""".strip()


class CPPSegmenter(TreeSitterSegmenter):
    """Code segmenter for C++."""

    def get_language(self):
        from tree_sitter_languages import get_language

        return get_language("cpp")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"// {text}"
