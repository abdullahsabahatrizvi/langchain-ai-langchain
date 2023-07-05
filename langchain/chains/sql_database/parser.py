from typing import Any
from langchain.schema import BaseModel
from langchain.schema import BaseOutputParser

class VectorSQLOutputParser(BaseOutputParser):
    model: BaseModel
    distance_func_name: str = 'distance'
    def from_embeddings(cls, model: BaseModel, distance_func_name: str ='distane', **kwargs: Any):
        return cls(model=model, distance_func_name=distance_func_name, **kwargs)
    
    def parse(self, text: str):
        start = text.find('NeuralArray(')
        if start > 0:
            _matched = text[text.find('NeuralArray(')+len('NeuralArray('):]
            end = _matched.find(')') + start + len('NeuralArray(') + 1
            entity = _matched[:_matched.find(')')]
            vecs = self.model.embed_query(entity)
            vecs_str = '[' + ','.join(map(str, vecs)) + ']'
            _sql_str_compl = text.replace('DISTANCE', self.distance_func_name).replace(text[start:end], vecs_str)
            if _sql_str_compl[-1] == ';':
                _sql_str_compl = _sql_str_compl[:-1]
            text = _sql_str_compl
        return text     