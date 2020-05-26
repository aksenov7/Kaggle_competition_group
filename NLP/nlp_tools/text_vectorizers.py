from abc import abstractmethod
import logging
import typing as t

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .core import OnetoOneModifierNode


class VectorizingNode(OnetoOneModifierNode):
    """
    Базовый узел, который осуществуляет векторизацию входных данных.
    """
    def __init__(
        self,
        name: str,
        target_key: str,
        source_key: str,
        vectorizer_settings: t.Dict[str, t.Any] = None,
        **kwargs
    ):
        super().__init__(name, target_key, source_key, **kwargs)

        if vectorizer_settings is None:
            vectorizer_settings = {}

        self._vectorizer = self.vectorizer(**vectorizer_settings)

    @property
    @abstractmethod
    def vectorizer(self):
        pass

    def _value_changer(self, value):
        processed = self._vectorizer.fit_transform(value)

        logging.info(self._vectorizer.get_feature_names())

        return processed


class SKLearnCountVectorizerNode(VectorizingNode):
    """
    Узел осуществляет векторизацию методом CountVectorizer.
    """
    @property
    def vectorizer(self):
        return CountVectorizer


class SKLearnTfidVectorizerNode(VectorizingNode):
    """
    Узел осуществляет векторизацию методом TfidfVectorizer.
    """
    @property
    def vectorizer(self):
        return TfidfVectorizer
