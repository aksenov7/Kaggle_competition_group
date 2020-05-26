import typing as t

from nltk import sent_tokenize, word_tokenize

from .core import IterableModifierNode


class SpaceTokenizerNode(IterableModifierNode):
    """
    Узел для разбивки текста по пробелам.
    """
    def _value_changer(self, value: str) -> t.List[str]:
        return value.split(' ')


class NLTKTokenizerNode(IterableModifierNode):
    """
    Базовый узел для токенайзеров, который устанавливает анализируемый язык.
    """
    def __init__(
        self,
        name: str,
        target_key: str,
        source_key: str,
        lang: str = 'english',
        **kwargs
    ):
        super().__init__(name, target_key, source_key, **kwargs)

        self._lang = lang


class WordNLTKTokenizerNode(NLTKTokenizerNode):
    """
    Узел разбивающий текст на токены встроенным word_tokenize из nltk.
    """
    def _value_changer(self, value: str) -> t.List[str]:
        return word_tokenize(value, language=self._lang)


class SentenceNLTKTokenizerNode(NLTKTokenizerNode):
    """
    Узел разбивающий текст на фразовые токены встроенным sent_tokenize из nltk.
    """
    def _value_changer(self, value: str) -> t.List[str]:
        return sent_tokenize(value, language=self._lang)
