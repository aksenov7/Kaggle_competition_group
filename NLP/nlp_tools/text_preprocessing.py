import re
import typing as t

from .core import IterableModifierNode


class RegexpNode(IterableModifierNode):
    """
    Узел последовательно применяет набор регулярных выражений на входные данные.
    """
    def __init__(
        self,
        name: str,
        target_key: str,
        source_key: str,
        pattern_to_sub_dict: t.Dict[str, str],
        **kwargs
    ):
        super().__init__(name, target_key, source_key, **kwargs)

        self._regex_mapping = {
            re.compile(k, flags=re.IGNORECASE | re.MULTILINE): v
            for k, v
            in pattern_to_sub_dict.items()
        }

    def _value_changer(self, value) -> str:
        for pattern, new_value in self._regex_mapping.items():
            value = pattern.sub(new_value, value)

        return value


class RegexpTrashReplaceNode(RegexpNode):
    """
    Узел с заготовленным набором регулярок, которые вычищают из текста мусор.
    """
    DEFAULT_RE_LIST = [
        r'(http|https)?:\/\/.*[\r\n]*',
        r' via .\w+',
        r'@\w+',
        r'\s+[a-zA-Z]\s+',
        r'\^[a-zA-Z]\s+',
        r'\W+',
        r'\s+',
        r'^\s+',
    ]
    _DEFAULT_MAPPING = {x: ' ' for x in DEFAULT_RE_LIST}

    def __init__(
        self,
        name: str,
        target_key: str,
        source_key: str,
        pattern_to_sub_dict: t.Optional[t.Dict[str, str]] = None,
        **kwargs
    ):
        if pattern_to_sub_dict is None:
            pattern_to_sub_dict = {}

        pattern_to_sub_dict.update(self._DEFAULT_MAPPING)

        super().__init__(name, target_key, source_key, pattern_to_sub_dict, **kwargs)
