from abc import abstractmethod
import copy
import logging

import pandas as pd
from consecution import Node


class IterableModifierNode(Node):
    """
    Узел, предназначенный для обработки массива данных.
    Итерируется по данным и применяет на них _value_changer.
    """
    def __init__(
        self,
        name: str,
        target_key: str,
        source_key: str,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.target_key = target_key
        self.source_key = source_key

    @abstractmethod
    def _value_changer(self, value):
        pass

    def process(self, item: pd.DataFrame):
        iterable = copy.deepcopy(self.global_state.processing_data[self.source_key])

        for idx, val in enumerate(iterable):
            iterable[idx] = self._value_changer(val)

        self.global_state.processing_data[self.target_key] = iterable

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)


class OnetoOneModifierNode(Node):
    """
    Всё тоже самое, что в IterableModifierNode,
    только для конкретного экземпляра данных.
    """
    def __init__(
        self,
        name: str,
        target_key: str,
        source_key: str,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.target_key = target_key
        self.source_key = source_key

    @abstractmethod
    def _value_changer(self, value):
        pass

    def process(self, item: pd.DataFrame):
        data = copy.deepcopy(self.global_state.processing_data[self.source_key])

        self.global_state.processing_data[self.target_key] = self._value_changer(data)

        logging.info(f'Узел {self.name} вида {self.__class__.__name__} закончил работу')

        self._push(item)
