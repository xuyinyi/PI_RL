"""Resolve adapter classes without a central algorithm switch statement."""

from __future__ import annotations

import importlib
from typing import Type

from reproduction.framework.contracts import AlgorithmAdapter, ContractError


def load_adapter_class(locator: str) -> Type[AlgorithmAdapter]:
    if ":" not in locator:
        raise ContractError("algorithm.adapter must use module:Class syntax")
    module_name, class_name = locator.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise ContractError("cannot load adapter {}: {}".format(locator, error))
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, AlgorithmAdapter):
        raise ContractError("{} is not an AlgorithmAdapter".format(locator))
    return adapter_class
