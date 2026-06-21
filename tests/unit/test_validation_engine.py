from __future__ import annotations

import inspect

from pydantic import BaseModel

from redstack.engines.validation import ValidationEngine


def test_validation_engine_does_not_shadow_pydantic_validate() -> None:
    """Regression: the engine's ranking-check method must not be named
    ``validate`` -- that name is pydantic.BaseModel's own deprecated
    classmethod (``validate(value) -> Self``), and overriding it with an
    unrelated instance method is a Liskov violation (caught by mypy --strict
    as an incompatible override) that also makes ``ValidationEngine.validate``
    silently stop behaving like the inherited pydantic API.
    """
    assert hasattr(ValidationEngine, "validate_ranking")
    assert ValidationEngine.validate.__func__ is BaseModel.validate.__func__
    sig = inspect.signature(ValidationEngine.validate_ranking)
    assert list(sig.parameters) == ["self", "ranking"]
