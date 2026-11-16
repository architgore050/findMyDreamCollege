"""Forced-verdict schema for the second pass.

There is no ``value`` field anywhere in this schema, by construction: the
verifier may return a verdict, a short reason and an optional verbatim
``corrected_quote``. ``extra="forbid"`` makes a model that tries to emit a value
fail schema validation rather than leak one into the pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    verdict: Literal["supported", "contradicted", "absent", "wrong_field"]
    corrected_quote: str | None = None
    reason: str | None = None


class VerdictBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[FieldJudgement] = Field(default_factory=list)
