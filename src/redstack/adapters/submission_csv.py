"""CsvSubmissionSinkAdapter implements SubmissionSinkPort. Fixed header/order/precision, RFC-4180, atomic temp+rename, output_sha256.

Owner layer: adapters.
Allowed imports: ports, domain, csv, io, hashlib.

Sprint 0 placeholder: declarations only, no implementation.
"""
from __future__ import annotations

__all__: tuple[str, ...] = (
    "CsvSubmissionSinkAdapter",
)
