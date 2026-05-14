"""Re-export shim — canonical definitions live in ``probeflow.core.file_type``.

``FileType`` and ``sniff_file_type`` moved to ``core`` so that
``core.loaders`` and ``core.indexing`` can use them without depending on
``io``.  This shim keeps the old import path working for any code that still
uses ``probeflow.io.file_type``.
"""

from probeflow.core.file_type import FileType, sniff_file_type  # noqa: F401

__all__ = ["FileType", "sniff_file_type"]
