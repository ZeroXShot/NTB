"""Semantic validation: located diagnostics over a document."""

from ntb.validate.diagnostics import Code, Diagnostic, Location, Report, Severity
from ntb.validate.validator import validate

__all__ = ["Code", "Diagnostic", "Location", "Report", "Severity", "validate"]
