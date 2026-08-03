"""Exceptions raised by the AI framework."""


class OutputValidationError(Exception):
    """Raised when a response fails output validation."""


class ScriptReviewNotApprovedError(Exception):
    """Raised when storyboard generation is requested for a rejected script review."""
