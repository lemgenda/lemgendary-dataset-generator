"""Single source of truth for the exceptions the Kaggle client stack raises.

The ``kaggle``, ``kagglehub`` and ``kagglesdk`` packages report expected remote
failures (network errors, HTTP errors, missing credentials, malformed JSON
responses, unknown datasets) through these types. Callers that talk to Kaggle
catch ``KAGGLE_REMOTE_ERRORS`` instead of ``Exception`` so programming errors
(``TypeError``, ``AttributeError``, ...) still surface.
"""

from kagglehub.exceptions import (
    BackendError,
    CredentialError,
    NotFoundError,
    UnauthenticatedError,
)

KAGGLE_REMOTE_ERRORS: tuple[type[BaseException], ...] = (
    # requests.RequestException (incl. HTTPError, KaggleApiHTTPError) and
    # socket timeouts derive from OSError; missing kaggle.json raises OSError.
    OSError,
    # json.JSONDecodeError and invalid-argument rejections derive from ValueError.
    ValueError,
    BackendError,
    CredentialError,
    NotFoundError,
    UnauthenticatedError,
)

# Lazy ``kaggle`` / ``kagglesdk`` imports can also fail; ``import kaggle``
# authenticates on import and raises OSError when kaggle.json is missing.
KAGGLE_CLIENT_ERRORS: tuple[type[BaseException], ...] = (ImportError, *KAGGLE_REMOTE_ERRORS)
