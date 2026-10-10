"""Logging filter that keeps the strict (``-W``) docs build robust to
TRANSIENT third-party doc-site outages.

sphinx-gallery hyperlinks the API names in gallery code to external docs (the
``reference_url`` sites in conf.py) by fetching each one's ``searchindex.js``.
A transient outage there (e.g. an HTTP 503) makes sphinx-gallery log a warning
and continue -- the affected links simply aren't embedded -- but the docs-clean
CI gate builds with ``-W``, which would turn that third-party network hiccup
into a hard build failure. This filter drops ONLY that specific fetch-failure
warning, on sphinx-gallery's own logger, before ``-W``'s raise handler sees it;
every other warning (broken refs, gallery-execution errors, missing
Chrome/pandoc) is left untouched. Read the Docs does not build with ``-W``, so
its behavior is unchanged.

Kept in its own SIDE-EFFECT-FREE module (no sphinx/plotly imports, no path
munging) so tests/test_docs_gallery_log_filter.py can unit-test it: the guard
there fails if sphinx-gallery ever changes the message wording this filter
matches on, instead of the filter silently ceasing to work.
"""
import logging

# The exact phrase sphinx-gallery's external documentation-link resolver puts
# in its HTTPError/URLError fetch-failure warning. See
# sphinx_gallery/docs_resolv.py::_handle_http_url_error, which builds
#   "The following {ExcName} has occurred {msg} {url}: ..."
# with msg defaulting to "fetching", so the emitted message always contains
# this substring. tests/test_docs_gallery_log_filter.py asserts that against
# sphinx-gallery's actual source so a wording change is caught, not missed.
GALLERY_LOGGER = 'sphinx.sphinx-gallery'
_TRANSIENT_FETCH_MARKER = 'has occurred fetching'


def is_transient_external_doc_link_fetch_warning(message):
    """True for sphinx-gallery's transient external ``searchindex.js`` fetch
    failure (a third-party doc site returning e.g. HTTP 503)."""
    return _TRANSIENT_FETCH_MARKER in message


class TransientDocLinkFetchFilter(logging.Filter):
    def filter(self, record):
        # returning False drops the record before -W's raise handler sees it
        return not is_transient_external_doc_link_fetch_warning(
            record.getMessage())


def install(logger_name=GALLERY_LOGGER):
    """Attach the filter to sphinx-gallery's logger (idempotent enough for a
    single docs build). Called from conf.py's ``setup(app)``."""
    logging.getLogger(logger_name).addFilter(TransientDocLinkFetchFilter())


def relative_warning_paths(root):
    """Make Python warnings name files under ``root`` by their path relative
    to it, e.g. ``examples/plot_impute.py:43: UserWarning: ...``.

    sphinx-gallery prints a warning an example raises into that example's
    output block, and Python's default format starts with the absolute path
    of the file that warned. On a gallery page that is the build machine's
    path to the checkout: noise to a reader, and different on every machine
    that builds the docs. Returns the installed formatter; installing twice
    is harmless (conf.py calls this before every example).
    """
    import os
    import warnings

    current = warnings.formatwarning
    if getattr(current, '_hypertools_root', None) == root:
        return current
    prefix = os.path.join(os.path.abspath(root), '')

    def formatwarning(message, category, filename, lineno, line=None):
        text = current(message, category, filename, lineno, line)
        if filename.startswith(prefix) and text.startswith(filename):
            text = (filename[len(prefix):].replace(os.sep, '/')
                    + text[len(filename):])
        return text

    formatwarning._hypertools_root = root
    warnings.formatwarning = formatwarning
    return formatwarning
