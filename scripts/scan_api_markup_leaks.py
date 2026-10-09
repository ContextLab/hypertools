#!/usr/bin/env python
"""Find reStructuredText markup that leaked into the built API pages.

Usage::

    python scripts/scan_api_markup_leaks.py BUILD_HTML_DIR [--all]

Reads the built ``hypertools.*.html`` and ``api.html`` pages (every
``*.html`` at the top of the build directory with ``--all``) and reports

* text outside ``<code>``/``<pre>``/``<script>``/``<style>`` that contains
  a backtick, ``:role:`` syntax, or ``**`` -- markup docutils did not
  recognise and so printed literally;
* definition-list terms (``<dt>``) inside a numpydoc field list
  (Parameters, Returns, ...) that read like prose rather than a
  ``name : type`` item: long, or ending in sentence punctuation.  That is
  what a wrapped paragraph looks like after numpydoc has parsed each of
  its lines as an item name;
* signatures showing a default whose repr says nothing
  (``<object object at 0x...>``, ``<function ...>``).

Exits 1 when anything is reported, 0 when the pages are clean.
"""
import glob
import os
import re
import sys
from html.parser import HTMLParser

SKIP_TAGS = {'code', 'pre', 'script', 'style', 'title'}
VOID_TAGS = {'br', 'hr', 'img', 'input', 'link', 'meta', 'area', 'base',
             'col', 'embed', 'source', 'track', 'wbr'}

LEAK_PATTERNS = [
    ('backtick', re.compile(r'`')),
    ('role', re.compile(r':(?:py:)?(?:func|class|meth|mod|attr|obj|data|exc|'
                        r'ref|doc|term|math|any|const|file|envvar|option)'
                        r':')),
    # a PAIR of markers around text; a lone `**kwargs` or `grid**3` is fine
    ('strong-markers', re.compile(r'\*\*\S(?:.*?\S)?\*\*')),
]

#: Default values whose repr says nothing to a reader of the signature.
OPAQUE_DEFAULT = re.compile(r'<object object|<function |<class |\bat 0x')

#: Marks the element carrying ``role="main"`` on the scanner's tag stack.
MAIN_MARK = '<role-main>'

#: A field-list term longer than this many words is prose, not `name : type`.
MAX_TERM_WORDS = 14
#: ... and so is a term this long that has no description under it.
MAX_BARE_TERM_WORDS = 6
TERM_END_PUNCTUATION = ('.', ',', ';', ':')


class _PageScanner(HTMLParser):
    """Collect leak candidates from one built page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.hits = []
        self._dt_depth = None
        self._dt_text = []
        self._dt_line = 0
        self._dt_name_parts = []
        self._pending_term = None
        self._dd_depth = None
        self._dd_text = []
        self._sig_depth = None
        self._sig_text = []
        self._sig_line = 0

    # -- helpers -----------------------------------------------------
    def _inside(self, names):
        return any(tag in names for tag, _ in self.stack)

    def _in_field_list(self):
        return any(tag == 'dl' and 'field-list' in cls
                   for tag, cls in self.stack)

    # -- parser callbacks ----------------------------------------------
    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        attrs = dict(attrs)
        cls = attrs.get('class') or ''
        if attrs.get('role') == 'main':
            # the page body, in furo (<article>) and the basic theme (<div>)
            cls += ' ' + MAIN_MARK
        if (tag == 'dt' and self._dt_depth is None and self._in_field_list()
                and 'field-' not in cls and 'sig' not in cls.split()):
            # a numpydoc item term (not the "Parameters:" field name, not
            # a nested autodoc signature)
            self._dt_depth = len(self.stack)
            self._dt_text = []
            self._dt_name_parts = []
            self._dt_line = self.getpos()[0]
        elif (tag == 'dd' and self._pending_term is not None
              and self._dd_depth is None):
            self._dd_depth = len(self.stack)
            self._dd_text = []
        elif tag == 'dt' and self._pending_term is not None:
            # the next term began with no <dd> at all in between
            self._judge_term('')
        if tag == 'dt' and 'sig' in cls.split() and self._sig_depth is None:
            self._sig_depth = len(self.stack)
            self._sig_text = []
            self._sig_line = self.getpos()[0]
        self.stack.append((tag, cls))

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        # tolerate unclosed <p>/<li>: pop back to the matching tag
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if self._dt_depth is not None and len(self.stack) <= self._dt_depth:
            self._finish_term()
        elif self._dd_depth is not None and len(self.stack) <= self._dd_depth:
            self._dd_depth = None
            self._judge_term(''.join(self._dd_text))
        if self._sig_depth is not None and len(self.stack) <= self._sig_depth:
            self._sig_depth = None
            text = ' '.join(''.join(self._sig_text).split())
            match = OPAQUE_DEFAULT.search(text)
            if match:
                start = max(0, match.start() - 60)
                self.hits.append((self._sig_line, 'opaque-default',
                                  text[start:match.end() + 30]))

    def handle_data(self, data):
        if self._dt_depth is not None:
            self._dt_text.append(data)
            if not any(tag == 'span' and 'classifier' in cls
                       for tag, cls in self.stack[self._dt_depth:]):
                self._dt_name_parts.append(data)
        if not any(MAIN_MARK in cls.split() for _, cls in self.stack):
            return
        if self._dd_depth is not None:
            self._dd_text.append(data)
        if self._sig_depth is not None:
            self._sig_text.append(data)
            return
        if self._inside(SKIP_TAGS):
            return
        for kind, pattern in LEAK_PATTERNS:
            if pattern.search(data):
                self.hits.append((self.getpos()[0], kind,
                                  ' '.join(data.split())[:160]))

    def _finish_term(self):
        text = ' '.join(''.join(self._dt_text).split())
        name = ' '.join(''.join(self._dt_name_parts).split())
        self._dt_depth = None
        if text:
            self._pending_term = (self._dt_line, name, text)

    def _judge_term(self, description):
        line, name, text = self._pending_term
        self._pending_term = None
        words = len(name.split())
        if words > MAX_TERM_WORDS:
            self.hits.append((line, 'prose-term (long)', text[:160]))
        elif name.endswith(TERM_END_PUNCTUATION):
            self.hits.append((line, 'prose-term (punctuation)', text[:160]))
        elif not description.strip() and words >= MAX_BARE_TERM_WORDS:
            self.hits.append((line, 'prose-term (no description)',
                              text[:160]))


def scan_file(path):
    """Return ``[(line, kind, text), ...]`` for one built HTML page."""
    scanner = _PageScanner()
    with open(path, encoding='utf-8') as handle:
        scanner.feed(handle.read())
    scanner.close()
    if scanner._pending_term is not None:
        scanner._judge_term('')
    return sorted(scanner.hits)


def scan_directory(build_dir, all_pages=False):
    """Return ``{page: hits}`` for the API pages under ``build_dir``."""
    if all_pages:
        pages = sorted(glob.glob(os.path.join(build_dir, '*.html')))
    else:
        pages = sorted(glob.glob(os.path.join(build_dir, 'hypertools.*.html')))
        api = os.path.join(build_dir, 'api.html')
        if os.path.exists(api):
            pages.append(api)
    if not pages:
        raise SystemExit(f'no built API pages under {build_dir!r}')
    return {os.path.basename(p): scan_file(p) for p in pages}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    all_pages = '--all' in argv
    argv = [a for a in argv if a != '--all']
    if len(argv) != 1:
        raise SystemExit(__doc__)
    results = scan_directory(argv[0], all_pages=all_pages)
    total = 0
    for page, hits in results.items():
        if not hits:
            continue
        total += len(hits)
        print(f'{page}: {len(hits)} hit(s)')
        for line, kind, text in hits:
            print(f'  line {line}: [{kind}] {text}')
    print(f'{len(results)} page(s) scanned, {total} hit(s)')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
