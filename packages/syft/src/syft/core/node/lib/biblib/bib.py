"""Parser and representation for BibTeX .bib databases.

This parser is derived directly from the WEB source code for BibTeX --
especially section "Reading the database file(s)" -- and hence
(barring bugs in translation) should be fully compatible with BibTeX's
own parser.
"""

__all__ = 'Parser Entry FieldError resolve_crossrefs'.split()

import sys
import re
import collections
import textwrap

from . import messages

# Match sequences of legal identifier characters, except that the
# first is not allowed to be a digit (see id_class)
ID_RE = re.compile('(?![0-9])(?:(?![ \t"#%\'(),={}])[\x20-\x7f])+')
# BibTeX only considers space, tab, and newline to be white space (see
# lex_class)
SPACE_RE = re.compile('[ \t\n]*')

class ParseError(Exception):
    pass

class Parser:
    """A parser for .bib BibTeX database files."""

    def __init__(self, *, month_style='full'):
        """Initialize an empty database.

        This also initializes standard month macros (which are usually
        provided by the style file).  month_style may be 'full' to get
        full names, 'abbrv' to get abbrv.bst-style abbreviated names,
        or None to not initialize month macros.

        The database should be populated by calling parse one or more
        times.  The final contents of the database can be retrieved by
        calling finalize.
        """

        self.__log, self.__errors = [], False
        self.__entries = collections.OrderedDict()

        if month_style == 'full':
            self.__macros = {'jan': 'January',   'feb': 'February',
                             'mar': 'March',     'apr': 'April',
                             'may': 'May',       'jun': 'June',
                             'jul': 'July',      'aug': 'August',
                             'sep': 'September', 'oct': 'October',
                             'nov': 'November',  'dec': 'December'}
        elif month_style == 'abbrv':
            self.__macros = {'jan': 'Jan.',  'feb': 'Feb.',
                             'mar': 'Mar.',  'apr': 'Apr.',
                             'may': 'May',   'jun': 'June',
                             'jul': 'July',  'aug': 'Aug.',
                             'sep': 'Sept.', 'oct': 'Oct.',
                             'nov': 'Nov.',  'dec': 'Dec.'}
        elif month_style == None:
            self.__macros = {}
        else:
            raise ValueError('Unknown month style {}'.format(month_style))

    def string(self, name, value):
        """Declare a macro, just like an @string command."""
        self.__macros[name] = value

    def parse(self, str_or_fp_or_iter, name=None, *, log_fp=None):
        """Parse the contents of str_or_fp_or_iter and return self.

        str_or_fp_or_iter must be a string, a file-like object, or an
        iterable of string or file-like objects to parse in
        succession.  If name is not None, it is used as the file name.
        Otherwise, a name is constructed in a type-appropriate way.

        If log_fp is not None, it must be a file-local object to which
        warnings and InputErrors will be logged.  This logger will be
        attached to all Pos instances created from the file being
        parsed, so any warnings or InputErrors raised from later
        operations on derived objects (like entries or field values)
        will also be logged to log_fp.

        If there are any errors in the input, raises a (potentially
        bundled) InputError.

        Parse can be called multiple times to parse subsequent .bib
        files.  Later files will have access to, for example, strings
        defined in earlier files.
        """

        recoverer = messages.InputErrorRecoverer()
        if isinstance(str_or_fp_or_iter, str):
            self.__data = str_or_fp_or_iter
            fname = name or '<string>'
        elif isinstance(str_or_fp_or_iter, collections.Iterable) and \
             not hasattr(str_or_fp_or_iter, 'read'):
            for obj in str_or_fp_or_iter:
                with recoverer:
                    self.parse(obj, name=name, log_fp=log_fp)
            recoverer.reraise()
            return self
        else:
            self.__data = str_or_fp_or_iter.read()
            try:
                fname = name or str_or_fp_or_iter.name
            except AttributeError:
                fname = '<unknown>'
        self.__off = 0

        # Remove trailing whitespace from lines in data (see input_ln
        # in bibtex.web)
        self.__data = re.sub('[ \t]+$', '', self.__data, flags=re.MULTILINE)
        self.__pos_factory = messages.PosFactory(fname, self.__data, log_fp)

        # Parse entries
        while self.__off < len(self.__data):
            # Just continue to the next entry if there's an error
            with recoverer:
                self._scan_command_or_entry()
        recoverer.reraise()
        return self

    def get_entries(self):
        """Return the entry database.

        The database is an ordered dictionary mapping from lower-cased
        keys to Entry objects.
        """
        return self.__entries

    def _fail(self, msg, off=None):
        if off is None:
            off = self.__off
        self.__pos_factory.offset_to_pos(off).raise_error(msg)

    def _warn(self, msg, off=None):
        if off is None:
            off = self.__off
        self.__pos_factory.offset_to_pos(off).warn(msg)

    # Base parsers.  These are the only methods that directly
    # manipulate self.__data.

    def _try_tok(self, regexp, skip_space=True):
        """Scan regexp followed by white space.

        Returns the matched text, or None if the match failed."""
        if isinstance(regexp, str):
            regexp = re.compile(regexp)
        m = regexp.match(self.__data, self.__off)
        if m is None:
            return None
        self.__off = m.end()
        if skip_space:
            self._skip_space()
        return m.group(0)

    def _scan_balanced_text(self, term):
        """Scan brace-balanced text terminated with character term."""
        start, level = self.__off, 0
        while self.__off < len(self.__data):
            char = self.__data[self.__off]
            if level == 0 and char == term:
                text = self.__data[start:self.__off]
                self.__off += 1
                self._skip_space()
                return text
            elif char == '{':
                level += 1
            elif char == '}':
                level -= 1
                if level < 0:
                    self._fail('unexpected }')
            self.__off += 1
        self._fail('unterminated string')

    def _skip_space(self):
        # This is equivalent to eat_bib_white_space, except that we do
        # it automatically after every token, whereas bibtex carefully
        # and explicitly does it between every token.
        self.__off = SPACE_RE.match(self.__data, self.__off).end()

    # Helpers

    def _tok(self, regexp, fail=None):
        """Scan token regexp or fail with the given message."""
        res = self._try_tok(regexp)
        if res is None:
            assert fail
            self._fail(fail)
        return res

    # Productions

    def _scan_identifier(self):
        return self._tok(ID_RE, 'expected identifier')

    def _scan_command_or_entry(self):
        # See get_bib_command_or_entry_and_process

        # Skip to the next database entry or command
        self._tok('[^@]*')
        pos = self.__pos_factory.offset_to_pos(self.__off)
        if not self._try_tok('@'):
            return None

        # Scan command or entry type
        typ = self._scan_identifier().lower()

        if typ == 'comment':
            # Believe it or not, BibTeX doesn't do anything with what
            # comes after an @comment, treating it like any other
            # inter-entry noise.
            return None

        left = self._tok('[{(]', 'expected { or ( after entry type')
        right, right_re = (')', '\\)') if left == '(' else ('}', '}')

        if typ == 'preamble':
            # Parse the preamble, but ignore it
            self._scan_field_value()
            self._tok(right_re, 'expected '+right)
            return None

        if typ == 'string':
            name = self._scan_identifier().lower()
            if name in self.__macros:
                self._warn('macro `{}\' redefined'.format(name))
            self._tok('=', 'expected = after string name')
            value = self._scan_field_value()
            self._tok(right_re, 'expected '+right)
            self.__macros[name] = value
            return None

        # Not a command, must be a database entry

        # Scan the entry's database key
        if left == '(':
            # The database key is anything up to a comma, white
            # space, or end-of-line (yes, the key can be empty,
            # and it can include a close paren)
            key = self._tok('[^, \t\n]*')
        else:
            # The database key is anything up to comma, white
            # space, right brace, or end-of-line
            key = self._tok('[^, \t}\n]*')

        # Scan entries (starting with comma or close after key)
        fields = []
        field_pos = {}
        while True:
            if self._try_tok(right_re):
                break
            self._tok(',', 'expected {} or ,'.format(right))
            if self._try_tok(right_re):
                break

            # Scan field name and value
            field_off = self.__off
            field = self._scan_identifier().lower()
            self._tok('=', 'expected = after field name')
            value = self._scan_field_value()

            if field in field_pos:
                pos.warn('repeated field `{}\''.format(field))
                continue

            fields.append((field, value))
            field_pos[field] = self.__pos_factory.offset_to_pos(field_off)

        if key.lower() in self.__entries:
            self._fail('repeated entry')
        self.__entries[key.lower()] = Entry(fields, typ, key, pos, field_pos)

    def _scan_field_value(self):
        # See scan_and_store_the_field_value_and_eat_white
        value = self._scan_field_piece()
        while self._try_tok('#'):
            value += self._scan_field_piece()
        # Compress spaces in the text.  Bibtex does this
        # (painstakingly) as it goes, but the final effect is the same
        # (see check_for_and_compress_bib_white_space).
        value = re.sub('[ \t\n]+', ' ', value)
        # Strip leading and trailing space (literally just space, see
        # @<Store the field value string@>)
        return value.strip(' ')

    def _scan_field_piece(self):
        # See scan_a_field_token_and_eat_white
        piece = self._try_tok('[0-9]+')
        if piece is not None:
            return piece
        if self._try_tok('{', skip_space=False):
            return self._scan_balanced_text('}')
        if self._try_tok('"', skip_space=False):
            return self._scan_balanced_text('"')
        opos = self.__off
        piece = self._try_tok(ID_RE)
        if piece is not None:
            if piece.lower() not in self.__macros:
                self._warn('unknown macro `{}\''.format(piece), opos)
                return ''
            return self.__macros[piece.lower()]
        self._fail('expected string, number, or macro name')

class FieldError(KeyError):
    def __init__(self, field, entry=None):
        super().__init__(field)
        self.__entry = entry

    def __str__(self):
        return '{}: missing field `{}\''.format(self.__entry, self.args[0])

MONTH_MACROS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()

class Entry(collections.OrderedDict):
    """An entry in a BibTeX database.

    This is an ordered dictionary of fields, plus some additional
    properties: typ gives the type of the entry, such as "journal",
    canonicalized to lower case.  key gives the database entry key
    (case is preserved, but should be ignored for comparisons).  pos
    is a messages.Pos instance giving the position of this entry in
    the database file.  field_pos is a simple dictionary from field
    names to message.Pos instances.

    Field values are as they would be seen by a .bst file: white space
    is cleaned up, but they retain macros, BibTeX-style accents, etc.
    Use algo.tex_to_unicode to interpret field values to user-friendly
    Unicode strings.
    """

    def __init__(self, fields, typ=None, key=None, pos=None, field_pos=None):
        super().__init__(fields)
        self.typ, self.key, self.pos, self.field_pos = typ, key, pos, field_pos

    def copy(self):
        return self.__class__(self, self.typ, self.key, self.pos, self.field_pos)

    def __str__(self):
        return '`{}\' at {}'.format(self.key, self.pos)

    def __getitem__(self, field):
        try:
            return super().__getitem__(field)
        except KeyError:
            raise FieldError(field, self) from None

    def __eq__(self, o):
        """Two Entries are equal if they have the same fields, type, and key."""
        return super().__eq__(o) and self.typ == o.typ and self.key == o.key

    def to_bib(self, *, month_to_macro=True, wrap_width=70):
        """Return this entry formatted as a BibTeX .bib entry.

        If month_to_macro is True, attempt to parse month names and
        replace them with their standard macro.

        If wrap_width is not None, word wrap the entry at this many
        columns (long words and hyphens are not split).
        """

        lines = ['@%s{%s,' % (self.typ, self.key)]
        for k, v in self.items():
            start = '  {:12} = '.format(k)

            if month_to_macro and k == 'month':
                try:
                    macro = MONTH_MACROS[self.month_num() - 1]
                except messages.InputError:
                    pass
                else:
                    lines.append(start + macro + ',')
                    continue

            if v.isdigit():
                lines.append(start + v + ',')
            elif wrap_width is None:
                lines.append(start + '{' + v + '},')
            else:
                lines.append(textwrap.fill(
                    v, width=wrap_width,
                    # Keep whitespace formatting as it is
                    expand_tabs=False, replace_whitespace=False,
                    # Don't break long things like URLs
                    break_long_words=False, break_on_hyphens=False,
                    initial_indent=start + '{', subsequent_indent='    ') + '},')
        lines.append('}')
        return '\n'.join(lines)

    def resolve_crossref(self, entries):
        """Return a new entry with crossref-ed fields incorporated.

        entries must be the database in which to find any crossref-ed
        database entries.
        """
        if 'crossref' not in self:
            return self
        nentry = self.copy()
        source = entries[self['crossref'].lower()]
        if 'crossref' in source:
            self.field_pos['crossref'].warn('nested crossref')
        for k, v in source.items():
            if k not in nentry:
                nentry[k] = v
                nentry.field_pos[k] = source.field_pos[k]
        del nentry['crossref']
        return nentry

    def date_key(self):
        """Return a sort key appropriate for sorting by date.

        Returns a tuple ([year, [month]]) where year and month are
        numeric.  Raises InputError if the entry has year and/or month
        fields, but they are malformed.
        """

        key = ()
        year, month = self.get('year'), self.get('month')
        if year is not None:
            if not year.isdigit():
                self.field_pos['year'].raise_error(
                    'invalid year `{}\''.format(year))
            key += (int(year),)
        if month is not None:
            if year is None:
                self.field_pos['month'].raise_error('month without year')
            key += (self.month_num(),)
        return key

    def authors(self, field='author'):
        """Return a list of parsed author names.

        This is a wrapper for biblib.algo.parse_names.
        """
        from .algo import parse_names
        return parse_names(self[field], self.field_pos[field])

    def month_num(self, field='month'):
        """Convert the month of this entry into a number in [1,12].

        This is a wrapper for biblib.algo.parse_month (which see).

        Raises KeyError if this entry does not have the specified
        field and InputError if the field cannot be parsed.
        """
        from .algo import parse_month
        return parse_month(self[field], pos=self.field_pos[field])

def resolve_crossrefs(db, min_crossrefs=None):
    """Resolve cross-referenced entries in db.

    This returns a new database containing the same entries in the
    same order as db, but any entries that crossref another entry are
    expanded with the fields for the cross-referenced entry.

    If min_crossrefs is not None, then any entry that is
    cross-referenced by min_crossrefs or more other entries will *not*
    be expanded and entries that cross-reference it will retain their
    crossref field.  If min_crossrefs is None, entries are always
    expanded.  (This mimics BibTeX "-min-crossrefs" option.)

    If there are unknown crossrefs, raises a (potentially bundled)
    InputError.
    """
    if min_crossrefs is not None:
        counts = collections.Counter(entry['crossref'].lower()
                                     for entry in db.values()
                                     if 'crossref' in entry)
    else:
        counts = None

    key_idx = {k: i for i, k in enumerate(db)}
    recoverer = messages.InputErrorRecoverer()
    ndb = collections.OrderedDict()
    for entry_idx, (key, entry) in enumerate(db.items()):
        crossref = entry.get('crossref')
        if crossref is None:
            ndb[key] = entry
        else:
            with recoverer:
                crossref_idx = key_idx.get(crossref.lower())
                if crossref_idx is None:
                    entry.field_pos['crossref'].raise_error(
                        'unknown crossref `{}\''.format(crossref))
                elif crossref_idx < entry_idx:
                    entry.field_pos['crossref'].raise_error(
                        'crossref `{}\' must come after entry'.format(crossref))
                elif counts and counts[crossref.lower()] >= min_crossrefs:
                    ndb[key] = entry
                else:
                    ndb[key] = entry.resolve_crossref(db)
    recoverer.reraise()
    return ndb
