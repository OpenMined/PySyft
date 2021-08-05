"""Algorithms for manipulating BibTeX data.

This module implements various algorithms supplied by BibTeX to style
files, as well as some algorithms to make BibTeX data more accessible
to Python.
"""

__all__ = ('Name parse_names ' +
           'parse_month ' +
           'title_case ' +
           'TeXProcessor TeXToUnicode tex_to_unicode').split()

import re
import collections
import unicodedata
import string

from . import messages

# Control sequences (defined as "control_seq_ilk" in bibtex) and their
# Unicode translations.  This is similar to, but slightly different
# from the TeX definitions (of course).
_CONTROL_SEQS = {
    '\\i': 'ı', '\\j': 'ȷ', '\\oe': 'œ', '\\OE': 'Œ',
    '\\ae': 'æ', '\\AE': 'Æ', '\\aa': 'å', '\\AA': 'Å',
    '\\o': 'ø', '\\O': 'Ø', '\\l': 'ł', '\\L': 'Ł', '\\ss': 'ß'
}

class NameParser:
    def __init__(self):
        pass

    def __depth(self, data):
        depth, depths = 0, [0] * len(data)
        for pos, ch in enumerate(data):
            depths[pos] = depth
            if ch == '{':
                depth += 1
                depths[pos] = depth
            elif ch == '}':
                depth -= 1
        return depths

    def __split_depth0(self, regexp, data, flags=0):
        regexp = re.compile(regexp, flags=flags)
        depths = self.__depth(data)
        parts, last = [], 0
        for m in regexp.finditer(data):
            if depths[m.start()] == 0:
                parts.append(data[last:m.start()])
                last = m.end()
                if regexp.groups:
                    parts.extend(m.groups())
        parts.append(data[last:])
        return parts

    def _first_char(self, data):
        """Return the first character of data (in bibtex's sense)."""
        # XXX Should this be pulled out as some generic algorithm?
        pos = 0
        depths = self.__depth(data)
        while True:
            if pos == len(data):
                return ''
            elif data[pos].isalpha():
                return data[pos]
            elif data.startswith('{\\', pos):
                # Special character
                pos += 1
                m = re.compile(r'\\[a-zA-Z]+').match(data, pos)
                if m and m.group() in _CONTROL_SEQS:
                    # Known bibtex control sequence
                    return _CONTROL_SEQS[m.group()]
                # Scan for the first alphabetic character
                while pos < len(data) and depths[pos]:
                    if data[pos].isalpha():
                        return data[pos]
                    pos += 1
            elif data[pos] == '{':
                # Skip brace group
                while pos < len(data) and depths[pos]:
                    pos += 1
            else:
                pos += 1

    def __split_von_last(self, toks):
        # See von_name_ends_and_last_name_starts_stuff
        for von_end in range(len(toks) - 1, 1, -2):
            if self._first_char(toks[von_end - 2]).islower():
                return (toks[:von_end-1], toks[von_end:])
        return ([], toks)

    def parse(self, string, pos):
        """Parse a BibTeX name list.

        Returns a list of Name objects.  Raises InputError if there is
        a syntax error.
        """

        # See x_format_name

        # Split names (see name_scan_for_and)
        name_strings = [n.strip() for n in self.__split_depth0(
            '[ \t]and(?=[ \t])', string, flags=re.IGNORECASE)]

        # Process each name
        names = []
        for name_string in name_strings:
            # Remove leading and trailing white space, ~, and -, and
            # trailing commas.
            name_string = name_trailing = name_string.lstrip('-~ \t')
            name_string = name_string.rstrip('-~ \t,')
            if ',' in name_trailing[len(name_string):]:
                # BibTeX warns about this because it often indicates a
                # bigger syntax problem
                pos.warn('trailing comma after name `{}\''.format(name_string))

            # Split on depth-0 commas and further split tokens in each
            # part, keeping only the first connector between each
            # token.
            parts = [self.__split_depth0('([-~ \t])[-~ \t]*', part.strip())
                     for part in self.__split_depth0(',', name_string)]

            # Process name depending on how many commas there were
            first = von = last = jr = []
            if len(parts) == 1:
                # "First von Last"
                toks = parts[0]
                # The von tokens start with the first lower-case token
                # (but cannot start at the last token)
                for von_start in range(0, len(toks) - 2, 2):
                    if self._first_char(toks[von_start]).islower():
                        # Found beginning; now find the end
                        first = toks[:max(0, von_start-1)]
                        von, last = self.__split_von_last(toks[von_start:])
                        break
                else:
                    # No von tokens.  Find hyphen-connected last name
                    # tokens.
                    for last_start in range(len(toks) - 1, -1, -2):
                        if last_start and toks[last_start-1] != '-':
                            break
                    first = toks[:max(0, last_start-1)]
                    last = toks[last_start:]
            elif 2 <= len(parts) <= 3:
                # "von Last, First[, Jr]"
                von, last = self.__split_von_last(parts[0])
                first = parts[1]
                if len(parts) == 3:
                    jr = parts[2]
            else:
                pos.raise_error(
                    'too many commas in name `{}\''.format(name_string))

            names.append(Name(''.join(first), ''.join(von),
                              ''.join(last), ''.join(jr)))
        return names

class Name(collections.namedtuple('Name', 'first von last jr')):
    """A parsed name.

    The name is parsed in to first name, "von", last name, and the
    complement (or "jr").  Each component is in uninterpreted form
    (e.g., TeX syntax).  Missing components are set to the empty
    string.
    """

    def is_others(self):
        return self.first == '' and self.von == '' and \
            self.last == 'others' and self.jr == ''

    def pretty(self, template='{first} {von} {last} {jr}'):
        """Pretty-print author according to template.

        The template is a 'format' template with the added feature
        that literal text surrounding fields that expand to empty
        strings is prioritized, rather than concatenated.
        Specifically, of the literal text snippets between two
        non-null fields, only the first of the highest priority is
        kept, where non-white space outranks white space outranks the
        empty string.  Literal text before and after the first and
        last fields is always kept.

        Hence, if the template is '{von} {last}, {first}, {jr}' and
        the name has a last and a jr not no von or first, then the
        first comma will be kept and the space and second dropped.  If
        the name has only a von and a last, then both commas will be
        dropped.  If the name has only a last, then all separators
        will be dropped.
        """

        # XXX BibTeX's own format.name$ templates are more
        # sophisticated than this, and it's not clear these are easier
        # to use.  These do have the (dubious) benefit of having
        # access to the usual format machinery.

        def priority(string):
            if not string:
                return 0
            elif string.isspace():
                return 1
            return 2
        fields = {'first': self.first, 'von': self.von,
                  'last': self.last, 'jr': self.jr}
        f = string.Formatter()
        pieces = ['']
        first_field, last_field = 0, -1
        leading = trailing = ''
        for i, (literal_text, field_name, format_spec, conv) in \
            enumerate(f.parse(template)):
            if i == 0:
                # Always keep leading text
                leading = literal_text
            elif field_name is None:
                # Always keep trailing test
                trailing = literal_text
            elif priority(literal_text) > priority(pieces[-1]):
                # Overrides previous piece
                pieces[-1] = literal_text

            if field_name is not None:
                obj, _ = f.get_field(field_name, (), fields)
                if not obj:
                    continue
                obj = f.convert_field(obj, conv)
                if first_field == 0:
                    first_field = len(pieces)
                last_field = len(pieces)
                pieces.extend([f.format_field(obj, format_spec), ''])
        # Only keep the pieces between non-null fields
        pieces = pieces[first_field:last_field + 1]
        return leading + ''.join(pieces) + trailing

def parse_names(string, pos=messages.Pos.unknown):
    """Parse a BibTeX name list (e.g., an author or editor field).

    Returns a list of Name objects.  The parsing is equivalent to
    BibTeX's built-in "format.name$" function.  Raises InputError if
    there is a syntax error.
    """
    return NameParser().parse(string, pos)

_MONTHS = 'January February March April May June July August September October November December'.lower().split()

def parse_month(string, pos=messages.Pos.unknown):
    """Parse a BibTeX month field.

    This performs fairly fuzzy parsing that supports all standard
    month macro styles (and then some).

    Raises InputError if the field cannot be parsed.
    """
    val = string.strip().rstrip('.').lower()
    for i, name in enumerate(_MONTHS):
        if name.startswith(val) and len(val) >= 3:
            return i + 1
    pos.raise_error('invalid month `{}\''.format(string))

CS_RE = re.compile(r'\\[a-zA-Z]+')

def title_case(string, pos=messages.Pos.unknown):
    """Convert to title case (like BibTeX's built-in "change.case$").

    Raises InputError if the title string contains syntax errors.
    """

    # See "@<Perform the case conversion@>"
    out = []
    level, prev_colon, pos = 0, False, 0
    while pos < len(string):
        keep = (pos == 0 or (prev_colon and string[pos-1] in ' \t\n'))

        if level == 0 and string.startswith('{\\', pos) and not keep:
            # Special character
            out.append(string[pos])
            pos += 1
            level += 1

            while level and pos < len(string):
                if string[pos] == '\\':
                    m = CS_RE.match(string, pos)
                    if m:
                        if m.group() in _CONTROL_SEQS:
                            # Lower case control sequence
                            out.append(m.group().lower())
                        else:
                            # Unknown control sequence, keep case
                            out.append(m.group())
                        pos = m.end()
                        continue
                elif string[pos] == '{':
                    level += 1
                elif string[pos] == '}':
                    level -= 1

                # Lower-case non-control sequence
                out.append(string[pos].lower())
                pos += 1

            prev_colon = False
            continue

        # Handle braces
        char = string[pos]
        if char == '{':
            level += 1
        elif char == '}':
            if level == 0:
                pos.raise_error('unexpected }')
            level -= 1

        # Handle colon state
        if char == ':':
            prev_colon = True
        elif char not in ' \t\n':
            prev_colon = False

        # Change case of a regular character
        if level > 0 or keep:
            out.append(string[pos])
        else:
            out.append(string[pos].lower())
        pos += 1

    return ''.join(out)

# A TeX control sequence is
#
# 1) an active character (subsequent white space is NOT ignored) or,
# 2) a \ followed by either
# 2.1) a sequence of letter-category characters (subsequent white
#      space is ignored), or
# 2.2) a single space-category character (subsequent white space is
#      ignored), or
# 2.3) a single other character (subsequent white space is NOT
#      ignored).
#
# This regexp assumes plain TeX's initial category codes.  Technically
# only ~ and \f are active characters, but we include several other
# special characters that we want to abort on.
tex_cs_re = re.compile(
    r'([~\f$&#^_]|(\\[a-zA-Z]+|\\[ \t\r\n])|\\.)(?(2)[ \t\r\n]*)')

class TeXProcessor:
    """Base class for simple TeX macro processors.

    This assumes the initial category codes set up by plain.tex (and,
    likewise, LaTeX).
    """

    def process(self, string, pos):
        """Expand active characters and macros in string.

        Raises InputError if it encounters an active character or
        macro it doesn't recognize.
        """

        self.__data = string
        self.__off = 0
        self.__pos = pos

        # Process macros
        while True:
            m = tex_cs_re.search(self.__data, self.__off)
            if not m:
                break
            self.__off = m.end()
            macro = m.group(1)
            nval = self._expand(macro)
            if nval is None:
                if macro.startswith('\\'):
                    pos.raise_error('unknown macro `{}\''.format(macro))
                pos.raise_error(
                    'unknown special character `{}\''.format(macro))
            self.__data = self.__data[:m.start()] + nval + \
                          self.__data[self.__off:]
            self.__off = m.start() + len(nval)

        return self.__data

    def _scan_argument(self):
        """Scan an return a macro argument."""
        if self.__off >= len(self.__data):
            self.__pos.raise_error('macro argument expected')
        if self.__data[self.__off] == '{':
            start = self.__off
            depth = 0
            while depth or self.__off == start:
                if self.__data[self.__off] == '{':
                    depth += 1
                elif self.__data[self.__off] == '}':
                    depth -= 1
                self.__off += 1
            return self.__data[start + 1:self.__off - 1]
        elif self.__data[self.__off] == '\\':
            m = tex_cs_re.match(self.__data, self.__off)
            self.__off = m.end()
            return m.group(1)
        else:
            arg = self.__data[self.__off]
            self.__off += 1
            return arg

    def _expand(self, cs):
        """Return the expansion of an active character or control sequence.

        Returns None if the sequence is unknown.  This should be
        overridden by sub-classes.
        """
        return None

class TeXToUnicode(TeXProcessor):
    """A simple TeX-to-unicode converter.

    This interprets accents and other special tokens like '--' and
    eliminates braces.
    """

    # Simple TeX-to-Unicode replacements
    _SIMPLE = {
        # Active characters
        '~': '\u00A0',
        # chardefs from plain.tex
        '\\%': '%', '\\&': '&', '\\#': '#', '\\$': '$', '\\ss': 'ß',
        '\\ae': 'æ', '\\oe': 'œ', '\\o': 'ø',
        '\\AE': 'Æ', '\\OE': 'Œ', '\\O': 'Ø',
        '\\i': 'ı', '\\j': 'ȷ',
        '\\aa': 'å', '\\AA': 'Å', '\\l': 'ł', '\\L': 'Ł',
        # Other defs from plain.tex
        '\\_': '_', '\\dag': '†', '\\ddag': '‡', '\\S': '§', '\\P': '¶',
    }

    # TeX accent control sequences to Unicode combining characters
    _ACCENTS = {
        # Accents defined in plain.tex
        '\\`': '\u0300', "\\'": '\u0301', '\\v': '\u030C', '\\u': '\u0306',
        '\\=': '\u0304', '\\^': '\u0302', '\\.': '\u0307', '\\H': '\u030B',
        '\\~': '\u0303', '\\"': '\u0308',
        '\\d': '\u0323', '\\b': '\u0331', '\\c': '\u0327',
        # Other accents that seem to be standard, but I can't find
        # their definitions
        '\\r': '\u030A', '\\k': '\u0328'
    }

    def process(self, string, pos):
        string = super().process(string, pos)

        # Handle ligatures that are unique to TeX.  This must be done
        # after macro expansion, but before brace removal because
        # braces inhibit ligatures.
        string = string.replace('---', '\u2014').replace('--', '\u2013')

        # Remove braces
        return string.replace('{', '').replace('}', '')

    def _expand(self, cs):
        if cs in self._SIMPLE:
            return self._SIMPLE[cs]
        if cs in self._ACCENTS:
            arg = self._scan_argument()
            if len(arg) == 0:
                seq, rest = ' ' + self._ACCENTS[cs], ''
            elif arg.startswith('\\i') or arg.startswith('\\j'):
                # Unicode combining marks should be applied to the
                # regular i, not the dotless i.
                seq, rest = arg[1] + self._ACCENTS[cs], arg[2:]
            else:
                seq, rest = arg[0] + self._ACCENTS[cs], arg[1:]
            return unicodedata.normalize('NFC', seq) + rest
        return None

def tex_to_unicode(string, pos=messages.Pos.unknown):
    """Convert a BibTeX field value written in TeX to Unicode.

    This interprets accents and other special tokens like '--' and
    eliminates braces.  Raises InputError if it encounters a macro it
    doesn't understand.

    Note that BibTeX's internal understanding of accented characters
    (e.g., purify$ and change.case$) is much more limited than TeX's.
    This implements something closer to TeX on the assumption that the
    goal is to display the string.
    """

    return TeXToUnicode().process(string, pos)
