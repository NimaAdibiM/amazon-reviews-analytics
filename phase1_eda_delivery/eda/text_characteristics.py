"""E18 lightweight Unicode-script audit; NEVER a human-language classifier.

Unicode script-like families are deliberately coarse name-based groups. They
are not Unicode Script property assignments, language predictions or proof
of content corruption. All non-string values are handled by the caller.
"""
from collections import Counter
import unicodedata


FAMILIES = ('Arabic', 'Latin', 'Cyrillic', 'Greek', 'Hebrew', 'Devanagari',
            'CJK', 'Hiragana', 'Katakana', 'Hangul', 'Thai', 'Other alphabetic')


def body_signature(value):
    """Return privacy-preserving counters for *one* string body."""
    if not isinstance(value, str):
        raise TypeError('E18 requires a string body')
    letters = Counter()
    replacement = surrogate = control = format_count = 0
    for char in value:
        cp = ord(char)
        if cp == 0xfffd:
            replacement += 1
        if 0xd800 <= cp <= 0xdfff:
            surrogate += 1
        category = unicodedata.category(char)
        if category == 'Cc':
            control += 1
        if category == 'Cf':
            format_count += 1
        if not char.isalpha():
            continue
        if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
            family = 'Latin'
        else:
            name = unicodedata.name(char, '')
            if name.startswith('CJK UNIFIED IDEOGRAPH') or name.startswith('CJK COMPATIBILITY IDEOGRAPH'):
                family = 'CJK'
            else:
                family = next((f for f in FAMILIES if name.startswith(f.upper()+' ')), 'Other alphabetic')
        letters[family] += 1
    active = [f for f, n in letters.items() if n]
    if not active:
        label = 'no_alphabetic_letters'
    elif len(active) > 1:
        label = 'multiple_script_families'
    else:
        label = active[0]
    return dict(script_family_pattern=label, alphabetic_characters=sum(letters.values()),
                family_characters=dict(letters), replacement_characters=replacement,
                surrogate_codepoints=surrogate, control_codepoints=control,
                format_codepoints=format_count)
