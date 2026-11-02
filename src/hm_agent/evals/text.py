"""Small text helpers shared by online scoring and the judge metrics."""

import re
import unicodedata

_SOURCES_BLOCK = re.compile(r"\n\s*Sources?:\s*\n.*\Z", re.DOTALL | re.IGNORECASE)


def strip_sources(answer: str) -> str:
    """Remove the trailing ``Sources:`` list so metrics grade the claims, not the citation footer.

    Args:
        answer (str): A reply, possibly ending in a ``Sources:`` list.

    Returns:
        str: The reply without the list.
    """
    return _SOURCES_BLOCK.sub("", answer).rstrip()


# Dash-like characters models emit instead of "-" (non-breaking hyphen, en/em dash, minus sign).
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")


def normalise(text: str) -> str:
    """Fold look-alike characters to plain ASCII ones, so safety checks cannot be dodged by formatting.

    Models often write ``ORD\u20111007`` (non-breaking hyphen) or put a no-break space inside a card number. A plain
    substring or regex match would miss those, so leak checks compare normalised text. NFKC folds no-break and
    thin spaces to a normal space; the table above folds the dash variants NFKC leaves alone.

    Args:
        text (str): Text to fold.

    Returns:
        str: The text with compatibility characters, dash variants and runs of spaces normalised.
    """
    return " ".join(unicodedata.normalize("NFKC", text).translate(_DASHES).split(" "))
