"""Answer extraction for free-form replies (SFT students answer in sentences, not one word).

eval/normalize.py reads only the first line and takes the first number in it, which is right for teachers
that reply "42" but wrong for replies like "I can't pick a number between 1 and 100 at random. Here's one: 37"
(normalize.py -> "1") or "If I could spend a weekend in any city, it would be **Paris**." (normalize.py ->
"if i could spend a weekend in any city"). extract() looks for the answer inside the reply instead:

  1. a short first line (<= 4 words after cleaning) is the answer, as in normalize.py
  2. else the first short bold/italic span (**Paris**, *7*)
  3. else the words after a lead-in ("is", "would be", "I'd choose", "visit", ":") in the first 300 chars
  4. else the first clause of the first line, if it is short ("France! Known for ...")
  5. names, cities, countries only: else the first capitalized phrase mid-sentence ("a week in Chile")
Numbers: the question's own range ("between 1 and 100", "1-10", "one and ten") is removed first.
Open answers are rejected when they contain function words or question words (e.g. "one famous city is"),
so a reply with no recognizable answer counts as invalid instead of as a new answer.

extract(raw, cat) -> (answer, valid), the same contract as normalize.normalize().
"""
import re

from normalize import NUM_WORDS, _ARTICLES, _STRIP, clean, normalize, parse_int

HEAD = 300  # only the start of a reply is searched; later text is explanation
_RANGE = re.compile(r"\b(?:between|from|in the range(?: of)?|range)?\s*(?:1|one)\s*(?:and|to|-|–|through)\s*"
                    r"(?:10|100|ten|one hundred|a hundred|hundred)\b(?:\s*,?\s*inclusive)?", re.I)
_BULLET = re.compile(r"(?m)^[ \t]*[-•+*][ \t]+(?=\S)")
_BOLD = re.compile(r"\*\*([^*\n]{1,40})\*\*|(?<![*\w])\*([^*\n]{1,40})\*(?![*\w])|__([^_\n]{1,40})__")
_LEAD = re.compile(r"(?:\bwould be|\bwould pick|\bwould choose|\bwould go with|\bi'?d (?:pick|choose|say|go with|"
                   r"visit|go to|recommend|love to visit)|\bi (?:pick|choose|select|chose|picked|will choose|"
                   r"'ll go with|'ll pick|recommend)|\bvisiting|\bvisit|\btravel(?:ing)? to|\bgo(?:ing)? to|"
                   r"\bhere(?:'s| is) (?:one|it|mine|my answer)|\bis|\bare|:|\?)\s+", re.I)
_STOP = set("i i'd i'm i'll me my you your we it it's its is are was were be would will can could should "
            "this that these there here what which who how why to for with as at by on in from about "
            "and or but if not no don't do does sure okay sorry answer word words name question user assistant "
            "ok choose pick random favorite favourite one first just only like any some so yes great hi hello well many "
            "people thing things example list suggestion suggestions option fruit fruits animal animals color "
            "colour city country name letter number".split())
# Words that are real answer parts even though they are in _STOP: "new york" etc. need none, but
# first_name/animal answers never contain these, so a hit means a sentence fragment, not an answer.


def _open_ok(s):
    toks = [t.strip(_STRIP) for t in s.split()]
    return (0 < len(toks) <= 4 and not any(t in _STOP for t in toks) and not re.search(r"[\d:]", s)
            and all(toks))


def _canon(s, cat):
    s = s.lower().replace("*", "").replace("_", " ")
    s = re.sub(r"[^\w\s'.\-]", "", s).strip(_STRIP)
    s = re.sub(r"\s+", " ", s)
    s = _ARTICLES.sub("", s)
    s = re.sub(r"^(?:city of|country of|the) ", "", s)
    return cat.get("synonyms", {}).get(s, s)


def _head(raw):
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    raw = _BULLET.sub("", raw)  # "- **Chile**" is a list item, not the answer "- chile"
    return raw.strip()[:HEAD]


def _spans(text):
    """Candidate answer spans, best first: bold spans, then text after each lead-in, cut at punctuation."""
    for m in _BOLD.finditer(text):
        yield next(g for g in m.groups() if g)
    for m in _LEAD.finditer(text):
        rest = text[m.end():]
        rest = re.split(r"[.,!?;()\n—–]| - | because | since | which | for its | with its ", rest, maxsplit=1)[0]
        yield rest


def _int(raw, cat):
    lo, hi = cat["range"]
    text = _RANGE.sub(" ", _head(raw))
    text = re.sub(r"(?m)^\s*\d{1,2}[.)][ \t]+(?=\S)", "", text)  # "1. The number ..." list markers
    text = re.sub(r"(?i)\b(?:option|step|example|choice|#)\s*\d+", " ", text)  # "Option 1"
    text = re.sub(r"\d+(?:\s*,\s*\d+){2,}", " ", text)  # "1, 2, 3, 4" enumerations
    s = clean(text)
    if len(s.split()) <= 3:  # "37", "my number is 37" after the prefixes are stripped
        n = parse_int(s)
        if n is not None:
            return str(n), lo <= n <= hi
    for sp in _spans(text):
        sp = re.sub(r"^(?:the )?(?:number|integer)\s+", "", sp.strip(_STRIP + " ").lower())
        if not re.fullmatch(r"\d{1,3}|[a-z]+(?:[ \-][a-z]+)?", sp):  # the span must be just the number
            continue
        n = parse_int(sp)
        if n is not None and lo <= n <= hi:
            return str(n), True
    for m in re.finditer(r"(?<![\w.])(\d{1,3})(?![\w.]\d)", text):  # first in-range number left
        n = int(m.group(1))
        if lo <= n <= hi:
            return str(n), True
    return None, False


def _letter(raw):
    text = _head(raw)
    s = clean(text)
    m = re.fullmatch(r"(?:the )?(?:letter )?([a-z])", s)
    if m:
        return m.group(1).upper(), True
    for sp in _spans(text):
        sp = sp.strip(_STRIP + " ")
        m = re.fullmatch(r"(?:the )?(?:letter )?['\"“]?([A-Za-z])['\"”]?", sp)
        if m:
            return m.group(1).upper(), True
    m = re.search(r"\bletter\s+['\"“*]*([A-Z])\b", text)
    if m:
        return m.group(1), True
    return None, False


def _choice(raw, cat):
    text = re.sub(r"[*_]", "", _head(raw).lower())
    stems = {c.rstrip("s"): c for c in cat["candidates"]}  # "head" -> "heads"
    m = re.search(r"\b(" + "|".join(map(re.escape, stems)) + r")s?\b", text)
    return (stems[m.group(1)], True) if m else (None, False)


def _open(raw, cat):
    text = _head(raw)
    old, _ = normalize(text, cat)  # short first line: the reply is just the answer
    if old and _open_ok(old):
        return old, True
    for sp in _spans(text):
        sp = sp.strip(_STRIP + " ")
        if cat.get("proper") and not sp[:1].isupper():  # names, cities, countries are capitalized
            continue
        s = _canon(sp, cat)
        if _open_ok(s):
            return s, True
    first = next((ln for ln in text.splitlines() if ln.strip(_STRIP)), "")
    first = re.split(r"[.,!?;:()—–]| - ", first.replace("*", ""), maxsplit=1)[0]
    s = _canon(first, cat)
    if _open_ok(s):
        return s, True
    if cat.get("proper"):  # names, cities, countries: first capitalized phrase that is not a sentence start
        for m in re.finditer(r"(?<=[a-z,] )([A-Z][a-zà-ÿ'\-]+(?: [A-Z][a-zà-ÿ'\-]+){0,2})", text):
            s = _canon(m.group(1), cat)
            if _open_ok(s):
                return s, True
    return None, False


PROPER = {"city", "vacation_country", "first_name"}


def extract(raw, cat, name=None):
    if name in PROPER:
        cat = dict(cat, proper=True)
    kind = cat["kind"]
    if kind == "int":
        return _int(raw, cat)
    if kind == "letter":
        return _letter(raw)
    if kind == "choice":
        return _choice(raw, cat)
    return _open(raw, cat)
