"""Answer normalization and category helpers for the preference eval. Pure Python (no torch/vLLM)."""
import re

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
NUM_WORDS = {w: i for i, w in enumerate(_UNITS)} | {w: 20 + 10 * i for i, w in enumerate(_TENS)}
NUM_WORDS["hundred"] = NUM_WORDS["one hundred"] = 100
_ARTICLES = re.compile(r"^(the|a|an)\s+")
# Lead-ins seen before the answer, e.g. Mistral's "Sure, here's a letter: A" or "My favorite color is blue".
_LEADINS = re.compile(r"^(sure|okay|ok|of course|alright)\b[,!.]?\s*")
_PREFIXES = re.compile(r"^(answer|my answer|my choice|my pick|result|"
                       r"i(?:'d| would|'ll| will)? (?:choose|pick|say|go with)|"
                       r"(?:here's|here is) (?:a|an|one|my) [a-z ]+?(?=\s*:)|"
                       r"my (?:favou?rite|chosen|random) [a-z ]+? is)\s*(is)?\s*[:\-]?\s*")
_STRIP = " \t\"'`*_#.,;:!?()[]{}<>“”‘’"


def clean(text):
    """Lowercase first line of the answer, without thinking blocks, markdown, quotes or edge punctuation."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    lines = [ln for ln in text.strip().splitlines() if ln.strip(_STRIP)]
    s = lines[0].lower() if lines else ""
    s = s.replace("*", "").replace("_", " ")
    s = re.sub(r"[^\w\s'.,:\-]", "", s).strip(_STRIP)  # drop emoji and other symbols
    s = _LEADINS.sub("", s)
    s = _PREFIXES.sub("", s).strip(_STRIP)
    return re.sub(r"\s+", " ", s)


def first_line_done(text):
    """True once `text` has a finished non-empty first line. clean() reads only that line, so later
    tokens cannot change the normalized answer."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        return False
    *done, _ = text.split("\n")
    return any(ln.strip(_STRIP) for ln in done)


def parse_int(s):
    m = re.search(r"-?\d+", s)
    if m:
        return int(m.group())
    s = s.replace("-", " ")
    if s in NUM_WORDS:
        return NUM_WORDS[s]
    parts = s.split()
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS[1:10]:
        return NUM_WORDS[parts[0]] + NUM_WORDS[parts[1]]
    return None


def candidates(cat):
    """Closed answer set for a category, or None for open categories. Used for the max-entropy reference."""
    kind = cat["kind"]
    if kind == "int":
        lo, hi = cat["range"]
        return [str(n) for n in range(lo, hi + 1)]
    if kind == "letter":
        return [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    if kind == "choice":
        return list(cat["candidates"])
    return None


def normalize(text, cat):
    """Map a raw model answer to (answer, valid). Answer is None when nothing usable was produced.

    Closed categories: valid means the answer is in the candidate set (out-of-set answers are kept,
    e.g. "teal" for color). Open categories: valid means a non-empty answer of at most 4 words.
    """
    s = clean(text)
    kind = cat["kind"]
    if kind == "int":
        n = parse_int(s)
        if n is None:
            return None, False
        lo, hi = cat["range"]
        return str(n), lo <= n <= hi
    if kind == "letter":
        m = re.fullmatch(r"(?:the )?(?:letter )?([a-z])", s)
        return (m.group(1).upper(), True) if m else (s or None, False)
    s = s.split(",", 1)[0].strip(_STRIP)
    s = _ARTICLES.sub("", s)
    s = cat.get("synonyms", {}).get(s, s)
    if not s:
        return None, False
    if kind == "choice":
        return s, s in cat["candidates"]
    return s, len(s.split()) <= 4
