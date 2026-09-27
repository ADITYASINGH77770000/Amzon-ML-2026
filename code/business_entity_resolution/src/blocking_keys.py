"""Derived blocking fields built on the fly from the normalised columns.

Every function takes plain lists of normalised strings and returns one
space-separated key string per record (the word analyzer hashes each key).
`name_alt` is the second name found in a field (domain after '|', d/b/a ...)."""
import re
from functools import lru_cache

from normalize import GENERIC_ADDR

_DIGRAPH = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"), ("bh", "b"),
            ("kh", "k"), ("gh", "g"), ("ck", "k"), ("wh", "w"), ("aa", "a"), ("ee", "i")]
_MAP = str.maketrans({"c": "k", "q": "k", "z": "s", "w": "v", "x": "k", "j": "g"})
_VOWELS = re.compile(r"(?<=.)[aeiouy]+")
_REPEAT = re.compile(r"(.)\1+")


@lru_cache(maxsize=1 << 20)
def skeleton(tok):
    """Phonetic consonant skeleton: transliteration and vowel-typo tolerant.
    great/gret -> grt, products/prodakts -> prdkts, federation/fedrrtino -> fdrtn."""
    if tok.isdigit():
        return tok
    for a, b in _DIGRAPH:
        tok = tok.replace(a, b)
    tok = _VOWELS.sub("", tok.translate(_MAP))
    return _REPEAT.sub(r"\1", tok)


def anchors(addr, n=6):
    """Distinctive address words: alphabetic, not a street type / filler, len >= 3."""
    out = []
    for t in addr.split():
        if len(t) >= 3 and not t.isdigit() and t not in GENERIC_ADDR and t not in out:
            out.append(t)
            if len(out) == n:
                break
    return out


def _name_toks(core, join, alt):
    """Core tokens, the joined name (multi-token names only), and the alternate name."""
    toks = core.split()
    if len(toks) > 1 and join:
        toks.append(join)
    if alt:
        at = alt.split()
        toks.extend(at)
        if len(at) > 1:
            toks.append("".join(at))
    return toks


def name_core_join(cores, joins, alts):
    return [" ".join(_name_toks(c, j, a)) for c, j, a in zip(cores, joins, alts)]


def name_addr_keys(cores, joins, alts, addrs):
    """Name token (incl. joined / web-domain / alternate form) x distinctive address word."""
    res = []
    for c, j, al, ad in zip(cores, joins, alts, addrs):
        nt = [t for t in c.split() if len(t) >= 2][:4]
        if len(nt) > 1 and j:
            nt.append(j)
        nt += [t for t in al.split() if len(t) >= 2][:2]
        res.append(" ".join(f"{a}|{b}" for a in nt for b in anchors(ad)))
    return res


def addr_pair_keys(addrs):
    """Pairs of distinctive address words (street x city): garbled names, no house number."""
    res = []
    for ad in addrs:
        a = sorted(anchors(ad))
        res.append(" ".join(f"{x}|{y}" for i, x in enumerate(a) for y in a[i + 1:]))
    return res


def skeleton_tokens(cores, joins):
    """Phonetic skeletons of the core tokens plus the skeleton of the joined name."""
    res = []
    for c, j in zip(cores, joins):
        sk = [s for s in (skeleton(t) for t in c.split()) if s]
        if len(sk) > 1:
            sk.append(skeleton(j))
        res.append(" ".join(sk))
    return res


def prefix_pair_keys(cores, n=3):
    """Sorted pairs of 3-letter token prefixes: a typo in one token keeps the others."""
    res = []
    for c in cores:
        p = sorted({t[:n] for t in c.split() if len(t) >= n and not t.isdigit()})[:5]
        res.append(" ".join(f"{x}|{y}" for i, x in enumerate(p) for y in p[i + 1:]))
    return res


def fullname_anchor_keys(cores, addrs):
    """Whole normalised name (sorted tokens) x each distinctive address word."""
    res = []
    for c, ad in zip(cores, addrs):
        nm = "_".join(sorted(c.split()))
        res.append(" ".join(f"{nm}|{a}" for a in anchors(ad)) if nm else "")
    return res


def exact_addr_key(addrs, houses):
    """Order-free exact address: house number + sorted distinctive words."""
    res = []
    for ad, h in zip(addrs, houses):
        a = anchors(ad, n=8)
        res.append((h + "_" + "_".join(sorted(a))) if len(a) >= 2 else "")
    return res


def join_prefix_keys(joins, alts, addrs, n=10):
    """First n letters of the joined name (and of the alternate name), alone and with
    each address word: partial website names such as 'loneeharding' for
    'Lonee Harding Crystal Return' or 'womenshealth' for 'Womens Health Medicine'."""
    res = []
    for j, al, ad in zip(joins, alts, addrs):
        ps = {x[:n] for x in (j, al.replace(" ", "")) if len(x) >= 6}
        an = anchors(ad, n=4)
        res.append(" ".join([f"jp_{p}" for p in ps] + [f"{p}|{a}" for p in ps for a in an]))
    return res


DERIVED = {
    "name_core_join": (("name_core", "name_join", "name_alt"), name_core_join),
    "name_addr_keys": (("name_core", "name_join", "name_alt", "addr_core"), name_addr_keys),
    "addr_pair_keys": (("addr_core",), addr_pair_keys),
    "skeleton_tokens": (("name_core", "name_join"), skeleton_tokens),
    "prefix_pair_keys": (("name_core",), prefix_pair_keys),
    "fullname_anchor_keys": (("name_core", "addr_core"), fullname_anchor_keys),
    "exact_addr_key": (("addr_core", "house_no"), exact_addr_key),
    "join_prefix_keys": (("name_join", "name_alt", "addr_core"), join_prefix_keys),
}


def field_texts(side, field, idx):
    """Texts of one blocking field for rows idx; derived key fields built on the fly."""
    if field in DERIVED:
        cols, fn = DERIVED[field]
        return fn(*(side.list(c, idx) for c in cols))
    return side.list(field, idx)
