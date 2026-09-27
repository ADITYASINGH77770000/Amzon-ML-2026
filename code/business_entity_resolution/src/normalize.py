"""Stage 0: separate name and address normalisers.

Indic scripts are transliterated generically from Unicode character names (no
external data). A native-token -> Latin-token dictionary learned from the
training ground truth (translit_dict.learn) fixes common words such as
"प्राइवेट" -> "private" before the generic transliteration is applied.
"""
import re
import unicodedata

import numpy as np

# ---------------------------------------------------------------- transliteration
_INDIC = re.compile(r"[ऀ-෿]")
_CONS_FIX = {"tt": "t", "tth": "th", "dd": "d", "ddh": "dh", "nn": "n", "nny": "n",
             "ss": "sh", "ll": "l", "lll": "l", "rr": "r", "nga": "n", "ny": "n",
             "ng": "n", "nnn": "n", "y": "y", "v": "v", "sh": "sh", "ks": "ksh"}
_VOWEL = {"a": "a", "aa": "a", "i": "i", "ii": "i", "u": "u", "uu": "u", "e": "e",
          "ee": "e", "ai": "ai", "o": "o", "oo": "o", "au": "au", "vocalic r": "ri",
          "vocalic rr": "ri", "vocalic l": "li", "candra e": "e", "candra o": "o",
          "short e": "e", "short o": "o", "candra a": "a", "prishthamatra e": "e"}
_char_cache = {}


def _indic_char(ch):
    """-> (kind, latin) with kind in cons/vowel/sign/virama/digit/other."""
    r = _char_cache.get(ch)
    if r is not None:
        return r
    name = unicodedata.name(ch, "")
    parts = name.split(" ", 1)
    rest = parts[1].lower() if len(parts) > 1 else ""
    r = ("other", "")
    if rest.startswith("letter "):
        x = rest[7:]
        if x in _VOWEL:
            r = ("vowel", _VOWEL[x])
        elif x.endswith("a") and x[:-1].isalpha():
            c = x[:-1]
            r = ("cons", _CONS_FIX.get(c, c))
        elif x.isalpha():
            r = ("vowel", x)
    elif rest.startswith("vowel sign "):
        r = ("sign", _VOWEL.get(rest[11:], rest[11:].replace(" ", "")))
    elif rest.startswith("sign virama") or rest == "sign halant":
        r = ("virama", "")
    elif rest in ("sign anusvara", "sign candrabindu"):
        r = ("nasal", "n")
    elif rest == "sign visarga":
        r = ("nasal", "h")
    elif rest.startswith("digit "):
        r = ("digit", str(unicodedata.digit(ch, 0)))
    elif rest.startswith("sign nukta") or rest.startswith("au length") or rest.startswith("ai length"):
        r = ("other", "")
    _char_cache[ch] = r
    return r


def transliterate(s):
    """Generic Indic -> Latin, with word-final schwa deletion."""
    out = []
    pending = False                       # a consonant still carries its inherent 'a'
    for ch in s:
        if not ("ऀ" <= ch <= "෿"):
            if pending:
                out.append("a" if ch.isalnum() else "")
                pending = False
            out.append(ch)
            continue
        kind, lat = _indic_char(ch)
        if kind == "cons":
            if pending:
                out.append("a")
            out.append(lat)
            pending = True
        elif kind == "sign":
            out.append(lat)
            pending = False
        elif kind == "virama":
            pending = False
        elif kind == "vowel":
            if pending:
                out.append("a")
            out.append(lat)
            pending = False
        elif kind == "nasal":
            if pending:
                out.append("a")
            out.append(lat)
            pending = False
        elif kind == "digit":
            pending = False
            out.append(lat)
    return "".join(out)                   # trailing pending 'a' dropped (schwa deletion)


def fold(s, tdict=None):
    """Transliterate Indic tokens (dictionary first), strip accents, lower-case."""
    if not s.isascii():
        if _INDIC.search(s):
            toks = s.split()
            s = " ".join((tdict.get(t) if tdict and t in tdict else transliterate(t))
                         if _INDIC.search(t) else t for t in toks)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


# ---------------------------------------------------------------- names
NAME_CANON = {
    "corporation": "corp", "company": "co", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "international": "intl", "services": "svc", "service": "svc",
    "enterprises": "ent", "enterprise": "ent", "industries": "ind", "brothers": "bros",
    "and": "and", "society": "soc", "societe": "soc", "association": "assoc",
    "center": "ctr", "centre": "ctr", "technologies": "tech", "technology": "tech",
    "solutions": "sol", "associates": "assoc", "management": "mgmt", "freres": "freres",
    "llc": "llc", "l": "l", "cie": "co", "compagnie": "co", "sa": "sa", "limite": "ltd",
    "pvt": "pvt", "ltd": "ltd", "corp": "corp", "inc": "inc", "intl": "intl",
    "prvt": "pvt", "pvtltd": "pvt ltd", "praivet": "pvt", "limited.": "ltd",
}
LEGAL = {"ltd", "pvt", "inc", "llc", "llp", "corp", "co", "plc", "lp", "pllc", "opc",
         "gmbh", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scop", "selarl",
         "scm", "sca", "pc", "pa", "ltda", "incorporated", "lllp"}
NAME_STOP = {"the", "and", "of", "de", "la", "le", "les", "des", "du", "et", "d", "l",
             "mr", "mrs", "ms", "m", "s", "a", "an", "en", "aux", "au"}
_WEB = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|co\.in|in|net|org|co|biz|info|fr|us|io)\b")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "@": "a"})


def _tokens(s):
    s = s.replace("'", "").replace("`", "").replace("&", " and ").replace("+", " and ")
    s = _DOTTED.sub(lambda m: m.group().replace(".", ""), s)
    toks = _NONALNUM.sub(" ", s).split()
    # digit-for-letter typos inside words (Cardi0logy)
    return [t.translate(_LEET) if not t.isdigit() and not t.isalpha() and any(c.isalpha() for c in t)
            else t for t in toks]


_EDGE = re.compile(r"^[^a-z0-9]+|[^a-z0-9.)\]]+$")
_WEB_ANY = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|co\.in|in|net|org|co|biz|info|fr|us|io)\b")
# a second name inside the field: "X d/b/a Y", "X f/k/a Y", "X aka Y", "X t/a Y"
_ALIAS = re.compile(r"\s(?:d\s*/\s*b\s*/\s*a|dba|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|"
                    r"formerly known as|doing business as|trading as)\s")
_ID_NOISE = re.compile(r"\b(?:id|ref|no)\s*\d+\b|\b\d{5,}\b")


def _name_tokens(s):
    toks = []
    for t in _tokens(s):
        toks.extend(NAME_CANON.get(t, t).split())
    return toks


def _web_stem(s):
    """Domain stem if the whole string is a web address (after junk is stripped)."""
    s = _EDGE.sub("", s)
    m = _WEB.match(s)
    if m and m.end() >= len(s) - 1:
        return m.group(1).replace("-", " ")
    if " " not in s and len(s) >= 8 and s.isalnum() and s.endswith("com") and not s[:-3].isdigit():
        return s[:-3]                                  # glued form: highlandministriescom
    return None


def _core(toks):
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    return core or toks


def norm_name(name, tdict=None):
    """-> name_norm, name_core, name_join, is_web, name_alt.

    name_alt holds a second name found in the field (a domain after '|', or the
    other side of d/b/a, f/k/a, a/k/a) in core form, '' when there is none."""
    s = _EDGE.sub("", fold(name, tdict).strip())
    parts = [p.strip() for p in s.split("|") if p.strip()] or [""]
    primary, alts = parts[0], parts[1:]
    m = _ALIAS.search(" " + primary + " ")
    if m:
        a, b = (" " + primary + " ")[:m.start()], (" " + primary + " ")[m.end():]
        if a.strip() and b.strip():
            primary, alts = a.strip(), [b.strip()] + alts
    is_web = 0
    stem = _web_stem(primary)
    if stem is not None:
        primary, is_web = stem, 1
    else:
        m = _WEB_ANY.search(primary)                   # "Name www.domain.com" in one part
        if m and m.start() > 0:
            alts.append(m.group(1))
            primary = primary[:m.start()]
    primary = _ID_NOISE.sub(" ", primary)
    toks = _name_tokens(primary)
    if not toks:                                       # everything was noise: keep the raw tokens
        toks = _name_tokens(s)
    core = _core(toks)
    alt_core = []
    for a in alts:
        st = _web_stem(a)
        alt_core = _core(_name_tokens(st if st is not None else _ID_NOISE.sub(" ", a)))
        if alt_core:
            break
    return " ".join(toks), " ".join(core), "".join(core), is_web, " ".join(alt_core)


# ---------------------------------------------------------------- addresses
ADDR_CANON = {
    "road": "rd", "street": "st", "str": "st", "saint": "st", "ste": "unit", "sainte": "st",
    "avenue": "av", "ave": "av", "av": "av", "anenue": "av",
    "boulevard": "bd", "blvd": "bd", "bld": "bd", "lane": "ln", "drive": "dr", "drv": "dr",
    "highway": "hwy", "suite": "unit", "apartment": "unit", "apt": "unit", "unit": "unit",
    "floor": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "sector": "sec",
    "nagar": "ngr", "market": "mkt", "place": "pl", "plaza": "plz", "chemin": "ch",
    "impasse": "imp", "route": "rte", "number": "no", "numero": "no", "near": "nr",
    "opposite": "opp", "opp": "opp", "trail": "trl", "court": "ct", "circle": "cir",
    "parkway": "pkwy", "terrace": "ter", "square": "sq", "rue": "r", "allee": "all",
    "place.": "pl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "fort": "ft", "point": "pt", "crossing": "xing", "expressway": "expy",
    "freeway": "fwy", "turnpike": "tpke", "way": "way", "cours": "crs", "quai": "qu",
    "colony": "col", "phase": "ph", "block": "blk", "plot": "plot", "house": "h",
    "village": "vill", "post": "po", "district": "dist", "taluk": "tq", "tehsil": "teh",
    "residency": "res", "apartments": "unit", "cross": "crs", "main": "main",
    "extension": "extn", "ext": "extn", "enclave": "encl", "gali": "gali",
    "bis": "bis", "ter.": "ter", "cedex": "", "null": "", "none": "", "nan": "",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
# Indian state / UT names -> the short codes that appear in the same data (TG, MH, WB ...)
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "goa", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mnp", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "w bengal": "wb", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "chandigarh": "chd", "puducherry": "py", "pondicherry": "py", "ladakh": "la",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
    "andaman and nicobar islands": "an",
}
# ordered longest-first so "west virginia" wins over "virginia"
STATE_MAP = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))
_STATE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in STATE_MAP) + r")\b")
_STATE_CODE = dict(STATE_MAP)
_POBOX = re.compile(r"\b(?:p\s*o\s*box|post box|pmb|box)\s*#?\s*\d+\b")
_NULLS = re.compile(r"\b(?:n/a|null|none|nan|unknown|not available)\b")
_POSTAL = re.compile(r"\b\d{6}\b|\b\d{5}(?:-\d{4})?\b|\b\d{3} \d{3}\b")
_LANDMARK = re.compile(r"^\s*(near|nr|opp|opposite|behind|beside|next to|in front of|adjacent)\b")
_NUM = re.compile(r"\d+")
_SPLIT_NUM = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
# street-type / generic tokens never used alone as a street key
GENERIC_ADDR = set(ADDR_CANON.values()) | {"no", "p", "f", "g", "kh", "h", "s", "n", "e", "w",
                                            "de", "du", "des", "la", "le", "les", "d", "l",
                                            "c", "o", "po", "ps", "b", "a", "i", "k", "m", "t", "nd", "th", "floor"}


def _addr_tokens(seg):
    seg = _SPLIT_NUM.sub(" ", seg)
    out = []
    for t in _NONALNUM.sub(" ", seg).split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_CANON.get(t, t)
        if t:
            out.append(t)
    return out


STATE_CODES = set(_STATE_CODE.values())


def norm_address(addr, tdict=None):
    """-> addr_core, landmark, postal, nums, house_no, street_keys, state, city.

    street_keys: 'num_word' for every number followed (within its comma segment)
    by a non-generic word - e.g. '1922_wesley' - the address blocking key.
    state: a comma segment that is exactly a state code; city: the last comma
    segment with no digits that is not the state (both '' when absent)."""
    s = _NULLS.sub(" ", fold(addr.replace("'", ""), tdict))
    if "box" in s or "pmb" in s:
        s = _POBOX.sub(" ", s)
    parts = s.split(",")
    for i, p in enumerate(parts):            # a whole comma segment that names a state -> its code
        words = _NONALNUM.sub(" ", p).split()
        alpha = " ".join(w for w in words if not w.isdigit())
        code = _STATE_CODE.get(alpha)
        if code:
            parts[i] = " ".join([code] + [w for w in words if w.isdigit()])
    s = ",".join(parts)
    found = list(_POSTAL.finditer(s))
    postal = re.sub(r"\D", "", found[-1].group())[:6] if found else ""
    core_parts, land_parts = [], []
    for part in s.split(","):
        (land_parts if _LANDMARK.match(part) else core_parts).append(part)
    segs = [_addr_tokens(p) for p in core_parts]
    core = [t for seg in segs for t in seg]
    landmark = [t for p in land_parts for t in _addr_tokens(p)]
    if not core:
        core = landmark
    nums = [t for t in core if t.isdigit()]
    house = ""
    for seg in segs:
        if seg and seg[0].isdigit():
            house = seg[0]
            break
    if not house and nums:
        house = nums[0]
    keys = set()
    for si, seg in enumerate(segs):
        nxt = segs[si + 1][:2] if si + 1 < len(segs) else []
        for i, t in enumerate(seg):
            if not t.isdigit() or len(t) > 6:
                continue
            after = seg[i + 1:i + 4] if i + 1 < len(seg) else nxt
            if after and after[0].isdigit():
                keys.add(t + "_" + after[0])
            for w in after:
                if not w.isdigit() and w not in GENERIC_ADDR and len(w) > 1:
                    keys.add(t + "_" + w)
                    break
    state, city = "", ""
    for seg in segs:
        if len(seg) == 1 and seg[0] in STATE_CODES:
            state = seg[0]
        elif seg and not any(t.isdigit() for t in seg):
            city = " ".join(t for t in seg if t not in GENERIC_ADDR or len(seg) == 1)
    return (" ".join(core), " ".join(landmark), postal, " ".join(nums), house,
            " ".join(sorted(keys)), state, city)


def city_tokens(addr_core):
    return {t for t in addr_core.split() if not t.isdigit() and t not in GENERIC_ADDR}


def prepare_columns(names, addrs, tdict=None):
    """Vectorised-ish over python lists; returns dict of columns."""
    nn, nc, nj, web, na = zip(*(norm_name(x, tdict) for x in names)) if len(names) else ((),) * 5
    ac, lm, po, nu, ho, sk, st, ci = (zip(*(norm_address(x, tdict) for x in addrs)) if len(addrs)
                                      else ((),) * 8)
    return {"name_norm": list(nn), "name_core": list(nc), "name_join": list(nj),
            "name_alt": list(na), "is_web": np.asarray(web, np.int8), "addr_core": list(ac),
            "landmark": list(lm), "postal": list(po), "nums": list(nu), "house_no": list(ho),
            "street_keys": list(sk), "state": list(st), "city": list(ci),
            "name_native": np.asarray([bool(_INDIC.search(x)) for x in names], np.int8),
            "addr_empty": np.asarray([len(a) == 0 for a in ac], np.int8)}
