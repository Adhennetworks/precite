"""Parsing XML PubMed en streaming avec GZIP a la volee.

Capture le Status de MedlineCitation pour identifier les articles
non indexes par NLM. Ces articles ont des metadonnees incompletes
(pas de references) et sont exclus de l'entrainement MEDLINE.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from typing import Any

from lxml import etree

from precite.utils.logging import get_logger

log = get_logger(__name__)


_REF_PATHS = (
    "MedlineCitation/Article/ReferenceList/Reference",
    "MedlineCitation/ReferenceList/Reference",
    "PubmedData/ReferenceList/Reference",
)

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


class ChunkReader:
    def __init__(self, chunks: Iterator[bytes]):
        self._chunks = chunks
        self._buf = b""
        self._exhausted = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            if not self._exhausted:
                self._buf += b"".join(self._chunks)
                self._exhausted = True
            out, self._buf = self._buf, b""
            return out

        while len(self._buf) < size and not self._exhausted:
            try:
                self._buf += next(self._chunks)
            except StopIteration:
                self._exhausted = True

        out, self._buf = self._buf[:size], self._buf[size:]
        return out


def _text(node: etree._Element | None) -> str | None:
    if node is None:
        return None
    value = (node.text or "").strip()
    return value or None


def _texts(nodes: list[etree._Element]) -> list[str]:
    return [t for n in nodes if (t := _text(n))]


def _find_references(elem: etree._Element) -> list[etree._Element]:
    for path in _REF_PATHS:
        refs = elem.findall(path)
        if refs:
            return refs
    return elem.findall(".//Reference")


def _extract_ref_year(ref: etree._Element) -> int | None:
    y = _text(ref.find(".//PubDate/Year"))
    if y and y.isdigit() and 1800 <= int(y) <= 2100:
        return int(y)

    md = _text(ref.find(".//PubDate/MedlineDate"))
    if md:
        first = md.split()[0] if md.split() else None
        if first and first.isdigit():
            year = int(first)
            if 1800 <= year <= 2100:
                return year

    citation = _text(ref.find("Citation"))
    if citation:
        for m in _YEAR_RE.finditer(citation):
            year = int(m.group(1))
            if 1800 <= year <= 2100:
                return year

    return None


def _extract_ref_pmid(ref: etree._Element) -> str | None:
    for aid in ref.findall("ArticleIdList/ArticleId"):
        if aid.get("IdType") == "pubmed":
            return _text(aid)
    return None

def _count_pages(pages_raw: str | None) -> int | None:
    """Convertit une pagination XML en nombre de pages.

    Exemples :
        "16-28"      -> 13
        "e12345"     -> 1
        "16-28, 32"  -> 14
        "S1-5"       -> 5
        "1"          -> 1
    """
    if not pages_raw:
        return None

    import re
    total = 0

    # Cherche tous les patterns "A-B" ou "A"
    for match in re.finditer(r"([A-Za-z]*)(\d+)(?:\s*-\s*([A-Za-z]*)(\d+))?", pages_raw):
        _prefix, start, _, end = match.groups()
        start_n = int(start)
        if end:
            end_n = int(end)
            n = end_n - start_n + 1
            if n > 0 and n < 500:
                total += n
            else:
                total += 1
        else:
            total += 1

    return total if total > 0 else None

def iter_articles(
    chunks: Iterator[bytes],
    fields_cfg: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    item_tag = fields_cfg["item_tag"]
    fields = fields_cfg["fields"]

    ref_related_keys = {"references", "ref_year", "ref_medline_date", "ref_pmid"}

    reader = ChunkReader(chunks)
    with gzip.GzipFile(fileobj=reader) as gz:
        context = etree.iterparse(
            gz,
            events=("end",),
            tag=item_tag,
            recover=True,
            huge_tree=False,
        )

        for _event, elem in context:
            record: dict[str, Any] = {}

            # Capture du Status MedlineCitation
            mc = elem.find("MedlineCitation")
            if mc is not None:
                record["medline_status"] = mc.get("Status")
            else:
                record["medline_status"] = None
            # Extraction des pages
            record["pages_raw"] = None
            pgn = elem.find(".//MedlinePgn")
            if pgn is not None and pgn.text:
                record["pages_raw"] = pgn.text.strip()

            for name, xpath in fields.items():
                if name in ref_related_keys:
                    continue
                nodes = elem.findall(xpath)
                if not nodes:
                    record[name] = None
                elif len(nodes) == 1:
                    record[name] = _text(nodes[0])
                else:
                    record[name] = _texts(nodes)

            refs = _find_references(elem)
            ref_years: list[int] = []
            ref_pmids: list[str] = []
            for ref in refs:
                y = _extract_ref_year(ref)
                if y is not None:
                    ref_years.append(y)
                pmid = _extract_ref_pmid(ref)
                if pmid:
                    ref_pmids.append(pmid)

            record["n_references"] = len(refs)
            record["ref_years"] = ref_years
            record["ref_pmids"] = ref_pmids
            record["n_pages"] = _count_pages(record.get("pages_raw"))

            yield record

            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]