"""Construit le mapping MeSH term -> categories A/C/H (Weber 2013).

Regle de Weber (2013), adaptee au MeSH 2026 :
- H : sous-arbre Human (B01.050.150.900.649.313.988.400.112.400.400)
      + M01 (Named Groups) + DescriptorName "Humans"
- A : sous-arbre B01 (Eukaryota) SAUF le sous-arbre Human
- C : sous-arbres A11, B02, B03, B04, G02.111.570, G02.149
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import requests
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
LOCAL_XML = ROOT / "data" / "mesh" / "desc2026.xml"
OUTPUT = ROOT / "data" / "mesh" / "mesh_map.pkl"

# Prefixes MeSH 2026
H_PREFIXES = [
    "B01.050.150.900.649.313.988.400.112.400.400",  # Humans (2026)
    "M01",                                            # Named Groups
]
A_PREFIXES = ["B01"]
C_PREFIXES = [
    "A11",
    "B02",
    "B03",
    "B04",
    "G02.111.570",
    "G02.149",
]

# Fallback par nom pour les check tags standard
H_NAMES = {"humans"}


def classify(descriptor_name: str, tree_numbers: list[str]) -> set[str]:
    cats: set[str] = set()

    # H par nom (fallback robuste)
    if descriptor_name.lower() in H_NAMES:
        cats.add("H")

    for tn in tree_numbers:
        # H d'abord (le plus specifique)
        if any(tn == p or tn.startswith(p + ".") for p in H_PREFIXES):
            cats.add("H")
        elif any(tn == p or tn.startswith(p + ".") for p in A_PREFIXES):
            cats.add("A")
        # C independant
        if any(tn == p or tn.startswith(p + ".") for p in C_PREFIXES):
            cats.add("C")
    return cats


def download() -> Path:
    LOCAL_XML.parent.mkdir(parents=True, exist_ok=True)

    if LOCAL_XML.exists() and LOCAL_XML.stat().st_size > 100_000_000:
        size_mb = LOCAL_XML.stat().st_size // 1024 // 1024
        print(f"XML deja present : {LOCAL_XML} ({size_mb} MB)")
        return LOCAL_XML

    if LOCAL_XML.exists():
        LOCAL_XML.unlink()

    headers = {"User-Agent": "Mozilla/5.0 (compatible; Precite/1.0)"}
    urls = [
        "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml",
        "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml",
    ]

    for url in urls:
        print(f"Tentative : {url}")
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=600, allow_redirects=True)
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            print(f"  Status : {r.status_code}, taille attendue : {total // 1024 // 1024} MB")

            received = 0
            with open(LOCAL_XML, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        received += len(chunk)
                        if received % (50 * 1024 * 1024) < (1 << 20):
                            print(f"  {received // 1024 // 1024} MB / {total // 1024 // 1024} MB")

            if received < 10_000_000:
                print(f"  ECHEC : seulement {received} octets recus")
                LOCAL_XML.unlink(missing_ok=True)
                continue

            size_mb = received // 1024 // 1024
            print(f"  OK : {LOCAL_XML} ({size_mb} MB)")
            return LOCAL_XML

        except Exception as e:
            print(f"  Echec : {e}")
            if LOCAL_XML.exists():
                LOCAL_XML.unlink()

    raise RuntimeError("Impossible de telecharger le fichier MeSH")


def parse(xml_path: Path) -> dict[str, set[str]]:
    print("Parsing MeSH tree...")
    mapping: dict[str, set[str]] = {}
    n = 0

    context = etree.iterparse(
        str(xml_path), events=("end",), tag="DescriptorRecord"
    )
    for _, elem in context:
        name_node = elem.find("DescriptorName/String")
        if name_node is None or not name_node.text:
            elem.clear()
            continue

        name = name_node.text.strip().lower()
        tree_numbers = [
            tn.text.strip()
            for tn in elem.findall("TreeNumberList/TreeNumber")
            if tn.text
        ]

        cats = classify(name, tree_numbers)
        if cats:
            mapping[name] = cats

        n += 1
        if n % 5000 == 0:
            print(f"  {n} descriptors, {len(mapping)} classes")
        elem.clear()

    print(f"Total : {n} descriptors, {len(mapping)} avec categories")
    return mapping


def main() -> int:
    xml = download()
    if xml.stat().st_size < 1_000_000:
        raise RuntimeError(f"Fichier XML vide : {xml.stat().st_size} octets")

    mapping = parse(xml)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(mapping, f)
    print(f"Ecrit : {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")

    samples = [
        "humans", "mice", "animals", "cells", "proteins",
        "bacteria", "viruses", "molecular structure",
        "enzymes", "rats", "dogs",
    ]
    print("\nVerification sur des exemples :")
    for s in samples:
        cats = mapping.get(s, [])
        marker = "OK " if cats else "   "
        print(f"  {marker}{s:25s} -> {sorted(cats)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())