"""Print what GROBID extracted from one TEI file: header, sections, pages, figures, references.

Usage: uv run python experiments/stage1/inspect_tei.py data/parsed/<arxiv_id>.tei.xml
"""

import re, sys, xml.etree.ElementTree as ET
NS = {"t": "http://www.tei-c.org/ns/1.0"}
root = ET.parse(sys.argv[1]).getroot()
def txt(e): return " ".join(" ".join(e.itertext()).split()) if e is not None else None
hdr = root.find(".//t:teiHeader", NS)
print("TITLE:", txt(hdr.find(".//t:titleStmt/t:title", NS)))
authors = [txt(p) for p in hdr.findall(".//t:sourceDesc//t:analytic/t:author/t:persName", NS)]
print("AUTHORS:", len(authors), authors[:5])
abst = txt(hdr.find(".//t:profileDesc/t:abstract", NS)); print("ABSTRACT:", len(abst.split()), "words |", abst[:160], "...")
body = root.find(".//t:text/t:body", NS)
divs = body.findall("t:div", NS)
pages = set()
for el in root.iter():
    c = el.get("coords")
    if c:
        for box in c.split(";"):
            pages.add(int(box.split(",")[0]))
print("PAGES with coords:", min(pages), "-", max(pages))
print(f"BODY: {len(divs)} sections, {len(body.findall('.//t:p', NS))} paragraphs, {sum(len(txt(p).split()) for p in body.findall('.//t:p', NS))} words")
for d in divs:
    h = d.find("t:head", NS)
    ps = d.findall("t:p", NS)
    pg = sorted({int(b.split(',')[0]) for p in ps if p.get('coords') for b in p.get('coords').split(';')})
    print(f"  [{(h.get('n') if h is not None else '') or '-':>5}] {txt(h)[:55] if h is not None else '(no head)':55} paras={len(ps):2} pages={pg[:1]}..{pg[-1:]}")
figs = root.findall(".//t:figure", NS)
print("FIGURES/TABLES:", sum(1 for f in figs if f.get('type')!='table'), "figures,", sum(1 for f in figs if f.get('type')=='table'), "tables")
for f in figs[:3]: print("   ", f.get("type") or "figure", "|", (txt(f.find("t:head", NS)) or ""), "|", (txt(f.find("t:figDesc", NS)) or "")[:90])
refs = root.findall(".//t:back//t:listBibl/t:biblStruct", NS); print("REFERENCES:", len(refs))
if refs: print("   e.g.", txt(refs[0].find(".//t:title", NS)))
back = root.find(".//t:text/t:back", NS)
print("BACK divs (appendix etc.):", [d.get("type") for d in back.findall("t:div", NS)])
sample = body.findall(".//t:p", NS)[3]
print("SAMPLE PARAGRAPH (4th):", txt(sample)[:400])
print("  coords:", sample.get("coords")[:80])
print("FORMULAS:", len(root.findall(".//t:formula", NS)), "| e.g.", txt(root.find(".//t:formula", NS))[:100])
