"""Write the manuscript bibliography and references.bib from references_data.py.

The thebibliography block of the template is replaced by entries in order of
first citation (as PLOS requires); every cited key must exist and every entry
must be cited. references.bib holds the same records for the journal.

Usage: python make_bibliography.py <template.tex> <references.bib>
"""
import re
import sys
from pathlib import Path

from references_data import R


def vancouver(r):
    a = (r["authors"] + " ") if r.get("authors") else ""
    t = r["title"] + ("" if r["title"].endswith(("?", ".")) else ".")
    doi = " doi:" + r["doi"].replace("_", "\\_") if r.get("doi") else ""
    if r["type"] == "article":
        s = f"{a}{t} {r['journal']}. {r['year']};{r['volume']}:{r['pages']}.{doi}"
    elif r["type"] == "inproceedings":
        pages = f" p. {r['pages']}." if r.get("pages") else ""
        s = f"{a}{t} In: {r['booktitle']}. {r['publisher']}; {r['year']}.{pages}{doi}"
    elif r["type"] == "book":
        s = f"{a}{t} {r['publisher']}; {r['year']}.{doi}"
    else:
        where = f" Available from: \\url{{{r['url']}}}." if r.get("url") else ""
        s = f"{a}{t} {r['howpublished']}; {r['year']}.{doi}{where}"
    return s.strip()


def bib_authors(v):
    v = v.rstrip(".")
    parts = [p.strip() for p in v.split(",")]
    out = []
    for p in parts:
        if p == "et al":
            out.append("others"); continue
        w = p.split(" ")
        out.append(f"{{{' '.join(w[:-1])}}}, {'. '.join(w[-1])}.")
    return " and ".join(out)


def bibtex(key, r):
    f = {"title": "{" + r["title"] + "}", "year": str(r["year"])}
    if r.get("authors"):
        f["author"] = bib_authors(r["authors"])
    for k_src, k_bib in (("journal", "journal"), ("volume", "volume"), ("pages", "pages"), ("booktitle", "booktitle"),
                         ("publisher", "publisher"), ("howpublished", "howpublished"), ("doi", "doi"), ("url", "url")):
        if r.get(k_src):
            f[k_bib] = r[k_src]
    body = ",\n".join(f"  {k} = {v}" if k == "title" else f"  {k} = {{{v}}}" for k, v in f.items())
    return f"@{r['type']}{{{key},\n{body}\n}}\n"


def main():
    tpl, bibfile = Path(sys.argv[1]), Path(sys.argv[2])
    s = tpl.read_text(encoding="utf-8")
    a, b = s.index("\\begin{thebibliography}"), s.index("\\end{thebibliography}")
    body = s[:a]
    order = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in R]
    unused = [k for k in R if k not in order]
    if missing:
        sys.exit(f"cited but not defined: {missing}")
    items = "\n".join(f"\\bibitem{{{k}}} {vancouver(R[k])}" for k in order)
    s = s[:a] + f"\\begin{{thebibliography}}{{{len(order)}}}\n" + items + "\n" + s[b:]
    tpl.write_text(s, encoding="utf-8", newline="\n")
    bibfile.write_text("".join(bibtex(k, R[k]) for k in order), encoding="utf-8", newline="\n")
    print(f"{len(order)} references in citation order; unused records: {unused}")


if __name__ == "__main__":
    main()
