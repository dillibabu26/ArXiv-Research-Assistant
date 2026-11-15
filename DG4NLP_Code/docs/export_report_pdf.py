"""
Export REPORT.md to REPORT.pdf in the format the brief asks for: cover page, numbered
sections, minimum 10 pt font, embedded figures and tables.

    pip install markdown weasyprint      (one-off; see note below for Windows)
    python docs/export_report_pdf.py

Windows note: WeasyPrint needs the GTK runtime. If installing it is a problem, open
REPORT.md in VS Code with the "Markdown PDF" extension (yzane.markdown-pdf) and use
"Markdown PDF: Export (pdf)" instead; the cover page and figures export the same way.
"""

from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]

CSS = """
@page { size: A4; margin: 2.2cm 2cm; @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
@page:first { @bottom-center { content: none; } }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; line-height: 1.25; margin: 0 0 18pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; }
p { margin: 0 0 8pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; margin: 6pt 0 12pt; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0f0ee; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; }
em { color: #333; }
code { font-size: 10pt; font-family: "DejaVu Sans Mono", monospace; }
blockquote { display: none; }   /* the draft-status note is not exported */
.cover { page-break-after: always; text-align: center; padding-top: 5cm; }
.cover h1 { font-size: 22pt; }
.cover p { text-align: center; font-size: 12pt; margin: 6pt 0; }
"""


def main() -> None:
    md_text = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "md_in_html"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    out = ROOT / "REPORT.pdf"
    HTML(string=html, base_url=str(ROOT)).write_pdf(out)
    print("saved", out)


if __name__ == "__main__":
    main()
