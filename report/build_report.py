"""
Converts report.md -> report.pdf via headless Chrome/Edge's print-to-pdf.

(xhtml2pdf was tried first but has a confirmed image-embedding bug in this
environment - the second <img> on a page silently reuses the first image's
bytes, reproducible even in a two-line test HTML file with no markdown or
custom code involved. Headless Chromium renders HTML/CSS correctly and is
already installed on essentially every Windows machine, so it's the more
reliable choice here.)

    pip install markdown
    python report/build_report.py
"""

import shutil
import subprocess
from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).resolve().parent
MD_PATH = REPORT_DIR / "report.md"
PDF_PATH = REPORT_DIR / "report.pdf"
HTML_PATH = REPORT_DIR / "_report_render.html"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 1.7cm 1.8cm 1.8cm 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 21pt; color: #1f3b57; margin-top: 0; }
h2 { font-size: 15pt; color: #1f3b57; border-bottom: 1.5pt solid #1f3b57; padding-bottom: 3px; margin-top: 18px; page-break-after: avoid; }
h3 { font-size: 12.5pt; color: #2b4f70; margin-top: 16px; }
code { font-family: Courier, monospace; background-color: #f0f0f0; padding: 1px 3px; font-size: 9.5pt; }
pre { font-family: Courier, monospace; background-color: #f4f4f4; padding: 8px; font-size: 9pt;
      border: 0.5pt solid #ccc; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 0.5pt solid #999; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background-color: #1f3b57; color: white; }
th code { background: transparent; color: inherit; }
img { max-width: 100%; }
a { color: #1f5fa0; }
hr { border: none; border-top: 0.5pt solid #ccc; }
"""


def find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise RuntimeError("No Chrome/Edge executable found - install one or add it to PATH")


def build():
    md_text = MD_PATH.read_text(encoding="utf-8")
    body_html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])

    full_html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>CardioRisk-MLOps Report</title><style>{CSS}</style></head>
<body>
{body_html}
</body>
</html>
"""
    HTML_PATH.write_text(full_html, encoding="utf-8")

    browser = find_browser()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={PDF_PATH}",
        "--no-pdf-header-footer",
        HTML_PATH.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not PDF_PATH.exists():
        raise RuntimeError(f"PDF generation failed: {result.stdout}\n{result.stderr}")

    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
