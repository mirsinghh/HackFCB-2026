"""Builds docs/HackFCB.pdf from docs/report.md (needs the `markdown` package and Microsoft Edge or Chrome).

    python docs/make_figures.py && python docs/build_report.py
"""
import os
import re
import shutil
import subprocess
import sys
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1f2430; }
h1 { font-size: 22pt; margin: 0 0 2pt; color: #12306b; }
h2 { font-size: 14pt; margin: 18pt 0 4pt; color: #12306b; border-bottom: 1px solid #d9dce1; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 3pt; color: #1f2430; }
p { margin: 5pt 0; } em { color: #566070; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #d9dce1; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef1f6; }
code { background: #f1f3f6; padding: 0 2pt; border-radius: 2pt; font-size: 9.5pt; }
pre { background: #f1f3f6; padding: 6pt 8pt; border-radius: 4pt; font-size: 9pt; }
pre code { background: none; }
img { width: 100%; margin: 8pt 0; page-break-inside: avoid; }
.fig { margin: 8pt 0; page-break-inside: avoid; } .fig svg { width: 100%; height: auto; } .cap { font-size: 9pt; color: #566070; margin: 2pt 0 0; }
li { margin: 2pt 0; }
"""


def find_browser():
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              shutil.which("google-chrome") or "", shutil.which("chromium") or ""):
        if p and os.path.isfile(p):
            return p
    sys.exit("no Edge/Chrome found")


def main():
    with open(os.path.join(HERE, "report.md"), encoding="utf-8") as fh:
        body = markdown.markdown(fh.read(), extensions=["tables", "fenced_code"])
    def inline(m):  # embed the SVG so the HTML/PDF does not depend on relative paths
        with open(os.path.join(HERE, m.group(2)), encoding="utf-8") as svg:
            return "<div class='fig'>%s<p class='cap'>%s</p></div>" % (svg.read(), m.group(1))
    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+[.]svg)" /></p>', inline, body)
    html = "<!doctype html><html><head><meta charset='utf-8'><title>HackFCB</title><style>%s</style></head><body>%s</body></html>" % (CSS, body)
    html_path = os.path.join(HERE, "HackFCB.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    pdf = os.path.join(HERE, "HackFCB.pdf")
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--user-data-dir=" + os.path.join(HERE, ".browser-profile"),
                    "--print-to-pdf=" + pdf, "file:///" + html_path.replace("\\", "/")], check=True, timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", pdf, os.path.getsize(pdf), "bytes")


if __name__ == "__main__":
    main()
