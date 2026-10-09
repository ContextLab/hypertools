import json
import os
import sys
from playwright.sync_api import sync_playwright

# usage: python check_built_site.py HTML_DIR BASE_URL OUT_DIR   (serve HTML_DIR first: python -m http.server)
ROOT = sys.argv[1]
BASE = sys.argv[2].rstrip("/") + "/"
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
pages = sorted(
    os.path.relpath(os.path.join(d, f), ROOT)
    for d, _, fs in os.walk(ROOT)
    for f in fs
    if f.endswith(".html") and not d.startswith(os.path.join(ROOT, "_"))
)
shots = [
    p
    for p in pages
    if (
        ("/" not in p and not p.startswith("hypertools."))
        or p.startswith("tutorials/")
        or p == "auto_examples/index.html"
    )
]
shots += [
    "hypertools.plot.html",
    "hypertools.predict.html",
    "hypertools.describe.html",
    "hypertools.io.LSLStream.html",
    "hypertools.HyperAnimation.html",
]
JS = """() => {
  const out={};
  out.broken=[...document.images].filter(i=>i.complete&&i.naturalWidth===0).map(i=>i.getAttribute('src')).slice(0,5);
  out.overflow=document.documentElement.scrollWidth-document.documentElement.clientWidth;
  const pseudo=[]; for(const el of document.querySelectorAll('article *')){ for(const ps of ['::before','::after']){ const c=getComputedStyle(el,ps).content; if(c&&/ERROR|WARNING|unnecessary/i.test(c)) pseudo.push(el.tagName+'.'+el.className+': '+c.slice(0,80)); } }
  out.pseudo=pseudo.slice(0,3);
  out.sysmsg=document.querySelectorAll('.system-message, .problematic').length;
  out.videos=[...document.querySelectorAll('video')].length;
  const txt=[...document.querySelectorAll('article p, article li, article dd, article dt, article h1, article h2, article h3')].map(e=>{const c=e.cloneNode(true); c.querySelectorAll('code,pre,.math,script,style').forEach(x=>x.remove()); return c.textContent;}).join('\\n');
  out.leaks=(txt.match(/[^\\n]{0,30}(``|:[a-z]+:`|\\.\\. [a-z-]+::)[^\\n]{0,30}/g)||[]).slice(0,3);
  return out; }"""
res = {}
with sync_playwright() as pw:
    b = pw.chromium.launch()
    for theme in ["light"]:
        ctx = b.new_context(viewport={"width": 1400, "height": 1800})
        pg = ctx.new_page()
        errs = []
        pg.on(
            "console",
            lambda m: errs.append(m.text[:160]) if m.type == "error" else None,
        )
        pg.on("pageerror", lambda e: errs.append("PAGEERROR " + str(e)[:160]))
        for p in pages:
            errs.clear()
            try:
                pg.goto(BASE + p, wait_until="load", timeout=60000)
                pg.wait_for_timeout(400)
            except Exception as e:
                res[p] = {"goto": str(e)[:100]}
                continue
            r = pg.evaluate(JS)
            r["console"] = sorted(set(errs))[:4]
            res[p] = r
            if p in shots:
                pg.screenshot(path=os.path.join(OUT, p.replace("/", "__") + ".png"))
    # phone width overflow on every page
    ctx = b.new_context(viewport={"width": 400, "height": 800})
    pg = ctx.new_page()
    for p in pages:
        try:
            pg.goto(BASE + p, wait_until="load", timeout=60000)
            pg.wait_for_timeout(250)
            res[p]["phone_overflow"] = pg.evaluate(
                "()=>document.documentElement.scrollWidth-document.documentElement.clientWidth"
            )
        except Exception as e:
            res[p]["phone"] = str(e)[:80]
    b.close()
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
bad = {
    p: r
    for p, r in res.items()
    if r.get("broken")
    or r.get("overflow", 0) > 2
    or r.get("pseudo")
    or r.get("sysmsg")
    or r.get("leaks")
    or r.get("console")
    or r.get("phone_overflow", 0) > 2
    or "goto" in r
}
print(len(res), "pages checked;", len(bad), "with findings")
for p, r in bad.items():
    print(p, {k: v for k, v in r.items() if v and k not in ("videos",)})
