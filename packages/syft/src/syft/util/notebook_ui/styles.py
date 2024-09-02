# relative
from ..assets import load_css

FONT_CSS = load_css("fonts.css")
STYLESHEET_URLS = [
    "https://fonts.googleapis.com/css2?family=Karla:ital,wght@0,200;0,300;0,400;0,500;0,600;0,700;0,800;1,200;1,300;1,400;1,500;1,600;1,700;1,800&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap",
    "https://fonts.cdnfonts.com/css/dejavu-sans-mono",
]
STYLESHEET_JS_CALLS = "\n".join([f'addStyleSheet("{s}")' for s in STYLESHEET_URLS])

JS_DOWNLOAD_FONTS = f"""
<script>
function addStyleSheet(fileName) {{
  var head = document.head;
  var link = document.createElement("link");

  link.type = "text/css";
  link.rel = "stylesheet";
  link.href = fileName;

  head.appendChild(link);
}}

{STYLESHEET_JS_CALLS}
</script>
"""

CSS_CODE = f"""
<style>
  {load_css("style.css")}
</style>
"""
