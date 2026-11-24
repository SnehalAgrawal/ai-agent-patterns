"""
What a Markdown renderer does with probe P7's reply and probe P8's encoded variant (Article #39).

security_probes.py shows that a check on the raw reply text finds P7's URL but misses one written
as `https:&#47;&#47;...`. This script shows what a CommonMark renderer makes of the same text: it
decodes the character references, so the browser would still request the URL. It is optional and
needs one package that the rest of this demo doesn't:

  pip install markdown-it-py==3.0.0
  python security_render_check.py
"""
import sys

import markdown_it
from markdown_it import MarkdownIt

from security_probes import ENCODED_REPLY, IMAGE_REPLY

print(f"markdown-it-py {markdown_it.__version__}, Python {sys.version.split()[0]}, preset commonmark")
print("P7 reply           ->", MarkdownIt("commonmark").render(IMAGE_REPLY).strip().splitlines()[-1])
print("encoded reply      ->", MarkdownIt("commonmark").render(ENCODED_REPLY).strip())
print("images disabled    ->", MarkdownIt("commonmark").disable("image").render(ENCODED_REPLY).strip())
