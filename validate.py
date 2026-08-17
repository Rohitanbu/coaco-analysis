from html.parser import HTMLParser
import sys

class Validator(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.void_elements = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def handle_starttag(self, tag, attrs):
        if tag not in self.void_elements:
            self.tags.append(tag)

    def handle_endtag(self, tag):
        if tag in self.void_elements:
            return
        if not self.tags:
            print(f"Error: unmatched end tag {tag}")
            sys.exit(1)
        expected = self.tags.pop()
        if expected != tag:
            print(f"Error: unmatched end tag {tag}, expected {expected}")
            sys.exit(1)

parser = Validator()
with open('analyser.html') as f:
    parser.feed(f.read())
if parser.tags:
    print(f"Error: unclosed tags: {parser.tags}")
else:
    print("HTML is valid")
