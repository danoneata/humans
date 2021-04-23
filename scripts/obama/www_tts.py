import os

import dominate

from dominate import tags


OUT_DIR = "output/www/tts-obama"

doc = dominate.document(title="Text-to-speech for Obama")

with doc.head:
    tags.meta(**{"content": "text/html;charset=utf-8", "http-equiv": "Content-Type"})
    tags.meta(**{"content": "utf-8", "http-equiv": "encoding"})

    # jQuery
    tags.script(
        type="text/javascript", src="https://code.jquery.com/jquery-3.5.1.min.js",
    )
    # Bootstrap
    tags.link(
        rel="stylesheet",
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
    )
    tags.script(
        type="text/javascript",
        src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
        integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo",
        crossorigin="anonymous",
    )
    tags.script(
        type="text/javascript",
        src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
    )


with doc:
    with tags.body():
        with tags.div(cls="container"):
            tags.h1("Text-to-speech for Obama", cls="mt-5")

            path = "data/out_synth_{:03d}.wav"
            for i in range(11):
                with tags.audio(controls=True):
                    tags.source(src=path.format(i), type="audio/wav")

with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
    f.write(str(doc))
