#!/bin/bash
# Screenshot one lab screen and check its DOM for crash messages.
# usage: tools/shoot.sh <lab-id> <hash> <out.png> [height]     e.g. tools/shoot.sh optimization-2 batch-3 /tmp/b3.png
# <hash> is <lessonId> or <lessonId>-<screen 1..4>. Pass "" as lab-id to shoot the gallery.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB="$1"; HASH="$2"; OUT="$3"; HT="${4:-2000}"
URL="file://$ROOT/docs/${LAB:+$LAB/}index.html${HASH:+#$HASH}"
P=$(mktemp -d)
google-chrome --headless=new --disable-gpu --no-first-run --user-data-dir="$P" --window-size=1440,$HT --virtual-time-budget=6000 --screenshot="$OUT" "$URL" >/dev/null 2>&1
P2=$(mktemp -d)
google-chrome --headless=new --disable-gpu --no-first-run --user-data-dir="$P2" --window-size=1440,$HT --virtual-time-budget=6000 --dump-dom "$URL" 2>/dev/null > "$OUT.dom.html"
if grep -q 'class="err"' "$OUT.dom.html"; then echo "ERROR in $LAB#$HASH:"; sed -n 's/.*id="\(fatal-error\|lesson-error\)"[^>]*>\([^<]*\).*/\2/p' "$OUT.dom.html" | head -c 2000; else echo "ok $LAB#$HASH ($(wc -c < "$OUT.dom.html") bytes dom)"; fi
rm -rf "$P" "$P2"
