#!/usr/bin/env python3

import sys
import os
import tempfile

words = []
found_code = False

for line in sys.stdin.readlines():
    if line.startswith("Machine code:"):
        found_code = True
        continue
    if found_code:
        words.append(int("0x" + line.strip(), 16))

fd, filename = tempfile.mkstemp(suffix = ".bin")
f = os.fdopen(fd, "wb")
for word in words:
    f.write(bytes([word & 0xff, (word >> 8) & 0xff, (word >> 16) & 0xff, (word >> 24) & 0xff]))
f.close()

os.system("aarch64-linux-gnu-objdump -b binary -m aarch64 -EL -D %s" % filename)
os.unlink(filename)
