#!/bin/bash

set -e

cargo build
(cd cranelift-codegen && cargo test)

for f in filetests/vcode/arm64/*.clif; do
  echo $f
  target/debug/clif-util test $f
done
