#!/bin/bash

target/debug/clif-util compile --target arm64 arm64_tests/fib.clif > fib.s

cat > fib_final.s <<EOF
.global fib
fib:
EOF

cat fib.s > fib_final.s

cat > fib.c <<EOF
#include <stdint.h>
#include <stdio.h>

int64_t fib(int64_t);
int main() {
  int64_t n = 20;
  int64_t fib_n = fib(n);
  printf("fib(%ld) = %ld\n", n, fib_n);
}
EOF

gcc -O2 fib.c fib_final.s -o fib
