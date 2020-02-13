ARM64 and new-x86 porting effort:

- New code is in
  - cranelift-codegen/src/machinst/ (target-independent backend infra)
  - cranelift-codegen/src/isa/arm64/ (ARM64 target-specific)
  - cranelift-codegen/src/isa/x64/ (x86 target-specific)

- Invocations to try:

  - cd cranelift-codegen && cargo test
    - this runs unit tests, including encoding checks against golden bytes.

  - target/debug/clif-util compile --target arm64 \
                           -d -D -p filetests/vcode/arm64/file.clif
    - this compiles a test input (in CraneLift IR) and prints:
      - ARM64 assembly, which should be parseable by GNU as (-D flag)
      - Machine code, in 32-bit words (-p flag)
      - if RUST_LOG=debug is set, debug spew (-d flag)

  - the same, but with --target x86_64
    - does not do anything yet

  - target/debug/clif-util test filetests/vcode/arm64/file.clif
    - this runs the "filecheck" utility, which performs specified checks
      against the ARM64-assembly (really vcode) output of compilation
      - see https://docs.rs/filecheck/0.4.0/filecheck/ for directives
