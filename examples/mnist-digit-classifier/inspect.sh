mkdir -p inspect-out

zig build inspect -- all \
  > inspect-out/model.txt 2>&1

zig build disassemble-model -Doptimize=ReleaseFast \
  > inspect-out/disassembly.txt 2>&1

nm -nm zig-out/bin/zgc-executable \
  > inspect-out/symbols.txt

otool -l zig-out/bin/zgc-executable \
  > inspect-out/load-commands.txt
