import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { load } from "../dist/index.js";

const path = process.argv[2];
if (!path) throw new Error("usage: npm test -- path/to/smoke.wasm");

const runtime = await load(await readFile(path));
try {
  assert.equal(runtime.sources.length, 2);
  assert.equal(runtime.outputs.length, 1);
  const [output] = runtime.run({
    input: new Float32Array([1, 2, 3, 4]),
    bias: new Float32Array([-2, 2, 3, 4]),
  });
  assert.deepEqual(Array.from(output.data), [0, 4, 6, 8]);
} finally {
  runtime.close();
}
