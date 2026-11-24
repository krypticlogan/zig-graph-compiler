# JavaScript and TypeScript WASM runtime

This package loads a `wasm32-freestanding` ZGC executable. It contains no
compiler; build the graph first:

```sh
zgc compile graph.zgir --target wasm32-freestanding -o graph.wasm
cd bindings/js && npm install && npm run build
```

The TypeScript source builds to ESM JavaScript and `.d.ts` declarations. Both
JavaScript and TypeScript callers use the same package:

```ts
import { load } from "zgc-runtime-wasm";

const runtime = await load(await (await fetch("/graph.wasm")).arrayBuffer());
try {
  const [output] = runtime.run({ input: new Float32Array([1, 2, 3, 4]) });
  console.log(output.info.shape, output.data);
} finally {
  runtime.close();
}
```

`run` accepts named typed-array sources and returns copied outputs in logical
row-major order. `execute()` runs with existing source values; `output(i)`
copies an individual output. `sources` and `outputs` expose the compiled
names, shapes, dtypes, bindings, and strides. `f16` arrays use `Uint16Array`
bit patterns; boolean arrays use `Uint8Array`. For bound sources, `setSource`
packs logical values into retained WASM storage before binding them. `close()`
releases the runtime's allocations.

To run the smoke check after building the package:

```sh
zgc compile ../tests/fixtures/smoke.zgir --target wasm32-freestanding -o smoke.wasm
npm test -- smoke.wasm
```
