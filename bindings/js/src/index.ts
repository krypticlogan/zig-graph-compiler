/** Browser and Node runtime for wasm32-freestanding ZGC executables (ABI v2). */

export type Dtype = "f32" | "f16" | "i8" | "bool" | "i32" | "u8";
export type TensorData = Float32Array | Uint16Array | Int8Array | Int32Array | Uint8Array;

export interface TensorInfo {
  readonly dtype: Dtype;
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly elementCount: number;
  readonly byteCount: number;
  readonly offset: number;
}

export interface SourceInfo {
  readonly key: number;
  readonly name: string;
  readonly kind: "input" | "parameter" | "constant" | "state";
  readonly binding: "owned" | "bound" | "embedded";
  readonly tensor: TensorInfo;
}

export interface OutputInfo extends TensorInfo {
  readonly binding: "owned" | "bound";
}

export interface TensorResult {
  readonly info: OutputInfo;
  /** A copy in logical row-major order, independent of the WASM memory. */
  readonly data: TensorData;
}

type WasmExports = {
  memory: WebAssembly.Memory;
  zgc_abi_version(): number;
  zgc_executable_structural_fingerprint(): bigint;
  zgc_runtime_size(): number;
  zgc_runtime_alignment(): number;
  zgc_runtime_mutable_bytes(): number;
  zgc_runtime_init(storage: number, size: number): number;
  zgc_runtime_deinit(storage: number): void;
  zgc_runtime_run(storage: number): number;
  zgc_executable_source_count(): number;
  zgc_executable_get_source_descriptor(index: number, out: number): number;
  zgc_runtime_copy_source(runtime: number, key: number, data: number, size: number): number;
  zgc_runtime_bind_source(runtime: number, key: number, data: number, size: number): number;
  zgc_executable_output_count(): number;
  zgc_executable_get_output_descriptor(index: number, out: number): number;
  zgc_executable_get_output_binding(index: number, out: number): number;
  zgc_runtime_bind_output(runtime: number, index: number, data: number, size: number): number;
  zgc_runtime_copy_output(runtime: number, index: number, out: number, size: number): number;
  zgc_wasm_alloc(size: number, alignment: number): number;
  zgc_wasm_free(pointer: number, size: number, alignment: number): void;
};

const ABI_VERSION = 2;
const TENSOR_DESCRIPTOR_SIZE = 28;
const SOURCE_DESCRIPTOR_SIZE = 44;
const DTYPE_NAMES: readonly Dtype[] = ["f32", "f16", "i8", "bool", "i32", "u8"];
const SOURCE_KINDS = ["input", "parameter", "constant", "state"] as const;
const SOURCE_BINDINGS = ["owned", "bound", "embedded"] as const;
const OUTPUT_BINDINGS = ["owned", "bound"] as const;
const STATUS_NAMES = [
  "ok", "invalid argument", "invalid runtime storage", "invalid source",
  "invalid output", "size mismatch", "alignment mismatch",
  "incompatible binding", "missing binding", "buffer too small", "out of memory",
] as const;

function requireOk(status: number, operation: string): void {
  if (status !== 0) throw new Error(`${operation}: ${STATUS_NAMES[status] ?? `status ${status}`}`);
}

function isExports(value: WebAssembly.Exports): value is WebAssembly.Exports & WasmExports {
  const candidate = value as Record<string, unknown>;
  return candidate.memory instanceof WebAssembly.Memory &&
    ["zgc_abi_version", "zgc_executable_structural_fingerprint", "zgc_runtime_size",
      "zgc_runtime_alignment", "zgc_runtime_mutable_bytes", "zgc_runtime_init",
      "zgc_runtime_deinit", "zgc_runtime_run", "zgc_executable_source_count",
      "zgc_executable_get_source_descriptor", "zgc_runtime_copy_source",
      "zgc_runtime_bind_source", "zgc_executable_output_count",
      "zgc_executable_get_output_descriptor", "zgc_executable_get_output_binding",
      "zgc_runtime_bind_output", "zgc_runtime_copy_output", "zgc_wasm_alloc",
      "zgc_wasm_free"].every((name) => typeof candidate[name] === "function");
}

function itemSize(dtype: Dtype): number {
  return dtype === "f32" || dtype === "i32" ? 4 : dtype === "f16" ? 2 : 1;
}

function dataView(exports: WasmExports): DataView {
  return new DataView(exports.memory.buffer);
}

function tensorInfo(view: DataView, pointer: number): TensorInfo {
  const dtype = DTYPE_NAMES[view.getUint32(pointer, true) - 1];
  if (!dtype) throw new Error("unsupported ZGC tensor dtype");
  const rank = view.getUint32(pointer + 4, true);
  const shapePointer = view.getUint32(pointer + 20, true);
  const stridesPointer = view.getUint32(pointer + 24, true);
  const shape: number[] = [];
  const strides: number[] = [];
  for (let axis = 0; axis < rank; axis++) {
    shape.push(view.getUint32(shapePointer + axis * 4, true));
    strides.push(view.getInt32(stridesPointer + axis * 4, true));
  }
  return {
    dtype, shape, strides,
    elementCount: view.getUint32(pointer + 8, true),
    byteCount: view.getUint32(pointer + 12, true),
    offset: view.getUint32(pointer + 16, true),
  };
}

function cString(view: DataView, pointer: number): string {
  let end = pointer;
  while (view.getUint8(end) !== 0) end++;
  return new TextDecoder().decode(new Uint8Array(view.buffer, pointer, end - pointer));
}

function bytesFor(info: TensorInfo, value: TensorData): Uint8Array {
  const matches = value instanceof Uint8Array ||
    (info.dtype === "f32" && value instanceof Float32Array) ||
    (info.dtype === "f16" && value instanceof Uint16Array) ||
    (info.dtype === "i8" && value instanceof Int8Array) ||
    (info.dtype === "i32" && value instanceof Int32Array);
  if (!matches || value.byteLength !== info.byteCount) {
    throw new TypeError(`expected ${info.dtype} data with ${info.byteCount} bytes`);
  }
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function copiedData(info: TensorInfo, bytes: Uint8Array): TensorData {
  const buffer = bytes.slice().buffer;
  switch (info.dtype) {
    case "f32": return new Float32Array(buffer);
    case "f16": return new Uint16Array(buffer); // IEEE f16 bit patterns
    case "i8": return new Int8Array(buffer);
    case "i32": return new Int32Array(buffer);
    case "bool":
    case "u8": return new Uint8Array(buffer);
  }
}

function physicalBytes(info: TensorInfo, logical: Uint8Array): Uint8Array {
  let stride = 1;
  let contiguous = info.offset === 0;
  for (let axis = info.shape.length - 1; axis >= 0; axis--) {
    contiguous &&= info.strides[axis] === stride;
    stride *= info.shape[axis];
  }
  if (contiguous) return logical;

  const result = new Uint8Array(info.byteCount);
  const itemBytes = itemSize(info.dtype);
  for (let linear = 0; linear < info.elementCount; linear++) {
    let remaining = linear;
    let physical = info.offset;
    for (let axis = info.shape.length - 1; axis >= 0; axis--) {
      const coordinate = remaining % info.shape[axis];
      remaining = Math.floor(remaining / info.shape[axis]);
      physical += coordinate * info.strides[axis];
    }
    if (physical < 0 || physical >= info.elementCount) {
      throw new Error("bound source layout exceeds its storage extent");
    }
    result.set(logical.subarray(linear * itemBytes, (linear + 1) * itemBytes), physical * itemBytes);
  }
  return result;
}

/** One instance of a compiled ZGC executable. Call close() when finished. */
export class Runtime {
  readonly structuralFingerprint: bigint;
  readonly runtimeSize: number;
  readonly runtimeAlignment: number;
  readonly mutableBytes: number;
  readonly sources: readonly SourceInfo[];
  readonly outputs: readonly OutputInfo[];

  private readonly exports: WasmExports;
  private readonly storage: number;
  private readonly allocations = new Map<number, { size: number; alignment: number }>();
  private readonly boundSources = new Map<number, number>();
  private readonly boundOutputs = new Map<number, number>();
  private closed = false;

  constructor(instance: WebAssembly.Instance) {
    if (!isExports(instance.exports)) throw new Error("module does not export the ZGC WASM runtime ABI");
    this.exports = instance.exports;
    const version = this.exports.zgc_abi_version();
    if (version !== ABI_VERSION) throw new Error(`ZGC ABI ${version} is incompatible with ${ABI_VERSION}`);
    this.structuralFingerprint = this.exports.zgc_executable_structural_fingerprint();
    this.runtimeSize = this.exports.zgc_runtime_size();
    this.runtimeAlignment = this.exports.zgc_runtime_alignment();
    this.mutableBytes = this.exports.zgc_runtime_mutable_bytes();
    this.storage = this.allocate(this.runtimeSize, this.runtimeAlignment);
    try {
      requireOk(this.exports.zgc_runtime_init(this.storage, this.runtimeSize), "zgc_runtime_init");
      this.sources = this.readSources();
      this.outputs = this.readOutputs();
      this.bindOutputs();
    } catch (error) {
      this.close();
      throw error;
    }
  }

  private allocate(size: number, alignment: number): number {
    if (!Number.isSafeInteger(size) || size < 1 || size > 0xffffffff ||
        !Number.isSafeInteger(alignment) || alignment < 1 || alignment > 0xffffffff) {
      throw new RangeError("invalid WASM allocation size or alignment");
    }
    const pointer = this.exports.zgc_wasm_alloc(size, alignment);
    if (!pointer) throw new Error(`WASM allocation failed for ${size} bytes`);
    this.allocations.set(pointer, { size, alignment });
    return pointer;
  }

  private readSources(): SourceInfo[] {
    const result: SourceInfo[] = [];
    const pointer = this.allocate(SOURCE_DESCRIPTOR_SIZE, 4);
    for (let index = 0; index < this.exports.zgc_executable_source_count(); index++) {
      requireOk(this.exports.zgc_executable_get_source_descriptor(index, pointer), "zgc_executable_get_source_descriptor");
      const view = dataView(this.exports);
      const kind = SOURCE_KINDS[view.getUint32(pointer + 8, true) - 1];
      const binding = SOURCE_BINDINGS[view.getUint32(pointer + 12, true) - 1];
      if (!kind || !binding) throw new Error("unsupported ZGC source descriptor");
      result.push({
        key: view.getUint32(pointer, true),
        name: cString(view, view.getUint32(pointer + 4, true)),
        kind, binding, tensor: tensorInfo(view, pointer + 16),
      });
    }
    this.release(pointer);
    return result;
  }

  private readOutputs(): OutputInfo[] {
    const result: OutputInfo[] = [];
    const pointer = this.allocate(TENSOR_DESCRIPTOR_SIZE, 4);
    const bindingPointer = this.allocate(4, 4);
    for (let index = 0; index < this.exports.zgc_executable_output_count(); index++) {
      requireOk(this.exports.zgc_executable_get_output_descriptor(index, pointer), "zgc_executable_get_output_descriptor");
      requireOk(this.exports.zgc_executable_get_output_binding(index, bindingPointer), "zgc_executable_get_output_binding");
      const view = dataView(this.exports);
      const binding = OUTPUT_BINDINGS[view.getUint32(bindingPointer, true) - 1];
      if (!binding) throw new Error("unsupported ZGC output binding");
      result.push({ ...tensorInfo(view, pointer), binding });
    }
    this.release(bindingPointer);
    this.release(pointer);
    return result;
  }

  private bindOutputs(): void {
    this.outputs.forEach((output, index) => {
      if (output.binding !== "bound") return;
      const pointer = this.allocate(output.byteCount, itemSize(output.dtype));
      requireOk(this.exports.zgc_runtime_bind_output(this.storage, index, pointer, output.byteCount), "zgc_runtime_bind_output");
      this.boundOutputs.set(index, pointer);
    });
  }

  private release(pointer: number): void {
    const allocation = this.allocations.get(pointer);
    if (!allocation) return;
    this.exports.zgc_wasm_free(pointer, allocation.size, allocation.alignment);
    this.allocations.delete(pointer);
  }

  private requireOpen(): void {
    if (this.closed) throw new Error("ZGC runtime is closed");
  }

  /** Copy logical row-major values into an owned or bound source. */
  setSource(name: string, value: TensorData): void {
    this.requireOpen();
    const source = this.sources.find((candidate) => candidate.name === name);
    if (!source) throw new Error(`unknown ZGC source ${name}`);
    if (source.binding === "embedded") throw new Error(`source ${name} is embedded`);
    const bytes = bytesFor(source.tensor, value);
    if (source.binding === "owned") {
      const pointer = this.allocate(bytes.byteLength, itemSize(source.tensor.dtype));
      try {
        new Uint8Array(this.exports.memory.buffer, pointer, bytes.byteLength).set(bytes);
        requireOk(this.exports.zgc_runtime_copy_source(this.storage, source.key, pointer, bytes.byteLength), "zgc_runtime_copy_source");
      } finally {
        this.release(pointer);
      }
      return;
    }
    let pointer = this.boundSources.get(source.key);
    if (pointer === undefined) {
      pointer = this.allocate(source.tensor.byteCount, itemSize(source.tensor.dtype));
      requireOk(this.exports.zgc_runtime_bind_source(this.storage, source.key, pointer, source.tensor.byteCount), "zgc_runtime_bind_source");
      this.boundSources.set(source.key, pointer);
    }
    new Uint8Array(this.exports.memory.buffer, pointer, source.tensor.byteCount)
      .set(physicalBytes(source.tensor, bytes));
  }

  /** Run once using the current source values. */
  execute(): void {
    this.requireOpen();
    requireOk(this.exports.zgc_runtime_run(this.storage), "zgc_runtime_run");
  }

  /** Copy one output into a JS-owned typed array. */
  output(index: number): TensorResult {
    this.requireOpen();
    const info = this.outputs[index];
    if (!info) throw new RangeError(`unknown ZGC output ${index}`);
    const pointer = this.allocate(info.byteCount, itemSize(info.dtype));
    try {
      requireOk(this.exports.zgc_runtime_copy_output(this.storage, index, pointer, info.byteCount), "zgc_runtime_copy_output");
      const bytes = new Uint8Array(this.exports.memory.buffer, pointer, info.byteCount);
      return { info, data: copiedData(info, bytes) };
    } finally {
      this.release(pointer);
    }
  }

  /** Set named sources, execute, and return copied outputs. */
  run(sources: Record<string, TensorData> = {}): TensorResult[] {
    for (const [name, value] of Object.entries(sources)) this.setSource(name, value);
    this.execute();
    return this.outputs.map((_, index) => this.output(index));
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.exports.zgc_runtime_deinit(this.storage);
    for (const pointer of this.allocations.keys()) this.release(pointer);
    this.boundSources.clear();
    this.boundOutputs.clear();
  }
}

/** Instantiate a compiled `.wasm` artifact without a compiler dependency. */
export async function load(bytes: BufferSource | WebAssembly.Module): Promise<Runtime> {
  const module = bytes instanceof WebAssembly.Module ? bytes : await WebAssembly.compile(bytes);
  return new Runtime(new WebAssembly.Instance(module, {}));
}
