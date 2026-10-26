const std = @import("std");
const Diagnostics = @import("diagnostics.zig");
const Graph = @import("../core/graph.zig");
const Semantic = @import("../operations/semantic.zig");
const Elementwise = @import("../operations/elementwise.zig");
const elementwise_operation = @import("../kernels/elementwise_operation.zig");
const Tensor = @import("../core/tensor.zig");
const ScalarValue = @import("../storage/dtype.zig").ScalarValue;

/// Provenance retained across semantic graph rewrites. Forward entries are
/// null when an original node or tensor was removed. Several original tensors
/// may map to one optimized tensor when an identity view is eliminated.
pub fn Provenance(comptime capacity: Graph.Capacity) type {
    return struct {
        raw_to_optimized_tensor: [capacity.max_tensors]?Tensor.Id = @splat(null),
        optimized_to_raw_tensor: [capacity.max_tensors]?Tensor.Id = @splat(null),
        raw_to_optimized_node: [capacity.max_nodes]?usize = @splat(null),
        optimized_to_raw_node: [capacity.max_nodes]?usize = @splat(null),
    };
}

pub fn Result(comptime capacity: Graph.Capacity) type {
    return struct {
        graph: Graph.Graph(capacity, Semantic.Op),
        provenance: Provenance(capacity),
    };
}

/// Apply semantics-preserving structural normalization, then remove every
/// value that cannot contribute to a declared graph output.
pub fn SemanticOptimization(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const RewriteResult = Result(capacity);

        pub fn optimize(comptime raw: SemanticGraph, comptime diagnostics: Diagnostics.Options) RewriteResult {
            const Trace = Diagnostics.Trace(diagnostics);
            Trace.begin(.semantic_canonicalization);
            const canonical = canonicalize(raw, diagnostics);
            Trace.end(.semantic_canonicalization, .{ .nodes = canonical.graph.node_ct, .tensors = canonical.graph.tensor_ct });
            Trace.begin(.semantic_dead_elimination);
            const compacted = eliminateDead(canonical.graph);
            Trace.end(.semantic_dead_elimination, .{ .nodes = compacted.graph.node_ct, .tensors = compacted.graph.tensor_ct });
            Trace.begin(.semantic_provenance);
            const provenance = composeProvenance(canonical.provenance, compacted.provenance);
            Trace.end(.semantic_provenance, .{ .entries = provenance.raw_to_optimized_tensor.len });
            return .{
                .graph = compacted.graph,
                .provenance = provenance,
            };
        }

        fn canonicalize(comptime raw: SemanticGraph, comptime diagnostics: Diagnostics.Options) RewriteResult {
            const Trace = Diagnostics.Trace(diagnostics);
            Trace.begin(.canonical_use_counts);
            const use_counts = countUses(raw);
            Trace.end(.canonical_use_counts, .{ .entries = use_counts.len });
            Trace.begin(.canonical_outputs);
            const is_output = outputSet(raw);
            Trace.end(.canonical_outputs, .{ .entries = is_output.len });
            Trace.begin(.canonical_concat_analysis);
            const folded_concats = findFoldedConcats(raw, use_counts, is_output);
            Trace.end(.canonical_concat_analysis, .{ .entries = folded_concats.len });
            Trace.begin(.canonical_tensor_order);
            const tensor_order = topologicalTensorOrder(raw);
            Trace.end(.canonical_tensor_order, .{ .entries = tensor_order.len });
            Trace.begin(.canonical_rewrite);
            var result: RewriteResult = .{ .graph = .init(), .provenance = .{} };

            // Prefer the original tensor order whenever it is dependency
            // valid, but defer a tensor whose producer, inputs, or storage
            // root have not been visited. Existing graph identities therefore
            // remain stable without treating tensor IDs as an execution order.
            inline for (tensor_order[0..raw.tensor_ct]) |old_tensor_id| {
                const old_info = raw.tensors[old_tensor_id].?;
                switch (old_info.origin) {
                    .source => |source_index| {
                        var info = old_info;
                        info.storage_tensor = mappedStorage(
                            result.provenance,
                            old_info.storage_tensor,
                            old_tensor_id,
                            result.graph.tensor_ct,
                        );
                        const new_tensor_id = insertTensor(&result, old_tensor_id, info);
                        var source = raw.sources[source_index].?;
                        source.tensor = new_tensor_id;
                        result.graph.insertSource(source_index, source);
                    },
                    .literal => {
                        var info = old_info;
                        info.storage_tensor = mappedStorage(
                            result.provenance,
                            old_info.storage_tensor,
                            old_tensor_id,
                            result.graph.tensor_ct,
                        );
                        _ = insertTensor(&result, old_tensor_id, info);
                    },
                    .node => |old_node_id| {
                        if (folded_concats[old_node_id]) continue;
                        const old_node = raw.nodes[old_node_id].?;
                        var op = old_node.op;
                        var inputs: [capacity.max_input_refs]Tensor.Id = undefined;
                        var input_count: usize = 0;

                        switch (old_node.op) {
                            .compute => |compute| switch (compute) {
                                .concat => |attrs| {
                                    appendConcatInputs(raw, folded_concats, old_node, attrs.axis, result.provenance, &inputs, &input_count);
                                },
                                else => appendMappedInputs(raw, old_node, result.provenance, &inputs, &input_count),
                            },
                            .view => |view| {
                                appendMappedInputs(raw, old_node, result.provenance, &inputs, &input_count);
                                composeView(&result.graph, &op, &inputs, &input_count, view);
                            },
                        }

                        if (op.kind() == .compute) {
                            const compute = op.compute;
                            if (foldScalarCompute(result.graph, compute, inputs[0..input_count])) |value| {
                                var info = old_info;
                                info.origin = .{ .literal = value };
                                info.storage_tensor = result.graph.tensor_ct;
                                _ = insertTensor(&result, old_tensor_id, info);
                                continue;
                            }
                            if (simplifyCompute(result.graph, compute, old_info, inputs[0..input_count])) |existing| {
                                result.provenance.raw_to_optimized_tensor[old_tensor_id] = existing;
                                continue;
                            }
                        }

                        if (op.kind() == .view) {
                            if (identityInput(result.graph, old_info, inputs[0..input_count])) |existing| {
                                result.provenance.raw_to_optimized_tensor[old_tensor_id] = existing;
                                continue;
                            }
                        }

                        const new_node_id = result.graph.node_ct;
                        var info = old_info;
                        info.origin = .{ .node = new_node_id };
                        info.storage_tensor = mappedStorage(result.provenance, old_info.storage_tensor, old_tensor_id, result.graph.tensor_ct);
                        const new_tensor_id = insertTensor(&result, old_tensor_id, info);
                        for (inputs[0..input_count]) |input_id| result.graph.insertRef(input_id);
                        result.graph.insertNode(.{
                            .op = op,
                            .input_start = result.graph.input_ref_ct - input_count,
                            .input_count = input_count,
                            .result = new_tensor_id,
                        });
                        result.provenance.raw_to_optimized_node[old_node_id] = new_node_id;
                        result.provenance.optimized_to_raw_node[new_node_id] = old_node_id;
                    },
                }
            }

            inline for (0..raw.output_ct) |output_index| {
                const old_output = raw.outputs[output_index].?;
                result.graph.insertOutput(result.provenance.raw_to_optimized_tensor[old_output] orelse
                    @compileError("semantic canonicalization removed a graph output"));
            }
            Trace.end(.canonical_rewrite, .{ .nodes = result.graph.node_ct, .tensors = result.graph.tensor_ct });
            return result;
        }

        fn eliminateDead(comptime source: SemanticGraph) RewriteResult {
            var live_tensors: [capacity.max_tensors]bool = @splat(false);
            var live_nodes: [capacity.max_nodes]bool = @splat(false);
            inline for (0..source.output_ct) |output_index| {
                markLive(source, source.outputs[output_index].?, &live_tensors, &live_nodes);
            }

            var result: RewriteResult = .{ .graph = .init(), .provenance = .{} };
            var node_map: [capacity.max_nodes]?usize = @splat(null);
            comptime var next_node: usize = 0;
            inline for (0..source.node_ct) |old_node_id| {
                if (live_nodes[old_node_id]) {
                    node_map[old_node_id] = next_node;
                    next_node += 1;
                }
            }

            comptime var next_tensor: usize = 0;
            inline for (0..source.tensor_ct) |old_tensor_id| {
                if (live_tensors[old_tensor_id]) {
                    result.provenance.raw_to_optimized_tensor[old_tensor_id] = next_tensor;
                    result.provenance.optimized_to_raw_tensor[next_tensor] = old_tensor_id;
                    next_tensor += 1;
                }
            }

            inline for (0..source.tensor_ct) |old_tensor_id| {
                if (!live_tensors[old_tensor_id]) continue;
                var info = source.tensors[old_tensor_id].?;
                info.storage_tensor = result.provenance.raw_to_optimized_tensor[info.storage_tensor] orelse
                    @compileError("live tensor aliases dead storage");
                switch (info.origin) {
                    .source => |source_index| {
                        const new_tensor_id = result.graph.insertTensor(info);
                        var source_info = source.sources[source_index].?;
                        source_info.tensor = new_tensor_id;
                        result.graph.insertSource(source_index, source_info);
                    },
                    .literal => _ = result.graph.insertTensor(info),
                    .node => |old_node_id| {
                        info.origin = .{ .node = node_map[old_node_id] orelse
                            @compileError("live tensor has a dead producer") };
                        _ = result.graph.insertTensor(info);
                    },
                }
            }

            inline for (0..source.node_ct) |old_node_id| {
                if (!live_nodes[old_node_id]) continue;
                const node = source.nodes[old_node_id].?;
                inline for (0..node.input_count) |input_index| {
                    const old_input = source.input_refs[node.input_start + input_index].?;
                    result.graph.insertRef(result.provenance.raw_to_optimized_tensor[old_input] orelse
                        @compileError("live node refers to a dead tensor"));
                }
                result.graph.insertNode(.{
                    .op = node.op,
                    .input_start = result.graph.input_ref_ct - node.input_count,
                    .input_count = node.input_count,
                    .result = result.provenance.raw_to_optimized_tensor[node.result].?,
                });
                const new_node_id = node_map[old_node_id].?;
                result.provenance.raw_to_optimized_node[old_node_id] = new_node_id;
                result.provenance.optimized_to_raw_node[new_node_id] = old_node_id;
            }
            inline for (0..source.output_ct) |output_index| {
                result.graph.insertOutput(result.provenance.raw_to_optimized_tensor[source.outputs[output_index].?].?);
            }
            return result;
        }

        fn insertTensor(result: *RewriteResult, comptime old_tensor_id: usize, info: SemanticGraph.TensorInfo) Tensor.Id {
            const new_tensor_id = result.graph.insertTensor(info);
            result.provenance.raw_to_optimized_tensor[old_tensor_id] = new_tensor_id;
            result.provenance.optimized_to_raw_tensor[new_tensor_id] = old_tensor_id;
            return new_tensor_id;
        }

        fn mappedStorage(
            provenance: Provenance(capacity),
            old_storage: Tensor.Id,
            old_result: Tensor.Id,
            new_result: Tensor.Id,
        ) Tensor.Id {
            if (old_storage == old_result) return new_result;
            return provenance.raw_to_optimized_tensor[old_storage] orelse
                @compileError("tensor storage root was removed before its alias");
        }

        // Iterative DFS to produce a topological ordering of tensors.
        // The order is not unique, but it is guaranteed that every tensor appears after its producer and all of its inputs.
        fn topologicalTensorOrder(
            comptime graph: SemanticGraph,
        ) [capacity.max_tensors]usize {
            const n = graph.tensor_ct;
            // Graph construction and prior rewrites usually retain dependency
            // order. Check edges once and reuse IDs before allocating DFS state.
            var existing_order: [capacity.max_tensors]usize = @splat(0);
            var ordered = true;
            for (0..n) |id| {
                existing_order[id] = id;
                const info = graph.tensors[id].?;
                if (info.storage_tensor != id and info.storage_tensor >= id) ordered = false;
                if (info.origin == .node) {
                    const node = graph.nodes[info.origin.node].?;
                    for (0..node.input_count) |index| {
                        if (graph.input_refs[node.input_start + index].? >= id) ordered = false;
                    }
                }
            }
            if (ordered) return existing_order;

            // 0 = unseen, 1 = visiting, 2 = completed
            var state: [capacity.max_tensors]u8 = @splat(0);
            var order: [capacity.max_tensors]usize = @splat(0);

            const Frame = struct {
                tensor_id: usize,
                next_dep: usize,
            };

            var stack: [capacity.max_tensors]Frame = undefined;
            var depth: usize = 0;
            var count: usize = 0;

            const Dependency = struct {
                fn countDeps(g: *const SemanticGraph, id: usize) usize {
                    const info = g.tensors[id].?;
                    const storage_count: usize =
                        if (info.storage_tensor != id) 1 else 0;

                    return storage_count + switch (info.origin) {
                        .source, .literal => @as(usize, 0),
                        .node => |node_id| g.nodes[node_id].?.input_count,
                    };
                }

                fn at(g: *const SemanticGraph, id: usize, index: usize) usize {
                    const info = g.tensors[id].?;

                    if (info.storage_tensor != id) {
                        if (index == 0) return info.storage_tensor;
                        return nodeInput(g, info, index - 1);
                    }

                    return nodeInput(g, info, index);
                }

                fn nodeInput(
                    g: *const SemanticGraph,
                    info: anytype,
                    index: usize,
                ) usize {
                    return switch (info.origin) {
                        .node => |node_id| blk: {
                            const node = g.nodes[node_id].?;
                            break :blk g.input_refs[node.input_start + index].?;
                        },
                        .source, .literal => unreachable,
                    };
                }
            };

            for (0..n) |root| {
                if (state[root] != 0) continue;

                stack[depth] = .{ .tensor_id = root, .next_dep = 0 };
                state[root] = 1;
                depth += 1;

                while (depth > 0) {
                    const frame = &stack[depth - 1];
                    const id = frame.tensor_id;
                    const dep_count = Dependency.countDeps(&graph, id);

                    if (frame.next_dep < dep_count) {
                        const dep = Dependency.at(&graph, id, frame.next_dep);
                        frame.next_dep += 1;

                        switch (state[dep]) {
                            0 => {
                                state[dep] = 1;
                                stack[depth] = .{
                                    .tensor_id = dep,
                                    .next_dep = 0,
                                };
                                depth += 1;
                            },
                            1 => @compileError(
                                "semantic graph contains a tensor dependency cycle",
                            ),
                            2 => {},
                            else => unreachable,
                        }
                    } else {
                        // All dependencies have been emitted.
                        order[count] = id;
                        count += 1;
                        state[id] = 2;
                        depth -= 1;
                    }
                }
            }

            return order;
        }

        fn countUses(comptime graph: SemanticGraph) [capacity.max_tensors]usize {
            var uses: [capacity.max_tensors]usize = @splat(0);
            for (0..graph.input_ref_ct) |ref_index| uses[graph.input_refs[ref_index].?] += 1;
            return uses;
        }

        fn outputSet(comptime graph: SemanticGraph) [capacity.max_tensors]bool {
            var outputs: [capacity.max_tensors]bool = @splat(false);
            for (0..graph.output_ct) |index| outputs[graph.outputs[index].?] = true;
            return outputs;
        }

        fn findFoldedConcats(
            comptime graph: SemanticGraph,
            comptime uses: [capacity.max_tensors]usize,
            comptime outputs: [capacity.max_tensors]bool,
        ) [capacity.max_nodes]bool {
            var folded: [capacity.max_nodes]bool = @splat(false);
            for (0..graph.node_ct) |consumer_id| {
                const consumer = graph.nodes[consumer_id].?;
                const consumer_attrs = switch (consumer.op) {
                    .compute => |compute| switch (compute) {
                        .concat => |attrs| attrs,
                        else => continue,
                    },
                    .view => continue,
                };
                for (0..consumer.input_count) |input_index| {
                    const input_id = graph.input_refs[consumer.input_start + input_index].?;
                    if (uses[input_id] != 1 or outputs[input_id]) continue;
                    const producer_id = switch (graph.tensors[input_id].?.origin) {
                        .node => |id| id,
                        .source, .literal => continue,
                    };
                    const producer_attrs = switch (graph.nodes[producer_id].?.op) {
                        .compute => |compute| switch (compute) {
                            .concat => |attrs| attrs,
                            else => continue,
                        },
                        .view => continue,
                    };
                    if (producer_attrs.axis == consumer_attrs.axis) folded[producer_id] = true;
                }
            }
            return folded;
        }

        fn appendConcatInputs(
            comptime graph: SemanticGraph,
            comptime folded: [capacity.max_nodes]bool,
            comptime node: SemanticGraph.Node,
            comptime axis: i8,
            provenance: Provenance(capacity),
            inputs: *[capacity.max_input_refs]Tensor.Id,
            input_count: *usize,
        ) void {
            for (0..node.input_count) |input_index| {
                const tensor_id = graph.input_refs[node.input_start + input_index].?;
                const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                    .node => |id| id,
                    .source, .literal => {
                        appendMappedInput(tensor_id, provenance, inputs, input_count);
                        continue;
                    },
                };
                if (folded[producer_id]) {
                    const producer = graph.nodes[producer_id].?;
                    const producer_axis = switch (producer.op.compute) {
                        .concat => |attrs| attrs.axis,
                        else => unreachable,
                    };
                    if (producer_axis == axis) {
                        appendConcatInputs(graph, folded, producer, axis, provenance, inputs, input_count);
                        continue;
                    }
                }
                appendMappedInput(tensor_id, provenance, inputs, input_count);
            }
        }

        fn appendMappedInputs(
            comptime graph: SemanticGraph,
            comptime node: SemanticGraph.Node,
            provenance: Provenance(capacity),
            inputs: *[capacity.max_input_refs]Tensor.Id,
            input_count: *usize,
        ) void {
            for (0..node.input_count) |input_index| {
                appendMappedInput(graph.input_refs[node.input_start + input_index].?, provenance, inputs, input_count);
            }
        }

        fn appendMappedInput(
            comptime old_tensor_id: Tensor.Id,
            provenance: Provenance(capacity),
            inputs: *[capacity.max_input_refs]Tensor.Id,
            input_count: *usize,
        ) void {
            inputs[input_count.*] = provenance.raw_to_optimized_tensor[old_tensor_id] orelse
                @compileError("semantic rewrite refers to a tensor that has not been produced");
            input_count.* += 1;
        }

        fn composeView(
            graph: *const SemanticGraph,
            op: *Semantic.Op,
            inputs: *[capacity.max_input_refs]Tensor.Id,
            input_count: *usize,
            comptime view: Semantic.Op.View,
        ) void {
            if (input_count.* != 1) return;
            const input_id = inputs[0];
            const producer_id = switch (graph.tensors[input_id].?.origin) {
                .node => |id| id,
                .source, .literal => return,
            };
            const producer = graph.nodes[producer_id].?;
            const producer_input = graph.input_refs[producer.input_start].?;
            switch (view) {
                .slice => |child| switch (producer.op) {
                    .view => |producer_view| switch (producer_view) {
                        .slice => |parent| if (parent.axis == child.axis) {
                            const start = parent.start + child.start * parent.step;
                            const step = parent.step * child.step;
                            op.* = .{ .view = .{ .slice = .{
                                .axis = child.axis,
                                .start = start,
                                .length = child.length,
                                .step = step,
                            } } };
                            inputs[0] = producer_input;
                        },
                        else => {},
                    },
                    .compute => {},
                },
                .reshape => switch (producer.op) {
                    .view => |producer_view| switch (producer_view) {
                        .reshape => inputs[0] = producer_input,
                        else => {},
                    },
                    .compute => {},
                },
                else => {},
            }
        }

        fn identityInput(
            comptime graph: SemanticGraph,
            comptime original_output: SemanticGraph.TensorInfo,
            comptime inputs: []const Tensor.Id,
        ) ?Tensor.Id {
            if (inputs.len != 1) return null;
            var candidate: ?Tensor.Id = inputs[0];
            while (candidate) |tensor_id| {
                const info = graph.tensors[tensor_id].?;
                if (sameGeometry(original_output, info)) return tensor_id;
                candidate = switch (info.origin) {
                    .node => |node_id| blk: {
                        const node = graph.nodes[node_id].?;
                        if (node.op.kind() != .view or node.input_count != 1) break :blk null;
                        break :blk graph.input_refs[node.input_start].?;
                    },
                    .source, .literal => null,
                };
            }
            return null;
        }

        fn sameGeometry(lhs: SemanticGraph.TensorInfo, rhs: SemanticGraph.TensorInfo) bool {
            return lhs.dtype == rhs.dtype and
                lhs.shape.rank == rhs.shape.rank and
                std.mem.eql(usize, lhs.shape.slice(), rhs.shape.slice()) and
                lhs.layout.offset == rhs.layout.offset and
                std.mem.eql(isize, lhs.layout.strides[0..lhs.shape.rank], rhs.layout.strides[0..rhs.shape.rank]);
        }

        fn foldScalarCompute(
            comptime graph: SemanticGraph,
            comptime compute: Semantic.Op.Compute,
            comptime inputs: []const Tensor.Id,
        ) ?ScalarValue {
            const operation = Elementwise.fromCompute(compute) orelse return null;
            for (inputs) |input_id| {
                const info = graph.tensors[input_id].?;
                if (info.shape.rank != 0 or info.origin != .literal) return null;
            }
            const output_dtype = compute.inferDtype(inputInfos(graph, inputs));
            return switch (operation) {
                .select => foldSelect(graph, output_dtype, inputs),
                else => foldHomogeneous(graph, operation, output_dtype, inputs),
            };
        }

        fn InputInfos(comptime count: usize) type {
            return [count]SemanticGraph.TensorInfo;
        }

        fn inputInfos(comptime graph: SemanticGraph, comptime inputs: []const Tensor.Id) InputInfos(inputs.len) {
            var infos: InputInfos(inputs.len) = undefined;
            for (inputs, 0..) |input_id, index| infos[index] = graph.tensors[input_id].?;
            return infos;
        }

        fn foldHomogeneous(
            comptime graph: SemanticGraph,
            comptime operation: Elementwise.Operation,
            comptime output_dtype: @import("../storage/dtype.zig").Dtype,
            comptime inputs: []const Tensor.Id,
        ) ScalarValue {
            const input_dtype = graph.tensors[inputs[0]].?.dtype;
            return switch (input_dtype) {
                inline else => |dtype| blk: {
                    var params: [inputs.len]dtype.Scalar() = undefined;
                    for (inputs, 0..) |input_id, index| {
                        params[index] = graph.tensors[input_id].?.origin.literal.get(dtype);
                    }
                    const value = elementwise_operation.evaluateScalar(output_dtype, operation, params);
                    break :blk ScalarValue.init(output_dtype, value);
                },
            };
        }

        fn foldSelect(
            comptime graph: SemanticGraph,
            comptime output_dtype: @import("../storage/dtype.zig").Dtype,
            comptime inputs: []const Tensor.Id,
        ) ScalarValue {
            const condition = graph.tensors[inputs[0]].?.origin.literal.get(.bool);
            return switch (output_dtype) {
                inline else => |dtype| blk: {
                    const when_true = graph.tensors[inputs[1]].?.origin.literal.get(dtype);
                    const when_false = graph.tensors[inputs[2]].?.origin.literal.get(dtype);
                    break :blk ScalarValue.init(dtype, if (condition) when_true else when_false);
                },
            };
        }

        fn simplifyCompute(
            comptime graph: SemanticGraph,
            comptime compute: Semantic.Op.Compute,
            comptime output: SemanticGraph.TensorInfo,
            comptime inputs: []const Tensor.Id,
        ) ?Tensor.Id {
            const selected: ?usize = switch (compute) {
                .add => if (isIntegerZero(graph, inputs[1])) 0 else if (isIntegerZero(graph, inputs[0])) 1 else null,
                .sub => if (isNumericZero(graph, inputs[1])) 0 else null,
                .mul => if (isNumericOne(graph, inputs[1])) 0 else if (isNumericOne(graph, inputs[0])) 1 else null,
                .div => if (isNumericOne(graph, inputs[1])) 0 else null,
                .logical_and => if (isBoolLiteral(graph, inputs[1], true)) 0 else if (isBoolLiteral(graph, inputs[0], true)) 1 else null,
                .logical_or => if (isBoolLiteral(graph, inputs[1], false)) 0 else if (isBoolLiteral(graph, inputs[0], false)) 1 else null,
                .where => if (literalBool(graph, inputs[0])) |condition| if (condition) 1 else 2 else null,
                else => null,
            };
            const index = selected orelse return null;
            const candidate = inputs[index];
            return if (sameGeometry(output, graph.tensors[candidate].?)) candidate else null;
        }

        fn literalBool(comptime graph: SemanticGraph, comptime tensor_id: Tensor.Id) ?bool {
            const info = graph.tensors[tensor_id].?;
            if (info.shape.rank != 0 or info.dtype != .bool or info.origin != .literal) return null;
            return info.origin.literal.get(.bool);
        }

        fn isBoolLiteral(comptime graph: SemanticGraph, comptime tensor_id: Tensor.Id, comptime expected: bool) bool {
            return if (literalBool(graph, tensor_id)) |value| value == expected else false;
        }

        fn isIntegerZero(comptime graph: SemanticGraph, comptime tensor_id: Tensor.Id) bool {
            const info = graph.tensors[tensor_id].?;
            if (info.dtype.kind() != .signed_integer and info.dtype.kind() != .unsigned_integer) return false;
            return isNumericZero(graph, tensor_id);
        }

        fn isNumericZero(comptime graph: SemanticGraph, comptime tensor_id: Tensor.Id) bool {
            const info = graph.tensors[tensor_id].?;
            if (info.shape.rank != 0 or info.origin != .literal or info.dtype.kind() == .boolean) return false;
            return switch (info.dtype) {
                .f32 => info.origin.literal.bits == 0,
                .f16 => info.origin.literal.bits == 0,
                .i8 => info.origin.literal.get(.i8) == 0,
                .i32 => info.origin.literal.get(.i32) == 0,
                .u8 => info.origin.literal.get(.u8) == 0,
                .bool => false,
            };
        }

        fn isNumericOne(comptime graph: SemanticGraph, comptime tensor_id: Tensor.Id) bool {
            const info = graph.tensors[tensor_id].?;
            if (info.shape.rank != 0 or info.origin != .literal or info.dtype.kind() == .boolean) return false;
            return switch (info.dtype) {
                .f32 => info.origin.literal.get(.f32) == 1,
                .f16 => info.origin.literal.get(.f16) == 1,
                .i8 => info.origin.literal.get(.i8) == 1,
                .i32 => info.origin.literal.get(.i32) == 1,
                .u8 => info.origin.literal.get(.u8) == 1,
                .bool => false,
            };
        }

        fn markLive(
            comptime graph: SemanticGraph,
            comptime tensor_id: Tensor.Id,
            live_tensors: *[capacity.max_tensors]bool,
            live_nodes: *[capacity.max_nodes]bool,
        ) void {
            if (live_tensors[tensor_id]) return;
            live_tensors[tensor_id] = true;
            const node_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return,
            };
            live_nodes[node_id] = true;
            const node = graph.nodes[node_id].?;
            for (0..node.input_count) |input_index| {
                markLive(graph, graph.input_refs[node.input_start + input_index].?, live_tensors, live_nodes);
            }
        }

        fn composeProvenance(
            comptime first: Provenance(capacity),
            comptime second: Provenance(capacity),
        ) Provenance(capacity) {
            var result: Provenance(capacity) = .{};
            for (first.raw_to_optimized_tensor, 0..) |intermediate, raw_id| {
                if (intermediate) |id| result.raw_to_optimized_tensor[raw_id] = second.raw_to_optimized_tensor[id];
            }
            for (second.optimized_to_raw_tensor, 0..) |intermediate, optimized_id| {
                if (intermediate) |id| result.optimized_to_raw_tensor[optimized_id] = first.optimized_to_raw_tensor[id];
            }
            for (first.raw_to_optimized_node, 0..) |intermediate, raw_id| {
                if (intermediate) |id| result.raw_to_optimized_node[raw_id] = second.raw_to_optimized_node[id];
            }
            for (second.optimized_to_raw_node, 0..) |intermediate, optimized_id| {
                if (intermediate) |id| result.optimized_to_raw_node[optimized_id] = first.optimized_to_raw_node[id];
            }
            return result;
        }
    };
}
