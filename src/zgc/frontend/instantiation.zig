//! Allocation-free compile-time replay of `.zgir` definitions.
const std = @import("std");
const ir = @import("serialized.zig");
const def = @import("definition.zig");
const Dtype = @import("../storage/dtype.zig").Dtype;
const Source = @import("../storage/source.zig");

pub fn modelFromZgir(comptime text: []const u8) type {
    @setEvalBranchQuota(10_000_000);
    validateHeader(text);
    const names = sourceNames(text);
    var builder = def.DefinitionBuilder.init();
    const sources = builder.serializedSources(&names);
    var values: [nodeCount(text)]def.Value = undefined;
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    var node_index: usize = 0;
    var source_index: usize = 0;
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "outputs")) {
            const output_text = afterEqual(line);
            const outputs = referenceIndices(.{ .text = output_text, .line = 1 }, node_index);
            inline for (outputs) |output| builder.output(values[output]);
            break;
        }
        const call = parseCall(line, node_index);
        values[node_index] = if (isSource(call.operation)) blk: {
            const key: def.SerializedSourceKey = @enumFromInt(source_index);
            source_index += 1;
            const dtype = parseDtype(call.attr("dtype"));
            const shape = integers(call.attr("shape"), usize);
            break :blk switch (call.operation) {
                .input => sources.input(key, dtype, &shape),
                .parameter => sources.parameter(key, dtype, &shape),
                .constant => sources.constant(key, dtype, &shape),
                else => unreachable,
            };
        } else replay(&builder, call, &values);
        node_index += 1;
    }
    const definition = builder.finish();
    const Definition = @TypeOf(definition);
    const bound_count = boundCount(text);
    if (bound_count == 0) return definition.model();
    var overrides: [bound_count]Definition.SourceOverride = undefined;
    lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    node_index = 0;
    source_index = 0;
    var override_index: usize = 0;
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#' or std.mem.startsWith(u8, line, "outputs")) continue;
        const call = parseCall(line, node_index);
        node_index += 1;
        if (!isSource(call.operation)) continue;
        if (parseAtom(call.attr("binding")) == .bound) {
            overrides[override_index] = .{ .source = @enumFromInt(source_index), .binding = Source.bound };
            override_index += 1;
        }
        source_index += 1;
    }
    return definition.modelWith(&overrides);
}

fn replay(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    return switch (c.operation) {
        .scalar => scalar(b, parseDtype(c.attr("dtype")), c.attr("value")),
        .full => full(b, parseDtype(c.attr("dtype")), c.attr("shape"), c.attr("value")),
        .relu => b.relu(c.value(0, v)),
        .exp => b.exp(c.value(0, v)),
        .neg => b.neg(c.value(0, v)),
        .abs => b.abs(c.value(0, v)),
        .sqrt => b.sqrt(c.value(0, v)),
        .log => b.log(c.value(0, v)),
        .reciprocal => b.reciprocal(c.value(0, v)),
        .copy => b.copy(c.value(0, v)),
        .contiguous => b.contiguous(c.value(0, v)),
        .add => b.add(c.value(0, v), c.value(1, v)),
        .sub => b.sub(c.value(0, v), c.value(1, v)),
        .mul => b.mul(c.value(0, v), c.value(1, v)),
        .div => b.div(c.value(0, v), c.value(1, v)),
        .minimum => b.minimum(c.value(0, v), c.value(1, v)),
        .maximum => b.maximum(c.value(0, v), c.value(1, v)),
        .equal => b.equal(c.value(0, v), c.value(1, v)),
        .not_equal => b.notEqual(c.value(0, v), c.value(1, v)),
        .less_than => b.lessThan(c.value(0, v), c.value(1, v)),
        .less_equal => b.lessEqual(c.value(0, v), c.value(1, v)),
        .greater_than => b.greaterThan(c.value(0, v), c.value(1, v)),
        .greater_equal => b.greaterEqual(c.value(0, v), c.value(1, v)),
        .logical_and => b.logicalAnd(c.value(0, v), c.value(1, v)),
        .logical_or => b.logicalOr(c.value(0, v), c.value(1, v)),
        .logical_not => b.logicalNot(c.value(0, v)),
        .matmul => b.matmul(c.value(0, v), c.value(1, v)),
        .clamp => b.clamp(c.value(0, v), c.value(1, v), c.value(2, v)),
        .where => b.where(c.value(0, v), c.value(1, v), c.value(2, v)),
        .pad => pad(b, c, v),
        .shift => shift(b, c, v),
        .sum, .mean, .min, .max => reduction(b, c, v),
        .concat => concat(b, c, v),
        .softmax => b.softmax(c.value(0, v), int(i8, c.attr("axis"))),
        .transpose => b.transpose(c.value(0, v), int(i8, c.attr("axis_a")), int(i8, c.attr("axis_b"))),
        .reshape => reshape(b, c, v),
        .broadcast_to => broadcast(b, c, v),
        .flatten => b.flatten(c.value(0, v), .{ .start_axis = int(i8, c.attr("start_axis")), .end_axis = int(i8, c.attr("end_axis")) }),
        .squeeze => b.squeeze(c.value(0, v), int(i8, c.attr("axis"))),
        .unsqueeze => b.unsqueeze(c.value(0, v), int(i8, c.attr("axis"))),
        .permute => permute(b, c, v),
        .slice => slice(b, c, v),
        .windows => windows(b, c, v),
        .slice_loop => sliceLoop(b, c, v),
        .input, .parameter, .constant => unreachable,
    };
}

fn scalar(comptime b: *def.DefinitionBuilder, comptime d: Dtype, comptime x: Segment) def.Value {
    return switch (d) {
        .f32 => b.scalar(.f32, number(f32, x)),
        .f16 => b.scalar(.f16, number(f16, x)),
        .i8 => b.scalar(.i8, int(i8, x)),
        .bool => b.scalar(.bool, boolean(x)),
    };
}
fn full(comptime b: *def.DefinitionBuilder, comptime d: Dtype, comptime s: Segment, comptime x: Segment) def.Value {
    const shape = integers(s, usize);
    return switch (d) {
        .f32 => b.full(.f32, &shape, number(f32, x)),
        .f16 => b.full(.f16, &shape, number(f16, x)),
        .i8 => b.full(.i8, &shape, int(i8, x)),
        .bool => b.full(.bool, &shape, boolean(x)),
    };
}
fn pad(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const before = integers(c.attr("before"), usize);
    const after = integers(c.attr("after"), usize);
    return b.pad(c.value(0, v), c.value(1, v), .{ .before = &before, .after = &after });
}
fn shift(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const offsets = integers(c.attr("offsets"), isize);
    return switch (parseAtom(c.attr("boundary"))) {
        .wrap => b.shift(c.value(0, v), &offsets, .wrap),
        .edge => b.shift(c.value(0, v), &offsets, .edge),
        .reflect => b.shift(c.value(0, v), &offsets, .reflect),
        .constant => b.shift(c.value(0, v), &offsets, .{ .constant = reference(c.attr("fill"), v) }),
        else => c.fail("invalid shift boundary"),
    };
}
fn sliceLoop(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const encoded = c.attr("iterations");
    var items = listParts(encoded);
    var result: [listLen(encoded)]def.DefinitionBuilder.SliceLoopIteration = undefined;
    var index: usize = 0;
    while (items.next()) |item| : (index += 1) {
        const body = nestedBody(item, "iteration");
        const offsets_text = namedPart(body, "offsets");
        const offsets = integers(offsets_text, isize);
        const boundary = namedPart(body, "boundary");
        result[index] = .{
            .offsets = &offsets,
            .boundary = if (std.mem.startsWith(u8, boundary.text, "redirect("))
                .{ .redirect = int(usize, firstNestedArgument(boundary, "redirect")) }
            else switch (parseAtom(boundary)) {
                .wrap => .wrap,
                .edge => .edge,
                .reflect => .reflect,
                else => c.fail("invalid slice_loop boundary"),
            },
        };
    }
    return b.sliceLoop(c.value(0, v), .{
        .axis = int(i8, c.attr("axis")),
        .iterations = &result,
    });
}
fn reduction(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const x = c.value(0, v);
    const keep = boolean(c.attr("keep_dims"));
    const a = c.attr("axes");
    if (std.mem.eql(u8, a.text, "null")) return reduce(b, c.operation, x, .{ .keep_dims = keep });
    const axes = integers(a, i8);
    return reduce(b, c.operation, x, .{ .axes = &axes, .keep_dims = keep });
}
fn reduce(comptime b: *def.DefinitionBuilder, comptime op: ir.Operation, comptime x: def.Value, comptime o: def.ReductionOptions) def.Value {
    return switch (op) {
        .sum => b.sum(x, o),
        .mean => b.mean(x, o),
        .min => b.min(x, o),
        .max => b.max(x, o),
        else => unreachable,
    };
}
fn concat(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const xs = references(c.arg(0), v);
    return b.concat(&xs, int(i8, c.attr("axis")));
}
fn reshape(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const s = integers(c.attr("shape"), usize);
    return b.reshape(c.value(0, v), &s);
}
fn broadcast(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const s = integers(c.attr("shape"), usize);
    return b.broadcastTo(c.value(0, v), &s);
}
fn permute(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const a = integers(c.attr("axes"), i8);
    return b.permute(c.value(0, v), &a);
}
fn slice(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const e = c.attr("end");
    return b.slice(c.value(0, v), .{ .axis = int(i8, c.attr("axis")), .start = int(usize, c.attr("start")), .end = if (std.mem.eql(u8, e.text, "null")) null else int(usize, e), .step = int(usize, c.attr("step")) });
}
fn windows(comptime b: *def.DefinitionBuilder, comptime c: Call, comptime v: []const def.Value) def.Value {
    const sizes = integers(c.attr("sizes"), usize);
    const st = c.attr("strides");
    const di = c.attr("dilations");
    if (isNull(st) and isNull(di)) return b.windows(c.value(0, v), .{ .sizes = &sizes });
    if (isNull(st)) {
        const ds = integers(di, usize);
        return b.windows(c.value(0, v), .{ .sizes = &sizes, .dilations = &ds });
    }
    if (isNull(di)) {
        const ss = integers(st, usize);
        return b.windows(c.value(0, v), .{ .sizes = &sizes, .strides = &ss });
    }
    const ss = integers(st, usize);
    const ds = integers(di, usize);
    return b.windows(c.value(0, v), .{ .sizes = &sizes, .strides = &ss, .dilations = &ds });
}

const Segment = struct { text: []const u8, line: usize };
const Named = struct { name: []const u8, value: Segment };
const Call = struct {
    operation: ir.Operation,
    args: []const Segment,
    attrs: []const Named,
    line: usize,
    fn arg(comptime s: Call, comptime i: usize) Segment {
        if (i >= s.args.len) s.fail("missing argument");
        return s.args[i];
    }
    fn attr(comptime s: Call, comptime name: []const u8) Segment {
        for (s.attrs) |a| if (std.mem.eql(u8, a.name, name)) return a.value;
        s.fail("missing attribute '" ++ name ++ "'");
    }
    fn value(comptime s: Call, comptime i: usize, comptime v: []const def.Value) def.Value {
        return reference(s.arg(i), v);
    }
    fn fail(comptime s: Call, comptime msg: []const u8) noreturn {
        fatal(s.line, msg);
    }
};
fn CallData(comptime n: usize) type {
    return struct { args: [n]Segment = undefined, ac: usize = 0, attrs: [n]Named = undefined, nc: usize = 0 };
}

fn parseCall(comptime line: []const u8, comptime expected: usize) Call {
    const eq = std.mem.indexOfScalar(u8, line, '=') orelse fatal(1, "expected assignment");
    const lhs = std.mem.trim(u8, line[0..eq], " \t");
    if (lhs.len < 2 or lhs[0] != '%' or (std.fmt.parseInt(usize, lhs[1..], 10) catch fatal(1, "invalid value id")) != expected) fatal(1, "value ids must be sequential");
    const rhs = std.mem.trim(u8, line[eq + 1 ..], " \t");
    const open = std.mem.indexOfScalar(u8, rhs, '(') orelse fatal(1, "expected operation call");
    if (rhs.len == 0 or rhs[rhs.len - 1] != ')') fatal(1, "unterminated operation");
    const op = std.meta.stringToEnum(ir.Operation, std.mem.trim(u8, rhs[0..open], " \t")) orelse fatal(1, "unknown operation");
    const body = Segment{ .text = rhs[open + 1 .. rhs.len - 1], .line = 1 };
    var data: CallData(line.len) = .{};
    var it = Parts.init(body);
    var named = false;
    while (it.next()) |part| if (topEqual(part.text)) |at| {
        named = true;
        data.attrs[data.nc] = .{ .name = std.mem.trim(u8, part.text[0..at], " \t"), .value = .{ .text = std.mem.trim(u8, part.text[at + 1 ..], " \t"), .line = 1 } };
        data.nc += 1;
    } else {
        if (named) fatal(1, "positional argument follows attribute");
        data.args[data.ac] = part;
        data.ac += 1;
    };
    return .{ .operation = op, .args = data.args[0..data.ac], .attrs = data.attrs[0..data.nc], .line = 1 };
}

const Parts = struct {
    s: Segment,
    i: usize = 0,
    fn init(comptime s: Segment) Parts {
        return .{ .s = s };
    }
    fn next(comptime self: *Parts) ?Segment {
        while (self.i < self.s.text.len and std.ascii.isWhitespace(self.s.text[self.i])) self.i += 1;
        if (self.i == self.s.text.len) return null;
        const start = self.i;
        var square: usize = 0;
        var round: usize = 0;
        var quote = false;
        var escape = false;
        while (self.i < self.s.text.len) : (self.i += 1) {
            const ch = self.s.text[self.i];
            if (quote) {
                if (escape) escape = false else if (ch == '\\') escape = true else if (ch == '"') quote = false;
                continue;
            }
            switch (ch) {
                '"' => quote = true,
                '[' => square += 1,
                ']' => square -= 1,
                '(' => round += 1,
                ')' => round -= 1,
                ',' => if (square == 0 and round == 0) break,
                else => {},
            }
        }
        const end = self.i;
        if (self.i < self.s.text.len) self.i += 1;
        return .{ .text = std.mem.trim(u8, self.s.text[start..end], " \t"), .line = self.s.line };
    }
};

fn listParts(comptime s: Segment) Parts {
    if (s.text.len < 2 or s.text[0] != '[' or s.text[s.text.len - 1] != ']') fatal(s.line, "expected list");
    return Parts.init(.{ .text = s.text[1 .. s.text.len - 1], .line = s.line });
}
fn nestedBody(comptime s: Segment, comptime name: []const u8) Segment {
    if (!std.mem.startsWith(u8, s.text, name ++ "(") or s.text[s.text.len - 1] != ')') fatal(s.line, "invalid nested call");
    return .{ .text = s.text[name.len + 1 .. s.text.len - 1], .line = s.line };
}
fn firstNestedArgument(comptime s: Segment, comptime name: []const u8) Segment {
    var parts = Parts.init(nestedBody(s, name));
    return parts.next() orelse fatal(s.line, "missing nested argument");
}
fn namedPart(comptime body: Segment, comptime name: []const u8) Segment {
    var parts = Parts.init(body);
    while (parts.next()) |part| {
        const at = topEqual(part.text) orelse continue;
        if (std.mem.eql(u8, std.mem.trim(u8, part.text[0..at], " \t"), name)) {
            return .{ .text = std.mem.trim(u8, part.text[at + 1 ..], " \t"), .line = part.line };
        }
    }
    fatal(body.line, "missing nested attribute '" ++ name ++ "'");
}
fn listLen(comptime s: Segment) usize {
    var it = listParts(s);
    var n: usize = 0;
    while (it.next() != null) n += 1;
    return n;
}
fn integers(comptime s: Segment, comptime T: type) [listLen(s)]T {
    var out: [listLen(s)]T = undefined;
    var it = listParts(s);
    var i: usize = 0;
    while (it.next()) |x| : (i += 1) out[i] = int(T, x);
    return out;
}
fn references(comptime s: Segment, comptime v: []const def.Value) [listLen(s)]def.Value {
    var out: [listLen(s)]def.Value = undefined;
    var it = listParts(s);
    var i: usize = 0;
    while (it.next()) |x| : (i += 1) out[i] = reference(x, v);
    return out;
}
fn referenceIndices(comptime s: Segment, comptime limit: usize) [listLen(s)]usize {
    var out: [listLen(s)]usize = undefined;
    var it = listParts(s);
    var i: usize = 0;
    while (it.next()) |x| : (i += 1) {
        if (x.text.len < 2 or x.text[0] != '%') fatal(x.line, "expected reference");
        out[i] = std.fmt.parseInt(usize, x.text[1..], 10) catch fatal(x.line, "invalid reference");
        if (out[i] >= limit) fatal(x.line, "undefined reference");
    }
    return out;
}
fn reference(comptime s: Segment, comptime v: []const def.Value) def.Value {
    const ids = referenceIndices(.{ .text = "[" ++ s.text ++ "]", .line = s.line }, v.len);
    return v[ids[0]];
}
fn int(comptime T: type, comptime s: Segment) T {
    return std.fmt.parseInt(T, s.text, 10) catch fatal(s.line, "invalid integer");
}
fn number(comptime T: type, comptime s: Segment) T {
    return std.fmt.parseFloat(T, s.text) catch fatal(s.line, "invalid number");
}
fn boolean(comptime s: Segment) bool {
    if (std.mem.eql(u8, s.text, "true")) return true;
    if (std.mem.eql(u8, s.text, "false")) return false;
    fatal(s.line, "invalid boolean");
}
fn parseAtom(comptime s: Segment) ir.Atom {
    return std.meta.stringToEnum(ir.Atom, s.text) orelse fatal(s.line, "unknown atom");
}
fn parseDtype(comptime s: Segment) Dtype {
    return switch (parseAtom(s)) {
        .f32 => .f32,
        .f16 => .f16,
        .i8 => .i8,
        .bool => .bool,
        else => fatal(s.line, "invalid dtype"),
    };
}
fn isNull(comptime s: Segment) bool {
    return std.mem.eql(u8, s.text, "null");
}
fn topEqual(comptime text: []const u8) ?usize {
    var sq: usize = 0;
    var ro: usize = 0;
    var q = false;
    var esc = false;
    for (text, 0..) |ch, i| {
        if (q) {
            if (esc) esc = false else if (ch == '\\') esc = true else if (ch == '"') q = false;
            continue;
        }
        switch (ch) {
            '"' => q = true,
            '[' => sq += 1,
            ']' => sq -= 1,
            '(' => ro += 1,
            ')' => ro -= 1,
            '=' => if (sq == 0 and ro == 0) return i,
            else => {},
        }
    }
    return null;
}

fn validateHeader(comptime text: []const u8) void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    const expected = std.fmt.comptimePrint("zgir {d}", .{ir.format_version});
    if (!std.mem.eql(u8, std.mem.trim(u8, lines.next() orelse "", " \t\r"), expected)) fatal(1, "unsupported .zgir header");
}
fn afterEqual(comptime line: []const u8) []const u8 {
    const at = std.mem.indexOfScalar(u8, line, '=') orelse fatal(1, "expected '='");
    return std.mem.trim(u8, line[at + 1 ..], " \t");
}
fn nodeCount(comptime text: []const u8) usize {
    var n: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len != 0 and line[0] != '#' and !std.mem.startsWith(u8, line, "outputs")) n += 1;
    }
    return n;
}
fn sourceCount(comptime text: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#' or std.mem.startsWith(u8, line, "outputs")) continue;
        const c = parseCall(line, i);
        i += 1;
        if (isSource(c.operation)) n += 1;
    }
    return n;
}
fn sourceNames(comptime text: []const u8) [sourceCount(text)][:0]const u8 {
    var out: [sourceCount(text)][:0]const u8 = undefined;
    var i: usize = 0;
    var si: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#' or std.mem.startsWith(u8, line, "outputs")) continue;
        const c = parseCall(line, i);
        i += 1;
        if (!isSource(c.operation)) continue;
        const name = string(c.attr("name"));
        out[si] = &name;
        si += 1;
    }
    return out;
}
fn boundCount(comptime text: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#' or std.mem.startsWith(u8, line, "outputs")) continue;
        const c = parseCall(line, i);
        i += 1;
        if (isSource(c.operation) and parseAtom(c.attr("binding")) == .bound) n += 1;
    }
    return n;
}
fn isSource(op: ir.Operation) bool {
    return op == .input or op == .parameter or op == .constant;
}
fn stringLen(comptime s: Segment) usize {
    if (s.text.len < 2 or s.text[0] != '"' or s.text[s.text.len - 1] != '"') fatal(s.line, "expected string");
    var n: usize = 0;
    var i: usize = 1;
    while (i < s.text.len - 1) : (n += 1) {
        if (s.text[i] == '\\') {
            i += 1;
            if (i >= s.text.len - 1) fatal(s.line, "invalid string escape");
            if (s.text[i] == 'u') {
                i += 1;
                const decoded = escapedCodepoint(s, &i);
                n += (std.unicode.utf8CodepointSequenceLength(decoded) catch fatal(s.line, "invalid unicode scalar")) - 1;
                continue;
            }
        }
        i += 1;
    }
    return n;
}
fn string(comptime s: Segment) [stringLen(s):0]u8 {
    var out: [stringLen(s):0]u8 = undefined;
    var i: usize = 1;
    var j: usize = 0;
    while (i < s.text.len - 1) : (j += 1) {
        var ch = s.text[i];
        i += 1;
        if (ch == '\\') {
            ch = s.text[i];
            i += 1;
            if (ch == 'u') {
                const decoded = escapedCodepoint(s, &i);
                j += (std.unicode.utf8Encode(decoded, out[j..]) catch fatal(s.line, "invalid unicode scalar")) - 1;
                continue;
            }
            ch = switch (ch) {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                'b' => 8,
                'f' => 12,
                else => ch,
            };
        }
        out[j] = ch;
    }
    out[out.len] = 0;
    return out;
}
fn escapedCodepoint(comptime s: Segment, comptime index: *usize) u21 {
    if (index.* + 4 > s.text.len - 1) fatal(s.line, "invalid unicode escape");
    const first = std.fmt.parseInt(u16, s.text[index.*..][0..4], 16) catch fatal(s.line, "invalid unicode escape");
    index.* += 4;
    if (first < 0xd800 or first > 0xdbff) {
        if (first >= 0xdc00 and first <= 0xdfff) fatal(s.line, "unpaired unicode surrogate");
        return first;
    }
    if (index.* + 6 > s.text.len - 1 or !std.mem.eql(u8, s.text[index.*..][0..2], "\\u")) fatal(s.line, "unpaired unicode surrogate");
    const second = std.fmt.parseInt(u16, s.text[index.* + 2 ..][0..4], 16) catch fatal(s.line, "invalid unicode escape");
    if (second < 0xdc00 or second > 0xdfff) fatal(s.line, "unpaired unicode surrogate");
    index.* += 6;
    return 0x10000 + (@as(u21, first - 0xd800) << 10) + (second - 0xdc00);
}
fn fatal(comptime line: usize, comptime msg: []const u8) noreturn {
    @compileError(std.fmt.comptimePrint("invalid .zgir at line {d}: {s}", .{ line, msg }));
}
