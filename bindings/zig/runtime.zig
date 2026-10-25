//! Caller-owned Zig runtime for a statically linked ZGC executable artifact.
pub const abi = @import("zgc_abi_c");

pub const Error = error{
    AbiMismatch,
    StorageTooSmall,
    StorageMisaligned,
    InvalidArgument,
    InvalidRuntimeStorage,
    InvalidSource,
    InvalidOutput,
    SizeMismatch,
    AlignmentMismatch,
    IncompatibleBinding,
    MissingBinding,
    BufferTooSmall,
    OutOfMemory,
    UnexpectedStatus,
};

pub const Requirements = struct {
    size: usize,
    alignment: usize,
    mutable_bytes: usize,
};

pub fn requirements() Error!Requirements {
    if (abi.zgc_abi_version() != abi.ZGC_ABI_VERSION) return error.AbiMismatch;
    return .{
        .size = abi.zgc_runtime_size(),
        .alignment = abi.zgc_runtime_alignment(),
        .mutable_bytes = abi.zgc_runtime_mutable_bytes(),
    };
}

/// The caller retains `storage` for the lifetime of this runtime.
pub const Runtime = struct {
    storage: []u8,

    pub fn init(storage: []u8) Error!Runtime {
        const needed = try requirements();
        if (storage.len < needed.size) return error.StorageTooSmall;
        if (needed.alignment == 0 or @intFromPtr(storage.ptr) % needed.alignment != 0) {
            return error.StorageMisaligned;
        }
        try check(abi.zgc_runtime_init(@ptrCast(storage.ptr), storage.len));
        return .{ .storage = storage };
    }

    pub fn deinit(self: *Runtime) void {
        abi.zgc_runtime_deinit(@ptrCast(self.storage.ptr));
    }

    pub fn run(self: *Runtime) Error!void {
        try check(abi.zgc_runtime_run(@ptrCast(self.storage.ptr)));
    }

    /// Copy logical row-major bytes into an owned source.
    pub fn copySource(self: *Runtime, key: u32, data: []const u8) Error!void {
        try check(abi.zgc_runtime_copy_source(@ptrCast(self.storage.ptr), key, @ptrCast(data.ptr), data.len));
    }

    /// The caller retains bound source data until it is rebound or the runtime is deinitialized.
    pub fn bindSource(self: *Runtime, key: u32, data: []const u8) Error!void {
        try check(abi.zgc_runtime_bind_source(@ptrCast(self.storage.ptr), key, @ptrCast(data.ptr), data.len));
    }

    /// The caller retains writable output storage until it is rebound or the runtime is deinitialized.
    pub fn bindOutput(self: *Runtime, index: usize, data: []u8) Error!void {
        try check(abi.zgc_runtime_bind_output(@ptrCast(self.storage.ptr), index, @ptrCast(data.ptr), data.len));
    }

    pub fn copyOutput(self: *const Runtime, index: usize, destination: []u8) Error!void {
        try check(abi.zgc_runtime_copy_output(@ptrCast(self.storage.ptr), index, @ptrCast(destination.ptr), destination.len));
    }

    pub fn outputView(self: *const Runtime, index: usize) Error!abi.zgc_tensor_view {
        var view: abi.zgc_tensor_view = undefined;
        try check(abi.zgc_runtime_output_view(@ptrCast(self.storage.ptr), index, &view));
        return view;
    }
};

fn check(status: abi.zgc_status) Error!void {
    switch (status) {
        abi.ZGC_STATUS_OK => {},
        abi.ZGC_STATUS_INVALID_ARGUMENT => return error.InvalidArgument,
        abi.ZGC_STATUS_INVALID_RUNTIME_STORAGE => return error.InvalidRuntimeStorage,
        abi.ZGC_STATUS_INVALID_SOURCE => return error.InvalidSource,
        abi.ZGC_STATUS_INVALID_OUTPUT => return error.InvalidOutput,
        abi.ZGC_STATUS_SIZE_MISMATCH => return error.SizeMismatch,
        abi.ZGC_STATUS_ALIGNMENT_MISMATCH => return error.AlignmentMismatch,
        abi.ZGC_STATUS_INCOMPATIBLE_BINDING => return error.IncompatibleBinding,
        abi.ZGC_STATUS_MISSING_BINDING => return error.MissingBinding,
        abi.ZGC_STATUS_BUFFER_TOO_SMALL => return error.BufferTooSmall,
        abi.ZGC_STATUS_OUT_OF_MEMORY => return error.OutOfMemory,
        else => return error.UnexpectedStatus,
    }
}
