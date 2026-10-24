const std = @import("std");
const builtin = @import("builtin");

/// Loads a compiled executable's C ABI from a native shared library.
pub const Library = struct {
    inner: if (builtin.os.tag == .windows) WindowsLibrary else std.DynLib,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Library {
        if (comptime builtin.os.tag == .windows) {
            return .{ .inner = try WindowsLibrary.open(allocator, path) };
        } else {
            return .{ .inner = try std.DynLib.open(path) };
        }
    }

    pub fn close(self: *Library) void {
        self.inner.close();
    }

    pub fn lookup(self: *Library, comptime T: type, name: [:0]const u8) ?T {
        return self.inner.lookup(T, name);
    }
};

const WindowsLibrary = struct {
    const windows = std.os.windows;
    const search_dll_load_dir: windows.DWORD = 0x00000100;
    const search_default_dirs: windows.DWORD = 0x00001000;

    handle: windows.HMODULE,

    extern "kernel32" fn LoadLibraryExW(
        path: windows.LPCWSTR,
        reserved: ?windows.HANDLE,
        flags: windows.DWORD,
    ) callconv(.winapi) ?windows.HMODULE;

    extern "kernel32" fn GetProcAddress(
        module: windows.HMODULE,
        name: [*:0]const u8,
    ) callconv(.winapi) ?windows.FARPROC;

    extern "kernel32" fn FreeLibrary(module: windows.HMODULE) callconv(.winapi) windows.BOOL;

    fn open(allocator: std.mem.Allocator, path: []const u8) !WindowsLibrary {
        if (!std.fs.path.isAbsolute(path)) return error.ExpectedAbsoluteLibraryPath;
        const wide_path = try std.unicode.wtf8ToWtf16LeAllocZ(allocator, path);
        defer allocator.free(wide_path);
        const handle = LoadLibraryExW(
            wide_path.ptr,
            null,
            search_dll_load_dir | search_default_dirs,
        ) orelse return switch (windows.GetLastError()) {
            .FILE_NOT_FOUND, .PATH_NOT_FOUND, .MOD_NOT_FOUND => error.FileNotFound,
            .ACCESS_DENIED => error.AccessDenied,
            .BAD_EXE_FORMAT => error.InvalidLibraryFormat,
            else => error.LibraryLoadFailed,
        };
        return .{ .handle = handle };
    }

    fn close(self: *WindowsLibrary) void {
        _ = FreeLibrary(self.handle);
        self.* = undefined;
    }

    fn lookup(self: *WindowsLibrary, comptime T: type, name: [:0]const u8) ?T {
        const symbol = GetProcAddress(self.handle, name.ptr) orelse return null;
        return @ptrCast(symbol);
    }
};
