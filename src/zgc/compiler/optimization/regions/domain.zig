/// Logical iteration space shared by map-family regions. The shape is fixed
/// before executable search; kernels receive only a concrete traversal.
pub const Domain = struct {
    shape: []const usize,

    pub fn elementCount(comptime domain: Domain) usize {
        var count: usize = 1;
        for (domain.shape) |extent| count *= extent;
        return count;
    }
};
