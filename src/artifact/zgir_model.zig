const zgc = @import("zgc");
const graph = @import("graph");

pub const Model = zgc.frontend.modelFromZgir(graph.source);
