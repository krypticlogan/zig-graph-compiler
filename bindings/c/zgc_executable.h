#ifndef ZGC_EXECUTABLE_H
#define ZGC_EXECUTABLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZGC_ABI_VERSION 2u

typedef enum zgc_status {
    ZGC_STATUS_OK = 0,
    ZGC_STATUS_INVALID_ARGUMENT = 1,
    ZGC_STATUS_INVALID_RUNTIME_STORAGE = 2,
    ZGC_STATUS_INVALID_SOURCE = 3,
    ZGC_STATUS_INVALID_OUTPUT = 4,
    ZGC_STATUS_SIZE_MISMATCH = 5,
    ZGC_STATUS_ALIGNMENT_MISMATCH = 6,
    ZGC_STATUS_INCOMPATIBLE_BINDING = 7,
    ZGC_STATUS_MISSING_BINDING = 8,
    ZGC_STATUS_BUFFER_TOO_SMALL = 9,
    ZGC_STATUS_OUT_OF_MEMORY = 10,
} zgc_status;

typedef enum zgc_inspect_section {
    ZGC_INSPECT_ALL = 0,
    ZGC_INSPECT_SUMMARY = 1,
    ZGC_INSPECT_RAW_GRAPH = 2,
    ZGC_INSPECT_SEMANTIC_GRAPH = 3,
    ZGC_INSPECT_EXECUTABLE = 4,
    ZGC_INSPECT_REPRESENTATIONS = 5,
    ZGC_INSPECT_TREE = 6,
    ZGC_INSPECT_MEMORY_PLAN = 7,
} zgc_inspect_section;

typedef enum zgc_dtype {
    ZGC_DTYPE_F32 = 1,
    ZGC_DTYPE_F16 = 2,
    ZGC_DTYPE_I8 = 3,
    ZGC_DTYPE_BOOL = 4,
    ZGC_DTYPE_I32 = 5,
    ZGC_DTYPE_U8 = 6,
} zgc_dtype;

typedef enum zgc_source_kind {
    ZGC_SOURCE_INPUT = 1,
    ZGC_SOURCE_PARAMETER = 2,
    ZGC_SOURCE_CONSTANT = 3,
    ZGC_SOURCE_STATE = 4,
} zgc_source_kind;

typedef enum zgc_source_binding {
    ZGC_BINDING_OWNED = 1,
    ZGC_BINDING_BOUND = 2,
    ZGC_BINDING_EMBEDDED = 3,
} zgc_source_binding;

typedef enum zgc_output_binding {
    ZGC_OUTPUT_OWNED = 1,
    ZGC_OUTPUT_BOUND = 2,
} zgc_output_binding;

typedef struct zgc_tensor_descriptor {
    zgc_dtype dtype;
    uint32_t rank;
    size_t element_count;
    size_t logical_byte_count;
    size_t offset_elements;
    const size_t *shape;
    const ptrdiff_t *strides;
} zgc_tensor_descriptor;

typedef struct zgc_source_descriptor {
    uint32_t key;
    const char *name;
    zgc_source_kind kind;
    zgc_source_binding binding;
    zgc_tensor_descriptor tensor;
} zgc_source_descriptor;

typedef struct zgc_tensor_view {
    const void *data;
    size_t storage_byte_count;
    zgc_tensor_descriptor tensor;
} zgc_tensor_view;

uint32_t zgc_abi_version(void);
/* Fingerprint of the compiled executable graph, including its selected plan. */
uint64_t zgc_executable_structural_fingerprint(void);
size_t zgc_runtime_size(void);
size_t zgc_runtime_alignment(void);
size_t zgc_runtime_mutable_bytes(void);
/* Exported only by inspectable artifacts. Query required bytes with
   destination=NULL and capacity=0, then copy. */
zgc_status zgc_executable_inspect(
    zgc_inspect_section section,
    char *destination,
    size_t capacity,
    size_t *required
);

zgc_status zgc_runtime_init(void *storage, size_t storage_size);
void zgc_runtime_deinit(void *storage);
zgc_status zgc_runtime_run(void *storage);

size_t zgc_executable_source_count(void);
zgc_status zgc_executable_get_source_descriptor(size_t source_index, zgc_source_descriptor *out);
zgc_status zgc_runtime_copy_source(
    void *runtime,
    uint32_t source_key,
    const void *data,
    size_t byte_count
);
zgc_status zgc_runtime_bind_source(
    void *runtime,
    uint32_t source_key,
    const void *data,
    size_t byte_count
);

size_t zgc_executable_output_count(void);
zgc_status zgc_executable_get_output_descriptor(size_t output_index, zgc_tensor_descriptor *out);
zgc_status zgc_executable_get_output_binding(size_t output_index, zgc_output_binding *out);
zgc_status zgc_runtime_bind_output(
    void *runtime,
    size_t output_index,
    void *data,
    size_t byte_count
);
zgc_status zgc_runtime_output_view(
    const void *runtime,
    size_t output_index,
    zgc_tensor_view *out
);
zgc_status zgc_runtime_copy_output(
    const void *runtime,
    size_t output_index,
    void *destination,
    size_t destination_size
);

#ifdef __cplusplus
}
#endif

#endif
