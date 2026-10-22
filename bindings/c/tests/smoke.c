#include "../zgc_executable.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "C ABI smoke failed at line %d: %s\n", __LINE__,         \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  CHECK(zgc_abi_version() == ZGC_ABI_VERSION);
  CHECK(zgc_executable_source_count() == 2);
  CHECK(zgc_executable_output_count() == 1);
  zgc_output_binding output_binding;
  CHECK(zgc_executable_get_output_binding(0, &output_binding) == ZGC_STATUS_OK);
  CHECK(output_binding == ZGC_OUTPUT_OWNED);

  zgc_source_descriptor input_source;
  zgc_source_descriptor bias_source;
  zgc_tensor_descriptor output_descriptor;
  CHECK(zgc_executable_get_source_descriptor(0, &input_source) ==
        ZGC_STATUS_OK);
  CHECK(zgc_executable_get_source_descriptor(1, &bias_source) == ZGC_STATUS_OK);
  CHECK(zgc_executable_get_output_descriptor(0, &output_descriptor) ==
        ZGC_STATUS_OK);
  CHECK(strcmp(input_source.name, "input") == 0);
  CHECK(input_source.binding == ZGC_BINDING_BOUND);
  CHECK(input_source.tensor.dtype == ZGC_DTYPE_F32);
  CHECK(input_source.tensor.rank == 1 && input_source.tensor.shape[0] == 4);
  CHECK(strcmp(bias_source.name, "bias") == 0);
  CHECK(bias_source.binding == ZGC_BINDING_OWNED);
  CHECK(output_descriptor.rank == 1 && output_descriptor.shape[0] == 4);

  const size_t size = zgc_runtime_size();
  const size_t alignment = zgc_runtime_alignment();
  CHECK(size > 0 && alignment > 0);
  void *allocation = malloc(size + alignment - 1);
  CHECK(allocation != NULL);
  void *runtime = (void *)(((uintptr_t)allocation + alignment - 1) &
                           ~((uintptr_t)alignment - 1));
  CHECK(zgc_runtime_init(runtime, size) == ZGC_STATUS_OK);

  float input[4] = {-1.0f, 2.0f, -3.0f, 4.0f};
  const float bias[4] = {0.0f, 1.0f, 0.0f, 3.0f};
  float output[4];
  CHECK(zgc_runtime_bind_source(runtime, input_source.key, input,
                                sizeof(input)) == ZGC_STATUS_OK);
  CHECK(zgc_runtime_copy_source(runtime, bias_source.key, bias, sizeof(bias)) ==
        ZGC_STATUS_OK);
  CHECK(zgc_runtime_run(runtime) == ZGC_STATUS_OK);
  CHECK(zgc_runtime_copy_output(runtime, 0, output, sizeof(output)) ==
        ZGC_STATUS_OK);
  CHECK(output[0] == 0.0f && output[1] == 3.0f && output[2] == 0.0f &&
        output[3] == 7.0f);

  input[0] = 10.0f;
  CHECK(zgc_runtime_run(runtime) == ZGC_STATUS_OK);
  CHECK(zgc_runtime_copy_output(runtime, 0, output, sizeof(output)) ==
        ZGC_STATUS_OK);
  CHECK(output[0] == 10.0f);

  zgc_runtime_deinit(runtime);
  free(allocation);
  return 0;
}
