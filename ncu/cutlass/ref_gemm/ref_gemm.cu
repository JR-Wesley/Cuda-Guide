#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 32;
  int columns = 16;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  return 0;
}