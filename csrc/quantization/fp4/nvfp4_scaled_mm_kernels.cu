/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel Perf config
template <typename T>
struct KernelTraitsSm100;

template <>
struct KernelTraitsSm100<float> {
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

template <>
struct KernelTraitsSm100<cutlass::half_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

template <>
struct KernelTraitsSm100<cutlass::bfloat16_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

template <typename T>
struct Fp4GemmSm100 {
  // A matrix configuration
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = T;
  using ElementC = T;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraitsSm100<T>::MmaTileShape;
  using ClusterShape = typename KernelTraitsSm100<T>::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraitsSm100<T>::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

template <typename T>
typename T::Gemm::Arguments args_from_options_sm100(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf, at::Tensor const& alpha,
    int64_t M, int64_t N, int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm100BlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  return arguments;
}

template <typename T>
void runGemmSm100(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
                  at::Tensor const& A_sf, at::Tensor const& B_sf,
                  at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
                  cudaStream_t stream) {
  typename Fp4GemmSm100<T>::Gemm gemm;

  auto arguments = args_from_options_sm100<Fp4GemmSm100<T>>(D, A, B, A_sf, B_sf,
                                                            alpha, m, n, k);

  size_t workspace_size = Fp4GemmSm100<T>::Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}
#else
template <typename T>
void runGemmSm100(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
                  at::Tensor const& A_sf, at::Tensor const& B_sf,
                  at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
                  cudaStream_t stream) {
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
}
#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
// ** not using template specialization, some unexpected issues here
struct Fp4GemmSm120Float16 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;

  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ThreadBlockShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

struct Fp4GemmSm120Bfloat16 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;

  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ThreadBlockShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

struct Fp4GemmSm120Float32 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementC = float;
  using ElementD = float;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;

  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ThreadBlockShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

template <typename T>
auto make_args(void const* A, void const* B, void const* C, void* D,
               void const* SFA, void const* SFB, void const* alpha, int M,
               int N, int K) {
  using namespace cute;

  typename T::StrideA stride_A;
  typename T::LayoutA layout_A;
  typename T::LayoutSFA layout_SFA;
  typename T::StrideB stride_B;
  typename T::LayoutB layout_B;
  typename T::LayoutSFB layout_SFB;
  typename T::StrideC stride_C;
  typename T::LayoutC layout_C;
  typename T::StrideD stride_D;
  typename T::LayoutD layout_D;

  stride_A = cutlass::make_cute_packed_stride(typename T::StrideA{}, {M, K, 1});
  stride_B = cutlass::make_cute_packed_stride(typename T::StrideB{}, {N, K, 1});
  stride_C = cutlass::make_cute_packed_stride(typename T::StrideC{}, {M, N, 1});
  stride_D = cutlass::make_cute_packed_stride(typename T::StrideD{}, {M, N, 1});

  layout_A = make_layout(make_shape(M, K, 1), stride_A);
  layout_B = make_layout(make_shape(N, K, 1), stride_B);
  layout_C = make_layout(make_shape(M, N, 1), stride_C);
  layout_D = make_layout(make_shape(M, N, 1), stride_D);
  layout_SFA = T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
      tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  layout_SFB = T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
      tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          // Mainloop arguments
          static_cast<typename T::ElementA::DataType const*>(A),
          stride_A,
          static_cast<typename T::ElementB::DataType const*>(B),
          stride_B,
          static_cast<typename T::ElementA::ScaleFactorType const*>(SFA),
          layout_SFA,
          static_cast<typename T::ElementB::ScaleFactorType const*>(SFB),
          layout_SFB,
      },
      {{},
       static_cast<typename T::ElementC const*>(C),
       stride_C,
       static_cast<typename T::ElementD*>(D),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<float const*>(alpha);
  return arguments;
}

template <typename T>
void runGemmSm120(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
                  at::Tensor const& A_sf, at::Tensor const& B_sf,
                  at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
                  cudaStream_t stream) {
  using Gemm = typename T::Gemm;
  Gemm gemm;

  auto arguments =
      make_args<T>(A.data_ptr(), B.data_ptr(), A_sf.data_ptr(), D.data_ptr(),
                   A_sf.data_ptr(), B_sf.data_ptr(), alpha.data_ptr(), m, n, k);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}
#else
template <typename T>
void runGemmSm120(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
                  at::Tensor const& A_sf, at::Tensor const& B_sf,
                  at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
                  cudaStream_t stream) {
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
}
#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha) {
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");

  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1],
              "a and b shapes cannot be multiplied (", A.sizes()[0], "x",
              A.sizes()[1], " and ", B.sizes()[0], "x", B.sizes()[1], ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment,
              ", but got a shape: (", A.sizes()[0], "x", A.sizes()[1],
              "), k: ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b shape: (", B.sizes()[0], "x", B.sizes()[1], ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ", B_sf.sizes()[0],
              "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

  if (prop->major == 10) {
    if (out_dtype == at::ScalarType::Half) {
      runGemmSm100<cutlass::half_t>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                    stream);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      runGemmSm100<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                        stream);
    } else if (out_dtype == at::ScalarType::Float) {
      runGemmSm100<float>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else {
      TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm");
    }
  } else if (prop->major == 12) {
    if (out_dtype == at::ScalarType::Half) {
      runGemmSm120<Fp4GemmSm120Float16>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                        stream);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      runGemmSm120<Fp4GemmSm120Bfloat16>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                         stream);
    } else if (out_dtype == at::ScalarType::Float) {
      runGemmSm120<Fp4GemmSm120Float32>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                        stream);
    } else {
      TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm");
    }
  } else {
    TORCH_CHECK(false, "Unsupported GPU architecture for nvfp4 mm");
  }
}
