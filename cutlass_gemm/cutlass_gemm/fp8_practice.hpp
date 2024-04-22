// practice

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// A matrix
using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128/cutlass::sizeof_bits<ElementA>::value;

// B matrix
using ElementB = cutlass::float_e5m2_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// C matrix
using ElementC = cutlass::float_e4m3_t;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// D matrix
using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AligmentD = AlignmentC;

// Auxiliary matrix
using ElementAux = ElementC;
using LayoutAux = LayoutC;
using ElementAmax = float;
using ElementBias =float;


//Core kernel configurations
using ElementAccumulator = float;                                          // Element type for internal accumulation
using ElementCompute = float;                                              // Element type for epilogue computation
using ArchTag = cutlass::arch::Sm90;                                       // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;                      // Operator class tag
using TileShape = Shape<_64, _128, _128>;                                  // Threadblock-level tile size
using ClusterShape = Shape<_1, _2, _1>;                                    // Shape of the threadblocks in a cluster
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
    LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOperation
    >::CollectiveOp;

using CollectiveMainLoop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedulestatic
    >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int,int, int>, // Indicates ProblemShape
    CollectiveMainLoop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

//Extract information from Gemm kernel.
using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
using ElementScalar = typename EpilogueOutputOp::ElementScalar;
using ElementAmax = typename EpilogueOutputOp::ElementAmax;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using StrideAux = StrideD;

constexpr bool = isDFp8 =
cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

constexpr bool = isAuxFp8 =
cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
StrideAux stride_aux;
uint64_t seed;

cutlass::HostTensor<ElementA, LayoutA > tensor_A;
cutlass::HostTensor<ElementB, LayoutB > tensor_B;
cutlass::HostTensor<ElementC, LayoutC > tensor_C;
cutlass::HostTensor<ElementD, LayoutD > tensor_D;
cutlass::HostTensor<ElementD, LayoutD > tensor_ref_D;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_aux;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_ref_aux;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_aux;


#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// testbed utility
/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

//init block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {
    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;
    int bits_output = cutlass::sizeof_bits<Element>::value;

if (bits_input==1) {
  scope_max = 2;
  scope_min = 0;
}
else if (bits_input <= 8) {
  scope_max = 2;
  scope_min = -2;
}
else if (bits_output ==16) {
  scope_max = 5;
  scope_min = -5;
}
else {
  scope_max = 8;
  scope_min = -8;
}
cutlass::reference::host::TensorFillRandomUniform(view, seed,
    scope_max, scope_min, 0);
return true;

  }

template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;



#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// ------


///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}
