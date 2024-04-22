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
// ------
