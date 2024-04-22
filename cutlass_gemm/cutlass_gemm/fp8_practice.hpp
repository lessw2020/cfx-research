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
