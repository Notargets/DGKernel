# DGKernel Memory Management and Partitioning Guide

## Overview

DGKernel provides a unified interface for managing partitioned and non-partitioned computational kernels. This guide explains how memory is allocated, managed, and accessed when using `NewRunner()` and `DefineKernel()`.

## Kernel Modes

### Non-Partitioned Mode
When `NewRunner()` is called with `K = []int{totalElements}` (single element), the kernel operates in non-partitioned mode:
- All arrays are treated as single contiguous blocks
- No partition offsets are calculated
- Memory layout is straightforward

### Partitioned Mode
When `NewRunner()` is called with `K = []int{k1, k2, ..., kn}` (multiple elements), the kernel operates in partitioned mode:
- Arrays must be provided as partitioned data structures
- Each partition i contains K[i] elements
- Partition boundaries are managed automatically

## Data Binding Rules

### Non-Partitioned Kernels

For kernels created with a single K value:

```go
runner := NewRunner(device, Config{K: []int{625}})

// Arrays: use standard slices
data := make([]float64, 625 * Np)
builder.Input("U").Bind(data).CopyTo()

// Matrices: use single matrix
U := mat.NewDense(Np, 625, nil)
builder.Input("U").Bind(U).CopyTo()
```

### Partitioned Kernels

For kernels created with multiple K values:

```go
runner := NewRunner(device, Config{K: []int{100, 150, 200, 175}})

// Arrays: use slice of slices
data := [][]float64{
    make([]float64, 100 * Np),  // Partition 0
    make([]float64, 150 * Np),  // Partition 1
    make([]float64, 200 * Np),  // Partition 2
    make([]float64, 175 * Np),  // Partition 3
}
builder.Input("U").Bind(data).CopyTo()

// Matrices: use slice of matrices
U := []mat.Matrix{
    mat.NewDense(Np, 100, nil),  // Partition 0
    mat.NewDense(Np, 150, nil),  // Partition 1
    mat.NewDense(Np, 200, nil),  // Partition 2
    mat.NewDense(Np, 175, nil),  // Partition 3
}
builder.Input("U").Bind(U).CopyTo()
```

### Global Data (Shared Across Partitions)

Some data is global and shared by all partitions:

```go
// Global matrix - available to all partitions
Dr := mat.NewDense(Np, Np, drData)
builder.Input("Dr").Bind(Dr).ToMatrix().Static()

// Global scalar - available to all partitions
dt := 0.01
builder.Scalar("dt").Bind(dt)
```

## Type System Rules

### Supported Types for Partitioned Data

**Partitioned Arrays:**
- `[][]float32` - Partitioned float32 arrays
- `[][]float64` - Partitioned float64 arrays
- `[][]int32` - Partitioned int32 arrays
- `[][]int64` - Partitioned int64 arrays

**Partitioned Matrices:**
- `[]mat.Matrix` - Partitioned matrices (without `.ToMatrix()`)
- `[]*mat.Dense` - Partitioned dense matrices (without `.ToMatrix()`)

**Validation Requirements:**
- `len(partitionedData) == len(K)`
- `len(partitionedData[i]) == K[i] * valuesPerElement`

### Restrictions

1. **No `.ToMatrix()` on Partitioned Matrices**:
    - `[]mat.Matrix` cannot use `.ToMatrix()`
    - MATMUL macros require global scope
    - Use single `mat.Matrix` for shared matrices

2. **Type Consistency**:
    - All partitions must have the same element type
    - Matrix dimensions must be consistent (same Np across partitions)

## Memory Layout

### Matrix Storage Format

Matrices undergo automatic transpose during copy:
- Go matrices (gonum): Row-major format
- Device storage: Column-major format
- After transpose: Each element's Np values are contiguous

Example for a 3×4 matrix:
```
Go (row-major):    [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
Device (col-major): [1,5,9, 2,6,10, 3,7,11, 4,8,12]
                     ^elem0^ ^elem1^ ^elem2^ ^elem3^
```

### Partitioned Memory Organization

Partitioned arrays are allocated as a single contiguous pool on the device, with each partition's data placed at aligned offsets:

```
Device Memory Pool:
[Partition 0 data][alignment padding][Partition 1 data][padding][Partition 2 data]...
 ^                                    ^                          ^
 offset[0] = 0                        offset[1]                  offset[2]
```

**Offset Calculation:**
- Each partition starts at an aligned boundary
- `offset[i+1] = align(offset[i] + K[i] * valuesPerElement * sizeof(type))`
- Total pool size includes all data plus alignment padding

**Example with 64-byte alignment:**
```
Partition 0: 1000 bytes of data
Partition 1: 1500 bytes of data
Partition 2: 2000 bytes of data

Offsets: [0, 1024, 2560, 4608]  // Each aligned to 64-byte boundary
Total allocated: 4608 bytes (includes 108 bytes of padding)
```

### Generated Access Macros

For partitioned kernels, the system generates:

```c
#define U_PART(part) (U_global + U_offsets[part])
```

Where:
- `U_global` is the base pointer to the pooled memory
- `U_offsets[part]` is the aligned offset for partition `part`

## Memory Pool Management

### Allocation Strategy

1. **Partitioned Data**:
    - Single pooled allocation per array on device
    - Each partition's data placed at aligned offsets within the pool
    - Pool size = sum of partition sizes + alignment padding
    - Offset array tracks each partition's starting position

2. **Global Data**:
    - Single allocation shared by all partitions
    - No partitioning or offset calculations needed

3. **Temporary Arrays**:
    - Single pooled allocation sized for maximum partition
    - Each partition accesses the same pool with appropriate bounds

### Alignment Options

```go
builder.Input("U").Bind(data).Align(CacheLineAlign)  // 64-byte alignment
builder.Input("V").Bind(data).Align(WarpAlign)       // 128-byte alignment
builder.Input("W").Bind(data).Align(PageAlign)       // 4096-byte alignment
```

Alignment is applied to each partition's allocation individually.

## Parameter Builder Methods

### Core Methods

#### `.Bind(data)`
Associates host data with the parameter. The data remains on the host until explicitly copied.
- For non-partitioned kernels: accepts slices or matrices
- For partitioned kernels: accepts slice of slices or slice of matrices
- Required for all parameters except `.Temp()`

#### `.CopyTo()`
Copies data from host to device before kernel execution.
- Performs automatic transpose for matrices (row-major to column-major)
- For partitioned data, copies each partition to its aligned offset
- Often used with `Input()` parameters

#### `.CopyBack()`
Copies data from device to host after kernel execution.
- Reverses matrix transpose (column-major back to row-major)
- Updates the original bound host data
- Typically used with `Output()` parameters

#### `.Copy()`
Shorthand for `.CopyTo().CopyBack()` - bidirectional copy.
- Useful for in-place operations
- Common with `InOut()` parameters

#### `.NoCopy()`
Explicitly disables any data movement.
- Data must already be on device
- Useful for performance when reusing device data across kernels

### Matrix-Specific Methods

#### `.ToMatrix()`
Marks the parameter as a matrix and generates MATMUL macros.
- **Only valid for global (non-partitioned) matrices**
- Generates `MATMUL_<name>` and `MATMUL_ADD_<name>` macros
- Enables efficient matrix-vector and matrix-matrix operations
- Cannot be used with `[]mat.Matrix` (partitioned matrices)

Example generated macros for a matrix "Dr":
```c
// Standard multiply: OUT = Dr × IN
#define MATMUL_Dr(IN, OUT, K_VAL) \
    for (int i = 0; i < Np; ++i) { \
        for (int elem = 0; elem < KpartMax; ++elem; @inner) { \
            if (elem < K_VAL) { \
                real_t sum = 0.0; \
                for (int j = 0; j < Np; ++j) { \
                    sum += Dr[i][j] * (IN)[elem * Np + j]; \
                } \
                (OUT)[elem * Np + i] = sum; \
            } \
        } \
    }

// Accumulating multiply: OUT += Dr × IN  
#define MATMUL_ADD_Dr(IN, OUT, K_VAL) \
    // Similar but with += instead of =
```

#### `.Static()`
Embeds the matrix as a compile-time constant in the kernel.
- **Only valid with `.ToMatrix()`**
- Matrix data becomes part of kernel source code
- Excellent performance for small matrices (typically < 50×50)
- No device memory allocation needed
- Matrix values cannot be changed at runtime

Example:
```go
// This embeds Dr as a static const array in the kernel
Dr := mat.NewDense(Np, Np, drData)
builder.Input("Dr").Bind(Dr).ToMatrix().Static()
```

### Type Conversion Methods

#### `.Convert(toType)`
Performs type conversion during copy operations.
- Useful when host uses float64 but device uses float32
- Conversion happens during `.CopyTo()` or `.CopyBack()`
- Saves memory bandwidth on devices

Example:
```go
// Host data in float64 for precision
hostData := make([]float64, 1000)
// Convert to float32 on device to save memory
builder.Input("data").Bind(hostData).CopyTo().Convert(Float32)
```

### Other Methods

#### `.Type(dataType)`
Explicitly sets the data type.
- Mainly used with `.Temp()` arrays
- Required when type cannot be inferred from binding

#### `.Size(elements)`
Sets the size in number of elements.
- Used with `.Temp()` arrays
- For partitioned kernels, specifies per-partition size

#### `.Align(alignment)`
Sets memory alignment requirements.
- Applied to pooled memory allocation
- Options: `NoAlignment`, `CacheLineAlign`, `WarpAlign`, `PageAlign`

## Complete Example

```go
// Define a 4-partition kernel
K := []int{100, 150, 200, 175}
runner := NewRunner(device, Config{
    K:         K,
    FloatType: Float64,
})

// Prepare partitioned input data
U := make([]mat.Matrix, len(K))
V := make([]mat.Matrix, len(K))
for i := range K {
    U[i] = mat.NewDense(Np, K[i], nil)
    V[i] = mat.NewDense(Np, K[i], nil)
    // Initialize with data...
}

// Prepare output arrays
W := make([][]float64, len(K))
for i := range K {
    W[i] = make([]float64, K[i] * Np)
}

// Global differentiation matrix (small, suitable for static embedding)
Dr := mat.NewDense(Np, Np, drData)

// Global mass matrix (larger, keep on device)
Mass := mat.NewDense(Np, Np, massData)

// Define kernel parameters
err := runner.DefineKernel("compute",
    // Partitioned inputs - copied to device before kernel
    builder.Input("U").Bind(U).CopyTo(),
    builder.Input("V").Bind(V).CopyTo(),
    
    // Partitioned output - copied back after kernel
    builder.Output("W").Bind(W).CopyBack(),
    
    // Global matrix with static embedding and MATMUL macros
    builder.Input("Dr").Bind(Dr).ToMatrix().Static(),
    
    // Global matrix on device with MATMUL macros  
    builder.Input("Mass").Bind(Mass).ToMatrix().CopyTo(),
    
    // Scalar parameter (always global)
    builder.Scalar("alpha").Bind(2.5),
    
    // Temporary workspace (allocated per partition)
    builder.Temp("workspace").Type(Float64).Size(Np * Np),
)

// Kernel can now use:
// - U_PART(part), V_PART(part), W_PART(part) for partitioned data
// - MATMUL_Dr(IN, OUT, K[part]) for differentiation
// - MATMUL_Mass(IN, OUT, K[part]) for mass matrix multiply
// - alpha as a scalar constant
// - workspace_PART(part) for temporary storage
```

## Best Practices

1. **Data Organization**:
    - Pre-allocate partitioned arrays before binding
    - Ensure partition sizes match K values exactly
    - Initialize data after allocation but before binding

2. **Performance**:
    - Balance partition sizes for even load distribution
    - Use appropriate alignment for target architecture
    - Minimize data movement with `.NoCopy()` when possible

3. **Memory Efficiency**:
    - Reuse allocations across kernel calls when possible
    - Use `.Temp()` for kernel-local workspace
    - Free runners when done to release all pooled memory

## Debugging Guide

### Validation Checks

The system performs these checks:

1. **Partition Count**: Number of partitions in data must match len(K)
2. **Partition Sizes**: Each partition's size must match K[i] * elementsPerValue
3. **Type Consistency**: All partitions must have matching types
4. **Matrix Dimensions**: All matrix partitions must have same row count (Np)

### Common Error Messages

- `"partition count mismatch: expected 4, got 3"` - Data has wrong number of partitions
- `"partition 2 size mismatch: expected 4000, got 3000"` - Partition has wrong number of elements
- `"cannot use .ToMatrix() with partitioned matrices"` - Trying to generate per-partition MATMUL macros
- `"type mismatch in partition 1: expected float64, got float32"` - Inconsistent types across partitions