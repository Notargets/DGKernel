# 4 - Partitioning - Parallelism and Mixed Meshes

## Overview

Partitioning is the bridge between the mathematical formulation of DG methods and their efficient parallel execution. A partition groups elements that compute together, whether on the same CPU core, GPU block, or MPI rank. This chapter explains how DGKernel's partitioning system enables scalable parallelism while supporting mixed element types and heterogeneous architectures.

## The Partition Concept

### Definition

A **Partition** is a collection of elements that:
- Execute together as a computational unit
- Share the same memory space during kernel execution
- Communicate with other partitions through well-defined interfaces

In DGKernel, partitions map directly to OCCA's parallel execution model:
- **@outer loops** iterate over partitions (coarse-grained parallelism)
- **@inner loops** iterate over elements within partitions (fine-grained parallelism)

### Partition Structure

```go
type Partition struct {
    ID          int           // Unique partition identifier
    Elements    []int         // Global element indices in this partition
    NumElements int           // Actual number of elements
    MaxElements int           // Padded size for OCCA @inner uniformity
    
    // Element type information for mixed meshes
    ElementTypes []ElementType
    TypeOffsets  []int        // Starting index for each element type
}

type PartitionLayout struct {
    Partitions   []Partition
    KpartMax     int         // max(NumElements) across all partitions
    TotalElements int        // Sum of all partition sizes
}
```

### OCCA Constraint: Uniform Inner Loops

OCCA requires all @inner loops to have the same iteration count. This means:
1. All partitions must be padded to `KpartMax` elements
2. Kernels use bounds checking to skip padded iterations
3. Some computation is wasted but parallelism is maintained

Example kernel structure:
```c
@kernel void computeKernel(const int_t* K, ...) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int elem = 0; elem < KpartMax; ++elem; @inner) {
            if (elem < K[part]) {
                // Actual computation
            }
        }
    }
}
```

## Memory Layout for Partitioned Data

### Partition-Major Storage

DGKernel uses partition-major ordering for efficient memory access:

```
Global Array Layout:
[Partition 0 Data][Partition 1 Data]...[Partition N-1 Data]

Each Partition's Data:
[Element 0][Element 1]...[Element K-1][Padding]
```

This layout ensures:
- Contiguous memory access within partitions
- Efficient cache utilization
- Natural alignment with OCCA's execution model

### Offset Management

Each partitioned array maintains offsets for direct partition access:

```go
type PartitionedArray struct {
    GlobalData []float64    // Contiguous storage for all partitions
    Offsets    []int        // Starting index for each partition
    Stride     int          // Elements per node (e.g., Np)
}
```

## Inter-Partition Communication

### The PartitionBuffer System

When elements in different partitions need to exchange information (for flux computations, ghost cell updates, etc.), DGKernel uses a unified buffer system:

```go
type PartitionBuffer struct {
    // Contiguous buffers for both local and remote communication
    SendBuffer []float64
    RecvBuffer []float64
    
    // Index mappings for scatter/gather operations
    ScatterMap []PartitionMapping  // Local data → SendBuffer
    GatherMap  []PartitionMapping  // RecvBuffer → Local data
    
    // Remote communication metadata (for MPI)
    RemotePartitions []RemotePartition
}

type PartitionMapping struct {
    PartitionID    int
    LocalIndices   []int    // Indices within the partition
    BufferIndices  []int    // Positions in send/recv buffer
}

type RemotePartition struct {
    Rank        int         // MPI rank (-1 for local)
    SendOffset  int         // Starting position in SendBuffer
    SendCount   int         // Number of values to send
    RecvOffset  int         // Starting position in RecvBuffer
    RecvCount   int         // Number of values to receive
}
```

### Unified Local/Remote Communication

The key innovation is using the same buffers for both local and remote communication:

1. **Local partitions** (same host): OCCA kernels scatter/gather directly
2. **Remote partitions** (different hosts): MPI uses the packed buffers

This unification eliminates special cases and memory copies.

### Communication Flow

#### Step 1: Scatter to Send Buffer
```c
@kernel void scatterToBuffer(
    const real_t* PartitionData,
    real_t* SendBuffer,
    // ... index arrays ...
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int i = 0; i < MaxScatterPoints; ++i; @inner) {
            if (i < NumScatterPoints[part]) {
                int localIdx = ScatterLocal[part][i];
                int bufferIdx = ScatterBuffer[part][i];
                SendBuffer[bufferIdx] = PartitionData[part][localIdx];
            }
        }
    }
}
```

#### Step 2: Exchange (Local or Remote)
```go
func (pb *PartitionBuffer) Exchange() {
    // For remote partitions, use MPI
    for _, rp := range pb.RemotePartitions {
        if rp.Rank >= 0 {
            MPI_Isend(pb.SendBuffer[rp.SendOffset:], rp.Rank, ...)
            MPI_Irecv(pb.RecvBuffer[rp.RecvOffset:], rp.Rank, ...)
        }
    }
    
    // For local partitions, just copy (or share memory)
    for _, mapping := range pb.LocalMappings {
        copy(pb.RecvBuffer[mapping.RecvRange], 
             pb.SendBuffer[mapping.SendRange])
    }
}
```

#### Step 3: Gather from Receive Buffer
```c
@kernel void gatherFromBuffer(
    real_t* PartitionData,
    const real_t* RecvBuffer,
    // ... index arrays ...
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int i = 0; i < MaxGatherPoints; ++i; @inner) {
            if (i < NumGatherPoints[part]) {
                int bufferIdx = GatherBuffer[part][i];
                int localIdx = GatherLocal[part][i];
                PartitionData[part][localIdx] = RecvBuffer[bufferIdx];
            }
        }
    }
}
```

## Mixed Meshes and Heterogeneous Elements

### Supporting Multiple Element Types

Partitions can contain different element types, enabling:
- Hex/Tet hybrid meshes
- p-adaptive refinement
- Multi-physics with specialized elements

```go
type MixedPartition struct {
    Partition
    
    // Element type grouping
    TypeGroups []ElementGroup
}

type ElementGroup struct {
    ElementType ElementType
    StartIndex  int         // Starting position in partition
    Count       int         // Number of elements of this type
    Np          int         // Nodes per element for this type
}
```

### Kernel Strategies for Mixed Types

#### Strategy 1: Type-Specific Kernels
Execute different kernels for different element types:
```c
@kernel void hexKernel(...) {
    // Process only hex elements
}

@kernel void tetKernel(...) {
    // Process only tet elements
}
```

#### Strategy 2: Branching Within Kernels
Single kernel with type-based branching:
```c
@kernel void mixedKernel(const int_t* ElementTypes, ...) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int elem = 0; elem < KpartMax; ++elem; @inner) {
            if (elem < K[part]) {
                switch (ElementTypes[part][elem]) {
                    case HEX: processHex(...); break;
                    case TET: processTet(...); break;
                }
            }
        }
    }
}
```

## Partitioning Strategies

### Load Balancing

The quality of partitioning significantly impacts performance. Key considerations:

1. **Equal work distribution**: Balance total degrees of freedom, not just element count
2. **Minimize communication**: Keep connected elements together
3. **Respect hardware boundaries**: Align partitions with NUMA domains, GPU capabilities

### Dynamic Partitioning

For adaptive simulations, partitions may need rebalancing:

```go
type DynamicPartitioner interface {
    // Rebalance based on current load
    Rebalance(currentLayout PartitionLayout, 
              workEstimates []float64) PartitionLayout
    
    // Migrate data to new layout
    Migrate(oldLayout, newLayout PartitionLayout,
            data PartitionedArray) PartitionedArray
}
```

## Performance Considerations

### Memory Bandwidth

Partition size affects memory access patterns:
- **Too small**: Poor cache utilization, overhead dominates
- **Too large**: Exceeds cache capacity, more memory traffic
- **Optimal**: Fits in L2/L3 cache (CPU) or shared memory (GPU)

### Communication Overlap

Structure kernels to enable computation/communication overlap:

1. Start boundary scatter
2. Compute interior elements
3. Finish communication
4. Compute boundary elements

### Hardware-Specific Tuning

Different architectures require different partition strategies:
- **CPU**: Larger partitions, focus on cache blocking
- **GPU**: Smaller partitions matching block size
- **Multi-GPU**: Minimize inter-device communication

## Summary

Partitioning in DGKernel provides:
- Clean abstraction for parallel execution
- Unified treatment of local and remote communication
- Support for mixed element types
- Scalability from single-core to distributed systems

The partition concept bridges the gap between the mathematical formulation of DG methods and their efficient implementation on modern parallel hardware. By understanding and properly utilizing partitions, developers can achieve optimal performance across diverse architectures.