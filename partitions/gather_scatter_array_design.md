# Gather/Scatter Array Indexing Design Document

## Overview

This document specifies a gather/scatter index building system for inter-partition data communication in distributed mesh computations. The system operates in two distinct phases:
1. **Build Phase**: Uses mesh connectivity to compute pick/place indices
2. **Runtime Phase**: Uses only the computed indices for data transfer

## Core Concepts

### Entities
In the context of this design, an **entity** refers to a geometric component of mesh elements that requires data exchange:
- **Faces**: 2D surfaces bounding 3D elements (e.g., 4 triangular faces per tetrahedron, 6 quadrilateral faces per hexahedron)
- **Edges**: 1D segments bounding 2D elements or connecting vertices in 3D elements
- **Vertices**: 0D points at element corners
- **Elements**: The volumetric cells themselves (when ghost elements are needed)

Each entity type forms its own communication pattern and requires separate gather/scatter arrays.

### Partitions and Arrays
- Domain divided into `Npart` partitions
- Each partition has arrays of entity data organized by element
- Arrays use local element numbering (0 to K[part]-1)
- Fixed stride per array (all entities of same type have same point count)

### Entity-Level Indexing
- **Pick indices**: Point to start of entities to gather from source partition's array
- **Place indices**: Point to start of entities to scatter into destination partition's array
- One index per entity, not per point (e.g., one index per face, not per face point)

### Complete Communication Pattern
Each partition maintains **NumPartitions** sets of pick/place indices:
- **Local connections**: Pick indices for partition i → partition i (same partition)
- **Remote connections**: Pick indices for partition j → partition i (different partitions)

This unified approach treats all connections identically, whether within the same partition or across partition boundaries.

## Build Phase

### Inputs

```go
// Mesh connectivity (global element IDs)
type ElementConnectivity struct {
    EToE   [][]int  // [elem][entity] → neighbor element
    EToF   [][]int  // [elem][entity] → neighbor's entity
    BCType [][]int  // [elem][entity] → BC type (-1 for interior)
}

// Partitioning information
type PartitionInfo struct {
    EToP           []int   // [globalElem] → partition
    PartitionElems [][]int // [partition] → list of global elements
    K              []int   // [partition] → number of elements
    NumPartitions  int     // Total number of partitions
}

// Array specification
type ArraySpec struct {
    Name               string
    EntityType         string // "face", "edge", "vertex", "element"
    EntitiesPerElement int    // e.g., 6 faces per hex, 12 edges per hex
    PointsPerEntity    int    // e.g., 16 points per P3 face
}
```

### Build Algorithm

The builder iterates through each partition's elements in local order, determining communication needs for ALL connections (local and remote):

```go
func BuildIndices(
    conn *ElementConnectivity,
    partInfo *PartitionInfo,
    arraySpec *ArraySpec,
    partitionID int,
) *GatherScatterIndices {
    
    // Get this partition's elements
    globalElems := partInfo.PartitionElems[partitionID]
    numLocalElems := len(globalElems)
    
    // Storage for indices by source partition (including self)
    pickBySource := make(map[int][]int32)
    placeBySource := make(map[int][]int32)
    
    // Initialize map for all partitions including self
    for p := 0; p < partInfo.NumPartitions; p++ {
        pickBySource[p] = []int32{}
        placeBySource[p] = []int32{}
    }
    
    // Iterate in local element order
    for localElem := 0; localElem < numLocalElems; localElem++ {
        globalElem := globalElems[localElem]
        
        for entity := 0; entity < arraySpec.EntitiesPerElement; entity++ {
            neighbor := conn.EToE[globalElem][entity]
            
            // Skip boundaries
            if neighbor == globalElem {
                continue // Boundary entity
            }
            
            // Determine source partition (could be same as current)
            sourcePart := partInfo.EToP[neighbor]
            remoteEntity := conn.EToF[globalElem][entity]
            
            // Find neighbor's local position in source partition
            remoteLocalElem := findLocalIndex(neighbor, partInfo.PartitionElems[sourcePart])
            
            // Compute array locations
            pickLoc := computeEntityLocation(remoteLocalElem, remoteEntity, arraySpec)
            placeLoc := computeEntityLocation(localElem, entity, arraySpec)
            
            // Add to appropriate source partition's lists
            pickBySource[sourcePart] = append(pickBySource[sourcePart], int32(pickLoc))
            placeBySource[sourcePart] = append(placeBySource[sourcePart], int32(placeLoc))
        }
    }
    
    // Concatenate into final arrays with all NumPartitions entries
    return concatenateBySource(pickBySource, placeBySource, partInfo.NumPartitions)
}

func computeEntityLocation(elemID, entityID int, spec *ArraySpec) int {
    return elemID * spec.EntitiesPerElement * spec.PointsPerEntity + 
           entityID * spec.PointsPerEntity
}

func concatenateBySource(pickBySource, placeBySource map[int][]int32, numParts int) *GatherScatterIndices {
    result := &GatherScatterIndices{
        PickOffsets:  make([]int32, numParts+1),
        PlaceOffsets: make([]int32, numParts+1),
    }
    
    // Build offsets and concatenate indices for all partitions
    for p := 0; p < numParts; p++ {
        result.PickOffsets[p+1] = result.PickOffsets[p] + int32(len(pickBySource[p]))
        result.PlaceOffsets[p+1] = result.PlaceOffsets[p] + int32(len(placeBySource[p]))
        
        result.PickIndices = append(result.PickIndices, pickBySource[p]...)
        result.PlaceIndices = append(result.PlaceIndices, placeBySource[p]...)
    }
    
    return result
}
```

### Output Structure

```go
type GatherScatterIndices struct {
    // Entity-level indices (point to entity starts)
    PickIndices  []int32  // All pick indices concatenated
    PickOffsets  []int32  // [Npart+1] Start position per source partition
    PlaceIndices []int32  // All place indices concatenated  
    PlaceOffsets []int32  // [Npart+1] Start position per source partition
    
    // Metadata
    PointsPerEntity int
    PartitionID     int
    EntityType      string
}
```

### Index Array Properties

For a partition with K elements and F entities per element:
- Total entities in partition: K × F
- Sum of all pick index lengths: Equals total entities (each entity needs data from somewhere)
- Pick indices for source partition p: Elements between `PickOffsets[p]` and `PickOffsets[p+1]`
- Local connections: When p == partitionID
- Remote connections: When p != partitionID

## Runtime Phase

At runtime, only the indices are used. The gather/scatter operations treat all source partitions uniformly.

### Gather Operation (Pack Send Buffers)

```go
// Pack data for ALL destination partitions (including self)
for destPart := 0; destPart < Npart; destPart++ {
    start := indices.PickOffsets[destPart]
    end := indices.PickOffsets[destPart+1]
    
    for i := start; i < end; i++ {
        entityStart := indices.PickIndices[i]
        // Copy entity data from local array to send buffer
        for p := 0; p < pointsPerEntity; p++ {
            sendBuffer[destPart][...] = localArray[entityStart + p]
        }
    }
}
```

### Communication Step

```go
// Local exchange (no MPI needed)
if destPart == myPartition {
    // Direct copy from send buffer to receive buffer
    copyBuffer(sendBuffer[myPartition], recvBuffer[myPartition])
} else {
    // MPI communication for remote partitions
    MPI_Isend(sendBuffer[destPart], ...)
    MPI_Irecv(recvBuffer[sourcePart], ...)
}
```

### Scatter Operation (Unpack Receive Buffers)

```go
// Unpack data from ALL source partitions (including self)
for sourcePart := 0; sourcePart < Npart; sourcePart++ {
    start := indices.PlaceOffsets[sourcePart]
    end := indices.PlaceOffsets[sourcePart+1]
    
    for i := start; i < end; i++ {
        entityStart := indices.PlaceIndices[i]
        // Copy entity data from receive buffer to local array
        for p := 0; p < pointsPerEntity; p++ {
            localArray[entityStart + p] = recvBuffer[sourcePart][...]
        }
    }
}
```

## Example: Face Communication Pattern

Consider partition 0 in a 4-partition mesh where each element has 4 faces:

```
Element 0: 
  - Face 0 → Element 1 (same partition)    // Local connection
  - Face 1 → Element 15 (partition 2)      // Remote connection  
  - Face 2 → Boundary                      // No connection
  - Face 3 → Element 2 (same partition)    // Local connection

Element 1:
  - Face 0 → Element 0 (same partition)    // Local connection
  - Face 1 → Element 20 (partition 1)      // Remote connection
  - Face 2 → Element 3 (same partition)    // Local connection
  - Face 3 → Boundary                      // No connection
```

This generates pick indices organized by source partition:
- Source partition 0 (local): Pick from faces of elements 1, 2, 0, 3, ...
- Source partition 1: Pick from face 1 of element 20, ...
- Source partition 2: Pick from face 3 of element 15, ...
- Source partition 3: (empty if no connections)

Total pick indices = 6 (number of non-boundary faces in partition 0)

## Key Design Features

1. **Unified Local/Remote Treatment**: Same index structure handles both local and remote connections
2. **Complete Connectivity**: NumPartitions buffers per partition capture all communication patterns
3. **Entity-Level Operations**: Reduces index storage by factor of points_per_entity
4. **Fixed Stride Arrays**: Enables efficient memory access patterns and vectorization
5. **Local Indexing**: All indices reference partition-local array positions
6. **Clean Phase Separation**: Build phase uses connectivity; runtime uses only indices

## Performance Considerations

### Memory Access Patterns
- Sequential access through pick/place indices
- Contiguous copying of entity points
- Predictable stride enables prefetching

### Parallelization
- Entity-level granularity suits GPU thread blocks
- No race conditions when different threads handle different entities
- Local and remote operations can overlap

### Cache Efficiency
- Indices are compact (one per entity, not per point)
- Access pattern matches memory layout
- Reused across multiple solution variables

## Implementation Notes

### Index Validation
During the build phase, validate that:
- All pick indices are within valid range [0, K×F×P)
- All place indices are within valid range [0, K×F×P)
- Sum of pick lengths equals K×F (all entities accounted for)
- No duplicate place indices (each location written once)

### Mixed Meshes
For meshes with multiple element types:
- Build separate indices for each element type
- Each type maintains its own fixed stride
- No mixing within arrays ensures predictable performance

### Boundary Handling
Boundary entities (where EToE[elem][entity] == elem):
- Not included in pick/place indices
- Handled separately by boundary condition routines
- Reduces index storage and communication volume