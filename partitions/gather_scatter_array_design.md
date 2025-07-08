# Gather/Scatter Array Indexing Design Document

## Overview

This document specifies a gather/scatter index building system for inter-partition data communication. The system operates in two distinct phases:
1. **Build Phase**: Uses mesh connectivity to compute pick/place indices
2. **Runtime Phase**: Uses only the computed indices for data transfer

## Core Concepts

### Partitions and Arrays
- Domain divided into `Npart` partitions
- Each partition has arrays of entity data (faces, edges, vertices)
- Arrays use local element numbering (0 to K[part]-1)
- Fixed stride per array (all entities have same point count)

### Entity-Level Indexing
- **Pick indices**: Point to start of entities to gather from sender's array
- **Place indices**: Point to start of entities to scatter into receiver's array
- One index per entity, not per point

## Build Phase

### Inputs

```go
// Mesh connectivity (global element IDs)
type ElementConnectivity struct {
EToE   [][]int  // [elem][face] → neighbor element
EToF   [][]int  // [elem][face] → neighbor's face
BCType [][]int  // [elem][face] → BC type (-1 for interior)
}

// Partitioning information
type PartitionInfo struct {
EToP           []int   // [globalElem] → partition
PartitionElems [][]int // [partition] → list of global elements
K              []int   // [partition] → number of elements (computed during build)
}

// Array specification
type ArraySpec struct {
Name               string
EntitiesPerElement int    // e.g., 6 faces per hex
PointsPerEntity    int    // e.g., 16 points per P3 face
}
```

### Build Algorithm

The builder iterates through each partition's elements in local order, determining communication needs:

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
    
    // Temporary storage for indices by source partition
    pickBySource := make(map[int][]int32)
    placeBySource := make(map[int][]int32)
    
    // Iterate in local element order
    for localElem := 0; localElem < numLocalElems; localElem++ {
        globalElem := globalElems[localElem]
        
        for entity := 0; entity < arraySpec.EntitiesPerElement; entity++ {
            neighbor := conn.EToE[globalElem][entity]
            
            // Skip boundaries and local neighbors
            if neighbor == globalElem {
                continue // Boundary
            }
            if partInfo.EToP[neighbor] == partitionID {
                continue // Local neighbor
            }
            
            // Remote neighbor - need pick/place indices
            remotePart := partInfo.EToP[neighbor]
            remoteEntity := conn.EToF[globalElem][entity]
            
            // Find neighbor's local position in remote partition
            remoteLocalElem := findLocalIndex(neighbor, partInfo.PartitionElems[remotePart])
            
            // Compute array locations
            pickLoc := computeEntityLocation(remoteLocalElem, remoteEntity, arraySpec)
            placeLoc := computeEntityLocation(localElem, entity, arraySpec)
            
            pickBySource[remotePart] = append(pickBySource[remotePart], int32(pickLoc))
            placeBySource[remotePart] = append(placeBySource[remotePart], int32(placeLoc))
        }
    }
    
    // Concatenate into final arrays
    return concatenateBySource(pickBySource, placeBySource, partInfo.NumPartitions)
}

func computeEntityLocation(elemID, entityID int, spec *ArraySpec) int {
    return elemID * spec.EntitiesPerElement * spec.PointsPerEntity + 
           entityID * spec.PointsPerEntity
}

func findLocalIndex(globalElem int, partitionElems []int) int {
    for i, elem := range partitionElems {
        if elem == globalElem {
            return i
        }
    }
    panic("element not found in partition")
}
```

### Output Structure

```go
type GatherScatterIndices struct {
    // Entity-level indices (point to entity starts)
    PickIndices  []int32  // All pick indices concatenated
    PickOffsets  []int32  // [Npart+1] Start position per source
    PlaceIndices []int32  // All place indices concatenated
    PlaceOffsets []int32  // [Npart+1] Start position per source
    
    // Metadata
    PointsPerEntity int
    PartitionID     int
}

// Builder retains partition information
type IndexBuilder struct {
    K []int  // [partition] → number of elements per partition
    // Other builder state...
}
```

## Runtime Phase

At runtime, only the indices are used. The actual gather/scatter operations are implemented by the application (typically on device kernels).

### Example Device Kernel Usage

```c
// Example OCCA kernel for packing send buffer
@kernel void packSendBuffer(const int Npart,
                           const int myPart,
                           const int32* pickIndices,
                           const int32* pickOffsets,
                           const int pointsPerEntity,
                           const real* sourceArray,
                           real* sendBuffer) {
    for (int tgt = 0; tgt < Npart; ++tgt) {
        if (tgt == myPart) continue;
        
        const int32 start = pickOffsets[tgt];
        const int32 end = pickOffsets[tgt+1];
        
        for (int32 e = start; e < end; ++e) {
            const int32 entityStart = pickIndices[e];
            // Copy entity points...
        }
    }
}
```

The application determines how to use the indices based on its specific needs.

## Validation Example

A validation application can verify correctness by:

1. **Using the builder to create indices**
```go
indices := BuildIndices(conn, partInfo, arraySpec, partitionID)
```

2. **Creating test data with known values**
```go
type FaceCoordinate struct {
    ElementID int
    FaceID    int
}

// Use K[partitionID] to size the array correctly
numElems := builder.K[partitionID]
localFaces := make([]FaceCoordinate, numElems * facesPerElem)
for elem := 0; elem < numElems; elem++ {
    for face := 0; face < facesPerElem; face++ {
        idx := elem * facesPerElem + face
        localFaces[idx] = FaceCoordinate{elem, face}
    }
}
```

3. **Executing gather/scatter (validation app implementation)**
```go
// Validation app implements its own pack/unpack for testing
func (v *Validator) PackSendBuffer(indices *GatherScatterIndices, 
                                   localFaces []FaceCoordinate, 
                                   targetPart int) []FaceCoordinate {
    start := indices.PickOffsets[targetPart]
    end := indices.PickOffsets[targetPart+1]
    
    buffer := make([]FaceCoordinate, end-start)
    for i := start; i < end; i++ {
        entityIdx := indices.PickIndices[i] / facesPerElem
        faceIdx := (indices.PickIndices[i] % facesPerElem) / pointsPerFace
        buffer[i-start] = localFaces[entityIdx*facesPerElem + faceIdx]
    }
    return buffer
}

// Similarly for unpack...
```

4. **Verifying using original connectivity**
```go
// The validation app retained EToE, EToF, EToP to check
for localElem := 0; localElem < numElems; localElem++ {
globalElem := partInfo.PartitionElems[partitionID][localElem]

for face := 0; face < facesPerElem; face++ {
neighbor := conn.EToE[globalElem][face]
if isRemote(neighbor, partitionID, partInfo.EToP) {
// Check that remoteFaces contains expected data
idx := localElem * facesPerElem + face
expected := computeExpectedRemoteFace(neighbor, conn.EToF[globalElem][face])
if remoteFaces[idx] != expected {
return fmt.Errorf("mismatch at element %d face %d", localElem, face)
}
}
}
}
```

## Key Design Features

1. **Clean Phase Separation**: Build phase uses connectivity; runtime uses only indices
2. **Entity-Level Operations**: Reduces index storage by factor of points_per_entity
3. **Fixed Stride Arrays**: Enables efficient memory access patterns
4. **Local Indexing**: All indices reference partition-local array positions

## Implementation Notes

### Performance
- Indices are built once, used many times
- Fixed stride enables vectorization
- Entity-level copying minimizes index lookups
- Cache-line alignment for indices improves memory access

### Mixed Meshes
- Different element types require separate arrays
- Each array maintains fixed stride for its type
- No mixing within arrays ensures predictable performance

### Error Handling
- Build phase validates connectivity consistency
- Runtime assumes indices are correct (validated during build)
- Bounds checking optional in production for performance