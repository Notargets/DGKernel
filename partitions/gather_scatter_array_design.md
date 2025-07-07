# Gather/Scatter Array Indexing Design Document

## Overview

This document specifies the design for a gather/scatter array indexing system that enables efficient inter-partition data communication. The system computes entity-level pick and place indices on the host (golang) that kernels use to transfer data between partitions.

## Core Concepts

### Partitions and Elements
- Computational domain divided into `Npart` partitions
- Each partition contains elements of potentially different types/orders
- Elements are numbered locally within each partition (0 to K[part]-1)

### Arrays
An **Array** represents entity data for a specific element type/order combination:
- Fixed stride throughout the array (all entities have same point count)
- Organized as: element → entity → points
- Examples: "P3_Hex_Face_Array", "P2_Tet_Edge_Array"

### Entity-Level Indexing
- **Pick indices**: Point to start of entities in sender's M array
- **Place indices**: Point to start of entities in receiver's P array
- One index per entity (not per point)

### M and P Arrays
- **M array**: "Minus" values - this element's entity values
- **P array**: "Plus" values - neighbor's values (local, BC, or remote)

## Design Principles

1. **Fixed Stride per Array**: Each Array has uniform entity sizes
2. **Type Safety**: Arrays only communicate with compatible Arrays
3. **Entity-Level Operations**: Indices point to entity starts, not individual points
4. **Local Coordinates**: All indices use partition-local element numbering

## Core Data Structures

### Reference Geometry
Provides entity point groupings for each element type:
```go
type ReferenceGeometry struct {
    R, S, T        []float64   // Node coordinates (length Np)
    VertexPoints   []int       // Indices of vertex nodes
    EdgePoints     [][]int     // [edge][indices] - nodes on each edge
    FacePoints     [][]int     // [face][indices] - nodes on each face
    InteriorPoints []int       // Interior node indices
}
```

### Array Definition
```go
type Array struct {
    Name           string      // e.g., "P3_Hex_Face_M"
    ElementType    ElementType // Hex, Tet, Prism, etc.
    Order          int         // Polynomial order
    EntityType     EntityType  // Face, Edge, Vertex
    
    // Fixed sizes for this array
    ElementsPerPartition []int  // [partition] → number of elements
    EntitiesPerElement   int    // e.g., 6 faces per hex
    PointsPerEntity      int    // e.g., 16 points per P3 face
    
    // Total stride per element
    StridePerElement     int    // = EntitiesPerElement × PointsPerEntity
}

// Entity location within the array (fixed stride)
func (a *Array) Locate(localElementID, entityID int) int {
    return localElementID * a.StridePerElement + entityID * a.PointsPerEntity
}
```

### Connectivity Structures (Global IDs)
```go
type ElementConnectivity struct {
    EToE mat.Matrix  // [K × NFaces] Element k, face f connects to element EToE[k,f]
    EToF mat.Matrix  // [K × NFaces] Element k, face f connects to face EToF[k,f]
    BCType mat.Matrix // [K × NFaces] Boundary type (-1 for interior)
}

type PartitionMapping struct {
    EToP []int // [globalElementID] → partitionID
}
```

### ID Translation
```go
type PartitionLayout struct {
    PartitionID   int
    LocalToGlobal []int         // [localID] → globalID
    GlobalToLocal map[int]int   // globalID → localID (or -1)
}
```

### Index Storage
```go
type GatherScatterIndices struct {
    PartitionID int
    ArrayName   string  // Which array these indices are for
    
    // Entity-level indices (one per entity, not per point)
    PickIndices  []int32  // Concatenated for all source partitions
    PickOffsets  []int32  // [Npart+1] Starting position per source
    PickCounts   []int32  // [Npart] Number of entities per source
    
    PlaceIndices []int32  // Concatenated for all source partitions
    PlaceOffsets []int32  // [Npart+1] Starting position per source
    PlaceCounts  []int32  // [Npart] Number of entities per source
}
```

## Array Construction Process

Arrays are built by scanning through connectivity to determine data sources:

### 1. P Array Construction Algorithm

For each partition, iterate through elements and entities to build the P array structure:

```go
func BuildPArray(
    partition int,
    array *Array,
    connectivity *ElementConnectivity,
    partMapping *PartitionMapping,
    layout map[int]*PartitionLayout,
) *GatherScatterIndices {
    
    indices := &GatherScatterIndices{
        PartitionID: partition,
        ArrayName:   array.Name,
    }
    
    // Temporary storage per source partition
    pickMap := make(map[int][]int32)
    placeMap := make(map[int][]int32)
    
    // Current position in P array as we iterate
    pArrayPosition := 0
    
    // Iterate through array in order
    for localElem := 0; localElem < array.ElementsPerPartition[partition]; localElem++ {
        globalElem := layout[partition].LocalToGlobal[localElem]
        
        for entity := 0; entity < array.EntitiesPerElement; entity++ {
            neighbor := connectivity.EToE.At(globalElem, entity)
            neighborEntity := connectivity.EToF.At(globalElem, entity)
            
            if neighbor == globalElem {
                // Boundary face - P array will use BC
                // No pick/place indices needed
            } else if partMapping.EToP[neighbor] == partition {
                // Local neighbor - P array points to local M
                // No pick/place indices needed (handled in kernel)
            } else {
                // Remote neighbor - needs pick/place indices
                remotePart := partMapping.EToP[neighbor]
                remoteLocal := layout[remotePart].GlobalToLocal[neighbor]
                
                // Pick from remote M array at entity start
                remoteArray := getRemoteMArray(array, remotePart)
                pickLocation := remoteArray.Locate(remoteLocal, neighborEntity)
                
                pickMap[remotePart] = append(pickMap[remotePart], int32(pickLocation))
                placeMap[remotePart] = append(placeMap[remotePart], int32(pArrayPosition))
            }
            
            // Move to next entity position in P array
            pArrayPosition += array.PointsPerEntity
        }
    }
    
    // Concatenate indices with offsets
    return concatenateIndices(indices, pickMap, placeMap)
}
```

### 2. Index Concatenation

```go
func concatenateIndices(
    indices *GatherScatterIndices,
    pickMap, placeMap map[int][]int32,
) *GatherScatterIndices {
    
    numPart := len(pickMap) + 1 // Assuming we know total partitions
    indices.PickOffsets = make([]int32, numPart+1)
    indices.PlaceOffsets = make([]int32, numPart+1)
    indices.PickCounts = make([]int32, numPart)
    indices.PlaceCounts = make([]int32, numPart)
    
    offset := int32(0)
    for src := 0; src < numPart; src++ {
        indices.PickOffsets[src] = offset
        indices.PlaceOffsets[src] = offset
        
        if picks, ok := pickMap[src]; ok {
            indices.PickIndices = append(indices.PickIndices, picks...)
            indices.PlaceIndices = append(indices.PlaceIndices, placeMap[src]...)
            indices.PickCounts[src] = int32(len(picks))
            indices.PlaceCounts[src] = int32(len(picks))
            offset += int32(len(picks))
        }
    }
    indices.PickOffsets[numPart] = offset
    indices.PlaceOffsets[numPart] = offset
    
    return indices
}
```

## Usage Example

```go
// 1. Define reference elements
hexP3Ref := &ReferenceGeometry{
    FacePoints: [][]int{
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, // 16 points per P3 face
        // ... other faces
    },
}

// 2. Define arrays for each element type in each partition
arrays := make(map[int][]*Array)
for p := 0; p < nPartitions; p++ {
    // Count elements by type in this partition
    hexCount := countElementType(p, HexType)
    tetCount := countElementType(p, TetType)
    
    if hexCount > 0 {
        arrays[p] = append(arrays[p], &Array{
            Name:                 "P3_Hex_Face_M",
            ElementType:          HexType,
            Order:                3,
            EntityType:           Face,
            ElementsPerPartition: getElementCounts(),
            EntitiesPerElement:   6,
            PointsPerEntity:      16,
            StridePerElement:     6 * 16,
        })
    }
    
    if tetCount > 0 {
        arrays[p] = append(arrays[p], &Array{
            Name:                 "P2_Tet_Face_M",
            ElementType:          TetType,
            Order:                2,
            EntityType:           Face,
            ElementsPerPartition: getElementCounts(),
            EntitiesPerElement:   4,
            PointsPerEntity:      6,
            StridePerElement:     4 * 6,
        })
    }
}

// 3. Build P arrays and their indices
for p := 0; p < nPartitions; p++ {
    for _, array := range arrays[p] {
        // Create corresponding P array
        pArray := &Array{
            Name: strings.Replace(array.Name, "_M", "_P", 1),
            // ... copy other fields
        }
        
        indices := BuildPArray(p, pArray, connectivity, partMapping, layouts)
        // Store indices for device transfer
    }
}
```

## Kernel Usage

```c
// Pack send buffer (at sender)
@kernel void packSendBuffer(const int* pickIndices,
                           const int* pickOffsets,
                           const int pointsPerEntity,
                           const real* mArray,
                           real* sendBuffer) {
    
    for (int tgt = 0; tgt < Npart; ++tgt) {
        if (tgt == myPartition) continue;
        
        const int start = pickOffsets[tgt];
        const int count = pickOffsets[tgt+1] - start;
        
        for (int e = 0; e < count; ++e) {
            const int entityStart = pickIndices[start + e];
            
            // Copy all points of this entity
            for (int p = 0; p < pointsPerEntity; ++p) {
                sendBuffer[...] = mArray[entityStart + p];
            }
        }
    }
}

// Unpack receive buffer (at receiver)
@kernel void unpackReceiveBuffer(const int* placeIndices,
                                const int* placeOffsets,
                                const int pointsPerEntity,
                                const real* recvBuffer,
                                real* pArray) {
    
    for (int src = 0; src < Npart; ++src) {
        if (src == myPartition) continue;
        
        const int start = placeOffsets[src];
        const int count = placeOffsets[src+1] - start;
        
        for (int e = 0; e < count; ++e) {
            const int entityStart = placeIndices[start + e];
            
            // Copy all points of this entity
            for (int p = 0; p < pointsPerEntity; ++p) {
                pArray[entityStart + p] = recvBuffer[...];
            }
        }
    }
}
```

## Mixed Mesh Considerations

### Separate Arrays by Type
Each element type/order combination has its own set of arrays:
- No mixing within arrays maintains fixed stride
- Clear naming convention: "P{order}_{type}_{entity}_{M/P}"

### Mortar Elements
For interfaces between different element types:
```go
type MortarArray struct {
    LeftArray  *Array  // e.g., P3_Hex_Face
    RightArray *Array  // e.g., P2_Tet_Face
    
    // Interpolation data
    LeftToMortar  []float64  // Interpolation matrix
    RightToMortar []float64  // Interpolation matrix
}
```

## Validation

```go
func ValidateIndices(indices *GatherScatterIndices, array *Array) error {
    // Check entity-level bounds
    maxEntity := array.ElementsPerPartition[indices.PartitionID] * 
                 array.EntitiesPerElement
    
    for _, idx := range indices.PickIndices {
        entityIndex := idx / array.PointsPerEntity
        if entityIndex >= int32(maxEntity) {
            return fmt.Errorf("pick index out of bounds")
        }
    }
    
    // Verify counts match
    for p := 0; p < len(indices.PickCounts); p++ {
        if indices.PickCounts[p] != indices.PlaceCounts[p] {
            return fmt.Errorf("mismatched entity counts")
        }
    }
    
    return nil
}
```

## Performance Optimizations

1. **Cache-line alignment**: Align offset boundaries to 64 bytes
2. **Entity ordering**: Sort by element ID for sequential access
3. **Prefetch-friendly**: Fixed stride enables hardware prefetching
4. **Minimal indirection**: Direct entity-level indexing

## Summary

This design provides:
- Clean separation between mesh connectivity (global) and array indexing (local)
- Fixed-stride arrays for optimal performance
- Entity-level indexing to minimize index storage
- Type-safe array definitions that prevent incompatible communications
- Natural integration with DG face/edge/vertex operations