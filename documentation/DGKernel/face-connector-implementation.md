# FaceConnector Implementation Plan

## Overview
The FaceConnector package will implement the pick/place index design from partition-buffer-design.md for managing face point data exchange in partitioned DG meshes.

## Package Structure

### Location
`utils/face_connector.go` and `utils/face_connector_test.go`

### Core Data Structures

```go
// FaceConnector manages pick and place indices for partitioned meshes
type FaceConnector struct {
    // Mesh dimensions
    NumPartitions int
    K            int    // Total elements
    Nfaces       int    // Faces per element
    Nfp          int    // Face points per face
    Np           int    // Nodes per element
    
    // Input connectivity
    VmapP        []int  // Face point → global solution node of neighbor
    EToP         []int  // Element → partition mapping
    
    // Partition mappings
    ElemsPerPartition    []int              // Elements per partition
    GlobalToLocalElem    []map[int]int     // [partition][globalElem] → localElem
    LocalToGlobalElem    [][]int           // [partition][localElem] → globalElem
    
    // Pick/Place indices per partition
    PickIndices  [][]PickBuffer   // [sourcePartition][targetPartition]
    PlaceIndices [][]PlaceBuffer  // [targetPartition][sourcePartition]
}

// PickBuffer contains indices for gathering values to send
type PickBuffer struct {
    Indices         []int  // Local solution node indices
    TargetPartition int
}

// PlaceBuffer contains indices for scattering received values
type PlaceBuffer struct {
    Indices         []int  // P buffer positions
    SourcePartition int
}
```

### Core Methods

```go
// NewFaceConnector creates a face connector from mesh connectivity
func NewFaceConnector(K, Nfaces, Nfp, Np int, VmapP, EToP []int) (*FaceConnector, error)

// BuildIndices constructs pick and place indices for all partitions
func (fc *FaceConnector) BuildIndices() error

// GetPickIndices returns pick indices for sending from source to target partition
func (fc *FaceConnector) GetPickIndices(sourcePartition, targetPartition int) []int

// GetPlaceIndices returns place indices for target partition receiving from source
func (fc *FaceConnector) GetPlaceIndices(targetPartition, sourcePartition int) []int

// Verify checks index validity and conservation properties
func (fc *FaceConnector) Verify() error
```

## Test Strategy

### Test Meshes

1. **SingleTet (Non-partitioned)**
   - 1 partition containing 1 tetrahedron
   - All face connections are self-connections
   - Pick/place buffers only for partition 0 → 0

2. **TwoTetMesh (Simple Partitioned)**
   - 2 partitions, each with 1 tetrahedron
   - One shared face between partitions
   - Tests basic cross-partition communication

3. **CubeMesh (Complex Partitioned)**
   - Multiple partitions (2-3)
   - Multiple shared faces per partition
   - Tests complex connectivity patterns

### Test Cases

1. **Basic Functionality**
   - Correct number of pick/place buffers created
   - Buffer dimensions match expected sizes
   - Index ranges are valid

2. **Mathematical Properties**
   - Conservation: Each face point appears exactly once
   - Reciprocity: Pick/place maintain correspondence
   - Local validity: All indices within bounds

3. **Edge Cases**
   - Single partition (no communication)
   - Unbalanced partitions
   - Different element orders (N)

## Implementation Steps

1. Create basic structure and constructor
2. Implement partition mapping builders
3. Implement pick/place index construction
4. Add verification methods
5. Write comprehensive tests

## Key Algorithms

### Building Pick/Place Indices

```
For each partition p:
    For each face point in partition p:
        1. Find global source node from VmapP
        2. Determine source partition from EToP
        3. Convert global node to local node in source partition
        4. Add to PickIndices[sourcePartition][p]
        5. Add local P buffer position to PlaceIndices[p][sourcePartition]
```

### Global to Local Node Conversion

```
globalElem = globalNode / Np
localElem = GlobalToLocalElem[partition][globalElem]
nodeWithinElem = globalNode % Np
localNode = localElem * Np + nodeWithinElem
```

## Expected Outcomes

1. Efficient index structures for MPI communication
2. Verified conservation of face point data
3. Support for arbitrary partitioning schemes
4. Clear separation of concerns from mesh generation
