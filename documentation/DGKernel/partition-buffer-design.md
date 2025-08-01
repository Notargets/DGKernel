# Pick/Place Index Design for P Buffer Construction

## Overview

This document specifies a system for building pick and place indices that enable construction of P buffers (neighbor face values) in partitioned discontinuous Galerkin methods. The design handles both single and multi-partition cases through a unified code path.

## Core Concepts

### Entities
- **Face Point**: A single quadrature point on an element face
- **P Buffer**: Per-partition buffer containing neighbor face point values in natural traversal order
- **Pick Buffer**: Intermediate buffer containing solution values to send from one partition to another
- **Place Buffer**: Intermediate buffer containing solution values received from another partition
- **Local Solution Node**: Node index within a partition's local solution array (0 to Np*K_local-1)
- **Global Solution Node**: Node index in the unpartitioned mesh (0 to Np*K_total-1)

### Key Invariants
1. Every partition maintains exactly NumPartitions pick buffers (including one for itself)
2. Every partition maintains exactly NumPartitions place buffers (including one from itself)
3. Pick and place operations use the same code path regardless of partition count
4. Pick indices reference local partition solution nodes; place indices reference P buffer positions

## Input Data Structures

- `VmapP[totalFacePoints]`: Maps face point index → global solution node of matching neighbor
- `EToP[K]`: Maps global element ID → partition ID
- `NumPartitions`: Total number of partitions
- `K`: Total elements in mesh
- `Nfaces`: Faces per element
- `Nfp`: Face points per face
- `Np`: Nodes per element

## Output Data Structures

### Per Partition
```
PickBuffers[NumPartitions]
    - Values: array of solution values to send
    - TargetPartition: destination partition ID
    
PickIndices[NumPartitions]
    - Indices: local solution node indices within source partition
    
PlaceBuffers[NumPartitions]
    - Values: array of solution values received
    - SourcePartition: origin partition ID
    
PlaceIndices[NumPartitions]
    - Indices: positions in P buffer to place into
```

## Design Details

### Phase 1: Build Partition Mappings
Create bidirectional mappings between global and local element numbering:
1. `GlobalToLocalElem[partition][globalElem] → localElem`
2. `LocalToGlobalElem[partition][localElem] → globalElem`
3. `ElemsPerPartition[partition]` - count of elements in each partition

### Phase 2: Partition Face Point Ownership
For each partition p:
1. Identify face point range in global indexing
2. Build local face point range for P buffer allocation

### Phase 3: Build Pick/Place Indices
For each partition p:
1. Initialize arrays: `PickIndices[p][q]` and `PlaceIndices[p][q]` for all q ∈ [0, NumPartitions)
2. For each face point index i in partition p's range:
   - `globalSourceNode = VmapP[i]` (global solution node to fetch)
   - `globalSourceElem = globalSourceNode / Np` (global element containing this node)
   - `sourcePartition = EToP[globalSourceElem]` (which partition owns this element)
   - `localSourceElem = GlobalToLocalElem[sourcePartition][globalSourceElem]`
   - `nodeWithinElem = globalSourceNode % Np`
   - `localSourceNode = localSourceElem * Np + nodeWithinElem`
   - Append `localSourceNode` to `PickIndices[sourcePartition][p]`
   - Append local P buffer position to `PlaceIndices[p][sourcePartition]`

### Critical Implementation Details

#### Global to Local Node Conversion
For a global solution node index:
```
globalNode = globalElem * Np + nodeWithinElem
localNode = localElem * Np + nodeWithinElem

where:
localElem = GlobalToLocalElem[partition][globalElem]
```

#### Local P Buffer Position Calculation
For partition p with face point index i (global):
```
globalElemForFacePoint = i / (Nfaces * Nfp)
localElemForFacePoint = GlobalToLocalElem[p][globalElemForFacePoint]
faceWithinElem = (i % (Nfaces * Nfp)) / Nfp
fpWithinFace = i % Nfp
localPBufferPos = localElemForFacePoint * Nfaces * Nfp + faceWithinElem * Nfp + fpWithinFace
```

#### Index Correspondence
The pick and place indices maintain strict correspondence:
- `PickIndices[p][q][k]` identifies the k-th local solution node partition p sends to partition q
- `PlaceIndices[q][p][k]` identifies where partition q places the k-th value received from partition p
- These arrays have the same length and maintain element-wise correspondence

## Verification Properties

1. **Local Validity**: All pick indices for partition p are in range [0, Np*ElemsPerPartition[p])
2. **Conservation**: Each face point appears in exactly one pick buffer set across all partitions
3. **Reciprocity**: Pick/place indices maintain one-to-one correspondence between partitions

## Usage Pattern

1. **Gather Phase**: Use `PickIndices[p][q]` to fill `PickBuffers[p][q]` from partition p's local solution array
2. **Exchange Phase**: Transfer `PickBuffers[p][q]` to `PlaceBuffers[q][p]`
3. **Scatter Phase**: Use `PlaceIndices[q][p]` to scatter `PlaceBuffers[q][p]` into partition q's P buffer
4. **BC Override**: Apply boundary conditions to P buffer positions where needed

This design ensures that each partition works only with its local data structures while maintaining correct global connectivity relationships.
