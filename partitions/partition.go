package partitions

import (
	"fmt"
)

// GeometryType identifies the shape of an element
type GeometryType uint8

const (
	// 3D element types
	Tet     GeometryType = iota // Tetrahedron
	Hex                         // Hexahedron
	Prism                       // Triangular prism
	Pyramid                     // Square-based pyramid

	// 2D element types
	Tri       // Triangle
	Rectangle // Rectangle/Quadrilateral

	// 1D element type
	Line // Line segment
)

// Partition represents a collection of elements that execute together
// as a computational unit in the OCCA parallel execution model
type Partition struct {
	// Unique identifier for this partition
	ID int

	// Element membership
	Elements    []int // Global element indices in this partition
	NumElements int   // Actual number of active elements
	MaxElements int   // Padded size for OCCA @inner loop uniformity

	// Mixed element support
	ElementTypes []GeometryType // Type of each element (for heterogeneous meshes)
	TypeGroups   []ElementGroup // Grouped by element type for efficient processing
}

// ElementGroup represents elements of the same type within a partition
// This grouping enables type-specific operations in mixed meshes
type ElementGroup struct {
	ElementType GeometryType
	StartIndex  int   // Starting position in partition's element array
	Count       int   // Number of elements of this type
	Np          int   // Nodes per element for this type
	LocalIDs    []int // Indices within the partition
}

// PartitionLayout manages the complete mesh decomposition
type PartitionLayout struct {
	// All partitions in the mesh
	Partitions []Partition

	// Global sizing information
	KpartMax      int // max(NumElements) across all partitions for OCCA
	TotalElements int // Sum of all actual elements across partitions
	NumPartitions int // Total number of partitions

	// Element to partition mapping
	EToP []int // Length TotalElements: element k belongs to partition EToP[k]
}

// PartitionedArray represents data distributed across partitions
// with efficient access patterns for OCCA kernels
type PartitionedArray struct {
	// Contiguous global storage for all partitions
	// Layout: [Partition 0 Data][Partition 1 Data]...[Partition N-1 Data]
	GlobalData []float64

	// Offset for each partition's data in GlobalData
	// Partition p's data starts at GlobalData[Offsets[p]]
	Offsets []int

	// Number of values per element node (e.g., Np for scalar field)
	Stride int

	// Total allocated size including padding
	AllocatedSize int
}

// PartitionMapping defines how local partition data maps to communication buffers
type PartitionMapping struct {
	PartitionID int

	// Indices within the partition's local data
	LocalIndices []int

	// Corresponding positions in send/recv buffer
	BufferIndices []int

	// Number of points to transfer
	Count int
}

// RemotePartition describes communication with a partition on another MPI rank
type RemotePartition struct {
	Rank        int // MPI rank (-1 indicates local partition)
	PartitionID int // Partition ID on the remote rank

	// Location in communication buffers
	SendOffset int // Starting position in SendBuffer
	SendCount  int // Number of values to send
	RecvOffset int // Starting position in RecvBuffer
	RecvCount  int // Number of values to receive
}

// PartitionBuffer manages inter-partition communication
// Uses unified buffers for both local (shared memory) and remote (MPI) transfers
type PartitionBuffer struct {
	// Contiguous communication buffers
	// Used for both:
	// 1. Random-access scatter/gather for local partitions (OCCA kernels)
	// 2. Sequential send/recv for remote partitions (MPI)
	SendBuffer []float64
	RecvBuffer []float64

	// Scatter operation: local partition data -> SendBuffer
	ScatterMappings []PartitionMapping

	// Gather operation: RecvBuffer -> local partition data
	GatherMappings []PartitionMapping

	// Remote communication metadata
	RemotePartitions []RemotePartition

	// Size information for OCCA kernels
	MaxScatterPoints int // Maximum scatter points across all partitions
	MaxGatherPoints  int // Maximum gather points across all partitions

	// Buffer management
	SendBufferSize int
	RecvBufferSize int
}

// PartitionMetrics tracks computational load for dynamic balancing
type PartitionMetrics struct {
	// Computational cost metrics
	ElementCounts   map[GeometryType]int // Elements by type
	TotalDOFs       int                  // Total degrees of freedom
	ComputationCost float64              // Estimated FLOP count

	// Communication metrics
	LocalCommVolume  int // Data exchanged with local partitions
	RemoteCommVolume int // Data exchanged with remote partitions
	NumNeighbors     int // Number of connected partitions
}

// PartitionManager handles partition creation and optimization
type PartitionManager struct {
	// Target architecture constraints
	MaxPartitionSize int     // Maximum elements per partition (e.g., GPU shared memory limit)
	MinPartitionSize int     // Minimum for efficiency
	TargetImbalance  float64 // Acceptable load imbalance (e.g., 1.1 = 10%)

	// Hardware topology
	NumDevices       int   // Number of GPUs/accelerators
	DevicePartitions []int // Number of partitions per device
}

// Methods for PartitionLayout

// GetPartition returns the partition containing element k
func (pl *PartitionLayout) GetPartition(elementID int) int {
	if elementID < 0 || elementID >= len(pl.EToP) {
		return -1
	}
	return pl.EToP[elementID]
}

// ValidateLayout checks partition consistency
func (pl *PartitionLayout) ValidateLayout() error {
	// Verify KpartMax
	actualMax := 0
	for _, p := range pl.Partitions {
		if p.NumElements > actualMax {
			actualMax = p.NumElements
		}
		if p.MaxElements != pl.KpartMax {
			return fmt.Errorf("partition %d: MaxElements %d != KpartMax %d",
				p.ID, p.MaxElements, pl.KpartMax)
		}
	}
	if actualMax != pl.KpartMax {
		return fmt.Errorf("computed KpartMax %d != stored KpartMax %d",
			actualMax, pl.KpartMax)
	}
	return nil
}

// Methods for PartitionedArray

// GetPartitionData returns a slice for partition p's data
func (pa *PartitionedArray) GetPartitionData(partitionID int) []float64 {
	if partitionID >= len(pa.Offsets)-1 {
		return nil
	}
	start := pa.Offsets[partitionID]
	end := pa.Offsets[partitionID+1]
	return pa.GlobalData[start:end]
}

// Methods for PartitionBuffer

// RequiresRemoteCommunication checks if MPI communication is needed
func (pb *PartitionBuffer) RequiresRemoteCommunication() bool {
	for _, rp := range pb.RemotePartitions {
		if rp.Rank >= 0 {
			return true
		}
	}
	return false
}

// GetScatterIndices returns flattened index arrays for OCCA kernels
func (pb *PartitionBuffer) GetScatterIndices() (localIndices, bufferIndices []int) {
	totalPoints := 0
	for _, m := range pb.ScatterMappings {
		totalPoints += len(m.LocalIndices)
	}

	localIndices = make([]int, totalPoints)
	bufferIndices = make([]int, totalPoints)

	offset := 0
	for _, m := range pb.ScatterMappings {
		copy(localIndices[offset:], m.LocalIndices)
		copy(bufferIndices[offset:], m.BufferIndices)
		offset += len(m.LocalIndices)
	}

	return localIndices, bufferIndices
}
