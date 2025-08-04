package utils

import (
	"fmt"
)

// FaceConnector manages pick and place indices for partitioned meshes
type FaceConnector struct {
	// Mesh dimensions
	NumPartitions int
	K             int // Total elements
	Nfaces        int // Faces per element
	Nfp           int // Face points per face
	Np            int // Nodes per element

	// Input connectivity
	VmapP []int // Face point → global solution node of neighbor
	EToP  []int // Element → partition mapping

	// Partition mappings
	ElemsPerPartition []int         // Elements per partition
	GlobalToLocalElem []map[int]int // [partition][globalElem] → localElem
	LocalToGlobalElem [][]int       // [partition][localElem] → globalElem

	// Pick/Place indices per partition
	PickIndices  [][]PickBuffer  // [sourcePartition][targetPartition]
	PlaceIndices [][]PlaceBuffer // [targetPartition][sourcePartition]
}

// PickBuffer contains indices for gathering values to send
type PickBuffer struct {
	Indices         []int // Local solution node indices
	TargetPartition int
}

// PlaceBuffer contains indices for scattering received values
type PlaceBuffer struct {
	Indices         []int // P buffer positions
	SourcePartition int
}

// NewFaceConnector creates a face connector from mesh connectivity
func NewFaceConnector(K, Nfaces, Nfp, Np int, VmapP, EToP []int) (*FaceConnector, error) {
	// Validate inputs
	if K <= 0 || Nfaces <= 0 || Nfp <= 0 || Np <= 0 {
		return nil, fmt.Errorf("invalid dimensions: K=%d, Nfaces=%d, Nfp=%d, Np=%d", K, Nfaces, Nfp, Np)
	}

	expectedFacePoints := K * Nfaces * Nfp
	if len(VmapP) != expectedFacePoints {
		return nil, fmt.Errorf("VmapP length %d does not match expected %d", len(VmapP), expectedFacePoints)
	}

	if len(EToP) != K {
		return nil, fmt.Errorf("EToP length %d does not match K=%d", len(EToP), K)
	}

	// Determine number of partitions
	numPartitions := 0
	for _, p := range EToP {
		if p+1 > numPartitions {
			numPartitions = p + 1
		}
	}

	fc := &FaceConnector{
		NumPartitions: numPartitions,
		K:             K,
		Nfaces:        Nfaces,
		Nfp:           Nfp,
		Np:            Np,
		VmapP:         VmapP,
		EToP:          EToP,
	}

	// Build partition mappings
	if err := fc.buildPartitionMappings(); err != nil {
		return nil, err
	}

	// Initialize pick/place buffers
	fc.initializeBuffers()

	// Build indices
	if err := fc.BuildIndices(); err != nil {
		return nil, err
	}

	return fc, nil
}

// buildPartitionMappings creates bidirectional mappings between global and local element numbering
func (fc *FaceConnector) buildPartitionMappings() error {
	// Count elements per partition
	fc.ElemsPerPartition = make([]int, fc.NumPartitions)
	for _, p := range fc.EToP {
		fc.ElemsPerPartition[p]++
	}

	// Initialize mapping structures
	fc.GlobalToLocalElem = make([]map[int]int, fc.NumPartitions)
	fc.LocalToGlobalElem = make([][]int, fc.NumPartitions)
	for p := 0; p < fc.NumPartitions; p++ {
		fc.GlobalToLocalElem[p] = make(map[int]int)
		fc.LocalToGlobalElem[p] = make([]int, 0, fc.ElemsPerPartition[p])
	}

	// Build mappings
	for globalElem := 0; globalElem < fc.K; globalElem++ {
		partition := fc.EToP[globalElem]
		localElem := len(fc.LocalToGlobalElem[partition])

		fc.GlobalToLocalElem[partition][globalElem] = localElem
		fc.LocalToGlobalElem[partition] = append(fc.LocalToGlobalElem[partition], globalElem)
	}

	return nil
}

// initializeBuffers creates empty pick and place buffer structures
func (fc *FaceConnector) initializeBuffers() {
	fc.PickIndices = make([][]PickBuffer, fc.NumPartitions)
	fc.PlaceIndices = make([][]PlaceBuffer, fc.NumPartitions)

	for p := 0; p < fc.NumPartitions; p++ {
		fc.PickIndices[p] = make([]PickBuffer, fc.NumPartitions)
		fc.PlaceIndices[p] = make([]PlaceBuffer, fc.NumPartitions)

		for q := 0; q < fc.NumPartitions; q++ {
			fc.PickIndices[p][q] = PickBuffer{
				Indices:         make([]int, 0),
				TargetPartition: q,
			}
			fc.PlaceIndices[p][q] = PlaceBuffer{
				Indices:         make([]int, 0),
				SourcePartition: q,
			}
		}
	}
}

// BuildIndices constructs pick and place indices for all partitions
func (fc *FaceConnector) BuildIndices() error {
	// Process each partition
	for p := 0; p < fc.NumPartitions; p++ {
		// Process each element in this partition
		for localElem := 0; localElem < fc.ElemsPerPartition[p]; localElem++ {
			globalElem := fc.LocalToGlobalElem[p][localElem]

			// Process each face of this element
			for face := 0; face < fc.Nfaces; face++ {
				// Process each face point
				for fp := 0; fp < fc.Nfp; fp++ {
					// Global face point index
					globalFacePointIdx := globalElem*fc.Nfaces*fc.Nfp + face*fc.Nfp + fp

					// Global solution node to fetch (neighbor node)
					globalSourceNode := fc.VmapP[globalFacePointIdx]

					// Which element contains this node?
					globalSourceElem := globalSourceNode / fc.Np

					// Which partition owns this element?
					sourcePartition := fc.EToP[globalSourceElem]

					// Convert to local indices in source partition
					localSourceElem := fc.GlobalToLocalElem[sourcePartition][globalSourceElem]
					nodeWithinElem := globalSourceNode % fc.Np
					localSourceNode := localSourceElem*fc.Np + nodeWithinElem

					// Calculate local P buffer position in partition p
					localPBufferPos := localElem*fc.Nfaces*fc.Nfp + face*fc.Nfp + fp

					// Add to pick indices: source partition needs to send this node to partition p
					fc.PickIndices[sourcePartition][p].Indices = append(
						fc.PickIndices[sourcePartition][p].Indices, localSourceNode)

					// Add to place indices: partition p places received value at this position
					fc.PlaceIndices[p][sourcePartition].Indices = append(
						fc.PlaceIndices[p][sourcePartition].Indices, localPBufferPos)
				}
			}
		}
	}

	return nil
}

// GetPickIndices returns pick indices for sending from source to target partition
func (fc *FaceConnector) GetPickIndices(sourcePartition, targetPartition int) []int {
	if sourcePartition < 0 || sourcePartition >= fc.NumPartitions ||
		targetPartition < 0 || targetPartition >= fc.NumPartitions {
		return nil
	}
	return fc.PickIndices[sourcePartition][targetPartition].Indices
}

// GetPlaceIndices returns place indices for target partition receiving from source
func (fc *FaceConnector) GetPlaceIndices(targetPartition, sourcePartition int) []int {
	if targetPartition < 0 || targetPartition >= fc.NumPartitions ||
		sourcePartition < 0 || sourcePartition >= fc.NumPartitions {
		return nil
	}
	return fc.PlaceIndices[targetPartition][sourcePartition].Indices
}

// Verify checks index validity and conservation properties
func (fc *FaceConnector) Verify() error {
	// Verify 1: Local validity - all pick indices are within bounds
	for p := 0; p < fc.NumPartitions; p++ {
		maxLocalNode := fc.ElemsPerPartition[p] * fc.Np
		for q := 0; q < fc.NumPartitions; q++ {
			for _, idx := range fc.PickIndices[p][q].Indices {
				if idx < 0 || idx >= maxLocalNode {
					return fmt.Errorf("invalid pick index %d for partition %d (max %d)",
						idx, p, maxLocalNode-1)
				}
			}
		}
	}

	// Verify 2: Correspondence - pick and place arrays have same length
	for p := 0; p < fc.NumPartitions; p++ {
		for q := 0; q < fc.NumPartitions; q++ {
			pickLen := len(fc.PickIndices[p][q].Indices)
			placeLen := len(fc.PlaceIndices[q][p].Indices)
			if pickLen != placeLen {
				return fmt.Errorf("length mismatch: pick[%d][%d]=%d, place[%d][%d]=%d",
					p, q, pickLen, q, p, placeLen)
			}
		}
	}

	// Verify 3: Conservation - total pick operations equals total face points
	totalPicks := 0
	for p := 0; p < fc.NumPartitions; p++ {
		for q := 0; q < fc.NumPartitions; q++ {
			totalPicks += len(fc.PickIndices[p][q].Indices)
		}
	}

	totalFacePoints := 0
	for p := 0; p < fc.NumPartitions; p++ {
		totalFacePoints += fc.ElemsPerPartition[p] * fc.Nfaces * fc.Nfp
	}

	if totalPicks != totalFacePoints {
		return fmt.Errorf("conservation error: total picks %d != total face points %d",
			totalPicks, totalFacePoints)
	}

	return nil
}
