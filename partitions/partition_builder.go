package partitions

import (
	"fmt"
	"math"
)

// PartitionBuilder constructs partitions from mesh connectivity
type PartitionBuilder struct {
	// Mesh connectivity
	Mesh *MeshConnectivity

	// Partitioning parameters
	TargetPartitionSize int     // Desired elements per partition
	MaxImbalance        float64 // Acceptable load imbalance
	Strategy            PartitionStrategy
}

// MeshConnectivity provides the mesh topology needed for partitioning
type MeshConnectivity struct {
	NumElements  int
	ElementTypes []GeometryType
	NodesPerElem []int // Np for each element

	// Face connectivity for minimizing communication
	EToE [][]int // Element-to-element connectivity
	EToF [][]int // Element-to-face connectivity
}

// PartitionStrategy defines how elements are grouped
type PartitionStrategy int

const (
	// Simple strategies
	BlockPartition PartitionStrategy = iota // Consecutive elements
	RoundRobin                              // Distribute cyclically

	// Graph-based strategies
	GraphPartition    // Use METIS or similar
	SpaceFillingCurve // Hilbert/Morton curve ordering
)

// BuildPartitions creates a partition layout from mesh connectivity
func (pb *PartitionBuilder) BuildPartitions() (*PartitionLayout, error) {
	// Determine number of partitions needed
	numPartitions := pb.calculateNumPartitions()

	// Partition the elements
	eToP := pb.partitionElements(numPartitions)

	// Create partition structures
	partitions := pb.createPartitions(eToP, numPartitions)

	// Calculate KpartMax for OCCA
	kpartMax := pb.calculateKpartMax(partitions)

	// Set MaxElements for all partitions
	for i := range partitions {
		partitions[i].MaxElements = kpartMax
	}

	// Create the layout
	layout := &PartitionLayout{
		Partitions:    partitions,
		KpartMax:      kpartMax,
		TotalElements: pb.Mesh.NumElements,
		NumPartitions: numPartitions,
		EToP:          eToP,
	}

	// Validate the layout
	if err := layout.ValidateLayout(); err != nil {
		return nil, fmt.Errorf("invalid partition layout: %w", err)
	}

	return layout, nil
}

// calculateNumPartitions determines optimal partition count
func (pb *PartitionBuilder) calculateNumPartitions() int {
	// Basic calculation based on target size
	numPartitions := int(math.Ceil(float64(pb.Mesh.NumElements) / float64(pb.TargetPartitionSize)))

	// Ensure at least one partition
	if numPartitions < 1 {
		numPartitions = 1
	}

	return numPartitions
}

// partitionElements assigns elements to partitions
func (pb *PartitionBuilder) partitionElements(numPartitions int) []int {
	eToP := make([]int, pb.Mesh.NumElements)

	switch pb.Strategy {
	case BlockPartition:
		// Simple block partitioning
		elementsPerPartition := int(math.Ceil(float64(pb.Mesh.NumElements) / float64(numPartitions)))
		for i := 0; i < pb.Mesh.NumElements; i++ {
			eToP[i] = i / elementsPerPartition
			if eToP[i] >= numPartitions {
				eToP[i] = numPartitions - 1
			}
		}

	case RoundRobin:
		// Distribute elements cyclically
		for i := 0; i < pb.Mesh.NumElements; i++ {
			eToP[i] = i % numPartitions
		}

	case GraphPartition:
		// Would use METIS or similar graph partitioner
		// For now, fall back to block partitioning
		return pb.partitionWithStrategy(BlockPartition, numPartitions)

	default:
		// Default to block partitioning
		return pb.partitionWithStrategy(BlockPartition, numPartitions)
	}

	return eToP
}

// partitionWithStrategy recursively applies a different strategy
func (pb *PartitionBuilder) partitionWithStrategy(strategy PartitionStrategy, numPartitions int) []int {
	oldStrategy := pb.Strategy
	pb.Strategy = strategy
	result := pb.partitionElements(numPartitions)
	pb.Strategy = oldStrategy
	return result
}

// createPartitions builds partition structures from element assignments
func (pb *PartitionBuilder) createPartitions(eToP []int, numPartitions int) []Partition {
	partitions := make([]Partition, numPartitions)

	// Initialize partitions
	for i := range partitions {
		partitions[i] = Partition{
			ID:           i,
			Elements:     make([]int, 0),
			ElementTypes: make([]GeometryType, 0),
		}
	}

	// Assign elements to partitions
	for elem, part := range eToP {
		partitions[part].Elements = append(partitions[part].Elements, elem)
		if pb.Mesh.ElementTypes != nil {
			partitions[part].ElementTypes = append(partitions[part].ElementTypes,
				pb.Mesh.ElementTypes[elem])
		}
		partitions[part].NumElements++
	}

	// Create element groups for mixed meshes
	for i := range partitions {
		partitions[i].TypeGroups = pb.createElementGroups(&partitions[i])
	}

	return partitions
}

// createElementGroups organizes elements by type within a partition
func (pb *PartitionBuilder) createElementGroups(p *Partition) []ElementGroup {
	if len(p.ElementTypes) == 0 {
		return nil
	}

	// Count elements by type
	typeCounts := make(map[GeometryType][]int)
	for i, elemType := range p.ElementTypes {
		typeCounts[elemType] = append(typeCounts[elemType], i)
	}

	// Create groups
	groups := make([]ElementGroup, 0, len(typeCounts))
	currentIndex := 0

	for elemType, indices := range typeCounts {
		group := ElementGroup{
			ElementType: elemType,
			StartIndex:  currentIndex,
			Count:       len(indices),
			LocalIDs:    indices,
		}

		// Get Np for this element type if available
		if pb.Mesh.NodesPerElem != nil {
			// Assuming first element of this type represents all
			globalElemID := p.Elements[indices[0]]
			group.Np = pb.Mesh.NodesPerElem[globalElemID]
		}

		groups = append(groups, group)
		currentIndex += len(indices)
	}

	return groups
}

// calculateKpartMax finds maximum elements across all partitions
func (pb *PartitionBuilder) calculateKpartMax(partitions []Partition) int {
	kpartMax := 0
	for _, p := range partitions {
		if p.NumElements > kpartMax {
			kpartMax = p.NumElements
		}
	}
	return kpartMax
}

// BuildPartitionBuffers creates communication buffers for a partition layout
// using the face buffer design pattern for efficient face data exchange
func BuildPartitionBuffers(layout *PartitionLayout, mesh *MeshData) ([]*PartitionBuffer, error) {
	buffers := make([]*PartitionBuffer, layout.NumPartitions)

	// First pass: analyze all faces to determine communication patterns
	commPatterns := analyzePartitionCommunication(layout, mesh)

	// Second pass: build buffers for each partition
	for partID := range layout.Partitions {
		buffers[partID] = buildPartitionBuffer(partID, layout, mesh, commPatterns)
	}

	// Validate symmetry of communication
	if err := validateCommunicationSymmetry(buffers); err != nil {
		return nil, fmt.Errorf("asymmetric communication pattern: %w", err)
	}

	return buffers, nil
}

// MeshData contains the mesh connectivity needed for buffer building
type MeshData struct {
	EToE      [][]int // Element-to-element connectivity
	EToF      [][]int // Element-to-face connectivity
	FaceNodes [][]int // Nodes defining each face
	Nfp       int     // Number of points per face
	Nfaces    int     // Number of faces per element
}

// analyzePartitionCommunication determines which faces need communication
func analyzePartitionCommunication(layout *PartitionLayout, mesh *MeshData) map[int][]FaceCommunication {
	patterns := make(map[int][]FaceCommunication)

	// For each partition
	for partID, partition := range layout.Partitions {
		var faceComm []FaceCommunication

		// For each element in partition
		for localElemIdx := 0; localElemIdx < partition.NumElements; localElemIdx++ {
			globalElem := partition.Elements[localElemIdx]

			// For each face of element
			for face := 0; face < mesh.Nfaces; face++ {
				neighbor := mesh.EToE[globalElem][face]

				// Skip boundary faces
				if neighbor == globalElem {
					continue
				}

				// Check if neighbor is in different partition
				neighborPart := layout.GetPartition(neighbor)
				if neighborPart != partID && neighborPart >= 0 {
					// This face needs remote communication
					neighborFace := mesh.EToF[globalElem][face]

					fc := FaceCommunication{
						LocalElement:    localElemIdx,
						LocalFace:       face,
						RemotePartition: neighborPart,
						RemoteElement:   neighbor,
						RemoteFace:      neighborFace,
					}
					faceComm = append(faceComm, fc)
				}
			}
		}
		patterns[partID] = faceComm
	}

	return patterns
}

// FaceCommunication describes a face that needs inter-partition communication
type FaceCommunication struct {
	LocalElement    int // Element index within partition
	LocalFace       int // Face index within element
	RemotePartition int // Target partition ID
	RemoteElement   int // Global element ID in remote partition
	RemoteFace      int // Face index in remote element
}

// buildPartitionBuffer creates buffer for a single partition
func buildPartitionBuffer(partID int, layout *PartitionLayout, mesh *MeshData,
	patterns map[int][]FaceCommunication) *PartitionBuffer {

	partition := layout.Partitions[partID]
	_ = partition
	faceComm := patterns[partID]

	// Group faces by remote partition
	remoteGroups := make(map[int][]FaceCommunication)
	for _, fc := range faceComm {
		remoteGroups[fc.RemotePartition] = append(remoteGroups[fc.RemotePartition], fc)
	}

	// Build mappings and calculate buffer sizes
	var scatterMappings []PartitionMapping
	var gatherMappings []PartitionMapping
	var remotePartitions []RemotePartition

	sendOffset := 0
	recvOffset := 0

	// Process each remote partition
	for remotePart, faces := range remoteGroups {
		numFacePoints := len(faces) * mesh.Nfp

		// Build scatter mapping (what we send)
		localIndices := make([]int, 0, numFacePoints)
		bufferIndices := make([]int, 0, numFacePoints)

		for _, fc := range faces {
			// Calculate M buffer indices for this face
			// Using natural traversal order: elem*Nfaces*Nfp + face*Nfp + point
			mStart := fc.LocalElement*mesh.Nfaces*mesh.Nfp + fc.LocalFace*mesh.Nfp

			for p := 0; p < mesh.Nfp; p++ {
				localIndices = append(localIndices, mStart+p)
				bufferIndices = append(bufferIndices, sendOffset+len(localIndices)-1)
			}
		}

		scatterMappings = append(scatterMappings, PartitionMapping{
			PartitionID:   remotePart,
			LocalIndices:  localIndices,
			BufferIndices: bufferIndices,
			Count:         numFacePoints,
		})

		// Build gather mapping (what we receive)
		// For now, allocate space - actual mapping built by remote partition
		gatherMappings = append(gatherMappings, PartitionMapping{
			PartitionID:   remotePart,
			Count:         numFacePoints,
			BufferIndices: makeRange(recvOffset, recvOffset+numFacePoints),
		})

		// Record remote partition info
		remotePartitions = append(remotePartitions, RemotePartition{
			Rank:        getPartitionRank(remotePart), // Would map partition to MPI rank
			PartitionID: remotePart,
			SendOffset:  sendOffset,
			SendCount:   numFacePoints,
			RecvOffset:  recvOffset,
			RecvCount:   numFacePoints,
		})

		sendOffset += numFacePoints
		recvOffset += numFacePoints
	}

	// Calculate max points for OCCA kernels
	maxScatter := 0
	maxGather := 0
	for _, m := range scatterMappings {
		if m.Count > maxScatter {
			maxScatter = m.Count
		}
	}
	for _, m := range gatherMappings {
		if m.Count > maxGather {
			maxGather = m.Count
		}
	}

	return &PartitionBuffer{
		SendBuffer:       make([]float64, sendOffset),
		RecvBuffer:       make([]float64, recvOffset),
		ScatterMappings:  scatterMappings,
		GatherMappings:   gatherMappings,
		RemotePartitions: remotePartitions,
		MaxScatterPoints: maxScatter,
		MaxGatherPoints:  maxGather,
		SendBufferSize:   sendOffset,
		RecvBufferSize:   recvOffset,
	}
}

// Helper functions

func makeRange(start, end int) []int {
	r := make([]int, end-start)
	for i := range r {
		r[i] = start + i
	}
	return r
}

func getPartitionRank(partitionID int) int {
	// In a real implementation, this would map partition IDs to MPI ranks
	// For now, assume partition ID equals rank for remote partitions
	// Return -1 for local partitions (would be determined by actual distribution)
	return partitionID
}

func validateCommunicationSymmetry(buffers []*PartitionBuffer) error {
	// Verify that if partition A sends to partition B,
	// then partition B expects to receive from partition A

	// Build send expectations
	sendMap := make(map[string]int) // "sender:receiver" -> count
	for senderID, buf := range buffers {
		for _, rp := range buf.RemotePartitions {
			key := fmt.Sprintf("%d:%d", senderID, rp.PartitionID)
			sendMap[key] = rp.SendCount
		}
	}

	// Verify receive expectations match
	for receiverID, buf := range buffers {
		for _, rp := range buf.RemotePartitions {
			key := fmt.Sprintf("%d:%d", rp.PartitionID, receiverID)
			expectedCount, exists := sendMap[key]
			if !exists {
				return fmt.Errorf("partition %d expects to receive from %d, but %d doesn't send",
					receiverID, rp.PartitionID, rp.PartitionID)
			}
			if expectedCount != rp.RecvCount {
				return fmt.Errorf("count mismatch: partition %d sends %d to %d, but %d expects %d",
					rp.PartitionID, expectedCount, receiverID, receiverID, rp.RecvCount)
			}
		}
	}

	return nil
}

// AllocatePartitionedArray creates storage for field data across partitions
func AllocatePartitionedArray(layout *PartitionLayout, nodesPerElement []int) *PartitionedArray {
	// Calculate offsets and total size
	offsets := make([]int, layout.NumPartitions+1)
	offsets[0] = 0

	for i, p := range layout.Partitions {
		partitionSize := 0

		// Sum up space needed for all elements in partition
		for j := 0; j < p.NumElements; j++ {
			elemID := p.Elements[j]
			partitionSize += nodesPerElement[elemID]
		}

		// Pad to MaxElements for OCCA
		if p.NumElements < p.MaxElements {
			// Padding with zeros - use first element's Np for padding size
			if p.NumElements > 0 {
				elemID := p.Elements[0]
				paddingPerElem := nodesPerElement[elemID]
				partitionSize += paddingPerElem * (p.MaxElements - p.NumElements)
			}
		}

		offsets[i+1] = offsets[i] + partitionSize
	}

	totalSize := offsets[layout.NumPartitions]

	return &PartitionedArray{
		GlobalData:    make([]float64, totalSize),
		Offsets:       offsets,
		AllocatedSize: totalSize,
	}
}

// Example usage for creating partition buffers from face connectivity
func CreateFaceCommBuffers(layout *PartitionLayout, faceConn FaceConnectivity) []*PartitionBuffer {
	// This is a sketch of how face connectivity drives buffer creation
	buffers := make([]*PartitionBuffer, layout.NumPartitions)

	// For each partition, determine which faces need communication
	for pID := range layout.Partitions {
		sendIndices := make([]int, 0)
		recvIndices := make([]int, 0)

		// Analyze faces to find communication needs
		// ... (implementation depends on face data structure)

		buffers[pID] = &PartitionBuffer{
			SendBuffer: make([]float64, len(sendIndices)),
			RecvBuffer: make([]float64, len(recvIndices)),
			// ... populate mappings
		}
	}

	return buffers
}

// FaceConnectivity represents face-based mesh connectivity
type FaceConnectivity struct {
	// Face ownership
	FaceToElement [][]int // [faceID][elem1, elem2] (-1 if boundary)
	ElementToFace [][]int // [elemID][local_face_indices]

	// Face geometry
	FaceNodes [][]int // Nodes defining each face
}

// PartitionStatistics computes load balance metrics
func (layout *PartitionLayout) PartitionStatistics() PartitionStats {
	stats := PartitionStats{
		NumPartitions: layout.NumPartitions,
		MinElements:   math.MaxInt32,
		MaxElements:   0,
		AvgElements:   float64(layout.TotalElements) / float64(layout.NumPartitions),
	}

	for _, p := range layout.Partitions {
		if p.NumElements < stats.MinElements {
			stats.MinElements = p.NumElements
		}
		if p.NumElements > stats.MaxElements {
			stats.MaxElements = p.NumElements
		}
	}

	stats.Imbalance = float64(stats.MaxElements) / stats.AvgElements

	return stats
}

type PartitionStats struct {
	NumPartitions int
	MinElements   int
	MaxElements   int
	AvgElements   float64
	Imbalance     float64 // MaxElements / AvgElements
}

// PartitionFaceBuffer extends PartitionBuffer with face-specific indexing
// This integrates the face buffer design pattern with partitions
type PartitionFaceBuffer struct {
	*PartitionBuffer

	// Face-level indexing (inspired by face buffer design)
	FaceIndex []int32 // [Nfaces × K] encoding for all faces in partition

	// Dimensions
	Nfp    int // Points per face
	Nfaces int // Faces per element
	K      int // Elements in this partition
}

// Face index encoding constants (from face buffer design)
const (
	BoundaryPlaceholder int32 = -999
	RemoteFace          int32 = -9999
)

// BuildPartitionFaceBuffers creates face-aware partition buffers
// This integrates the face buffer pattern with partition communication
func BuildPartitionFaceBuffers(layout *PartitionLayout, mesh *MeshData) ([]*PartitionFaceBuffer, error) {
	// First build standard partition buffers
	baseBuffers, err := BuildPartitionBuffers(layout, mesh)
	if err != nil {
		return nil, err
	}

	// Enhance with face indexing
	faceBuffers := make([]*PartitionFaceBuffer, len(baseBuffers))

	for partID, baseBuffer := range baseBuffers {
		partition := layout.Partitions[partID]

		// Create face index array
		faceIndex := make([]int32, mesh.Nfaces*partition.NumElements)

		// Initialize all faces as boundary
		for i := range faceIndex {
			faceIndex[i] = BoundaryPlaceholder
		}

		// Fill in interior and remote faces
		err := populateFaceIndex(partID, layout, mesh, faceIndex, baseBuffer)
		if err != nil {
			return nil, fmt.Errorf("partition %d: %w", partID, err)
		}

		faceBuffers[partID] = &PartitionFaceBuffer{
			PartitionBuffer: baseBuffer,
			FaceIndex:       faceIndex,
			Nfp:             mesh.Nfp,
			Nfaces:          mesh.Nfaces,
			K:               partition.NumElements,
		}
	}

	return faceBuffers, nil
}

// populateFaceIndex fills the face index array following face buffer conventions
func populateFaceIndex(partID int, layout *PartitionLayout, mesh *MeshData,
	faceIndex []int32, buffer *PartitionBuffer) error {

	partition := layout.Partitions[partID]

	// Process each element in partition
	for localElem := 0; localElem < partition.NumElements; localElem++ {
		globalElem := partition.Elements[localElem]

		// Process each face
		for face := 0; face < mesh.Nfaces; face++ {
			faceIdx := face + localElem*mesh.Nfaces
			neighbor := mesh.EToE[globalElem][face]

			// Case 1: Boundary face
			if neighbor == globalElem {
				faceIndex[faceIdx] = BoundaryPlaceholder
				continue
			}

			// Check neighbor's partition
			neighborPart := layout.GetPartition(neighbor)

			// Case 2: Interior face (same partition)
			if neighborPart == partID {
				// Find neighbor's position in local ordering
				neighborLocal := findLocalElement(neighbor, partition.Elements)
				if neighborLocal < 0 {
					return fmt.Errorf("neighbor element %d not found in partition", neighbor)
				}

				// Calculate P location in M buffer
				neighborFace := mesh.EToF[globalElem][face]
				pStart := neighborLocal*mesh.Nfaces*mesh.Nfp + neighborFace*mesh.Nfp
				faceIndex[faceIdx] = int32(pStart)

				// Case 3: Remote face (different partition)
			} else if neighborPart >= 0 {
				faceIndex[faceIdx] = RemoteFace
			}
		}
	}

	return nil
}

// findLocalElement finds element's local index within partition
func findLocalElement(globalElem int, elements []int) int {
	for i, elem := range elements {
		if elem == globalElem {
			return i
		}
	}
	return -1
}

// applyBCOverlay replaces boundary placeholders with BC codes
func applyBCOverlay(faceIndex []int32, bcData map[int]int32, nfaces int) error {
	for i, code := range faceIndex {
		if code == BoundaryPlaceholder {
			// Look up BC type for this face
			if bcType, ok := bcData[i]; ok {
				faceIndex[i] = -bcType // Negative values encode BC types
			} else {
				// Default BC if not specified
				faceIndex[i] = -2 // Default to outflow
			}
		}
	}
	return nil
}
