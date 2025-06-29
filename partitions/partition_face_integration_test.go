package partitions

import (
	"testing"
)

// TestBuildPartitionFaceBuffers_SinglePartition tests face buffer creation for single partition
func TestBuildPartitionFaceBuffers_SinglePartition(t *testing.T) {
	// Create a simple 4-element tetrahedral mesh
	mesh := &MeshData{
		EToE: [][]int{
			{0, 1, 2, 0}, // Element 0: neighbors on faces 0,1,2; boundary on face 3
			{1, 0, 3, 1}, // Element 1: neighbors on faces 0,1,2; boundary on face 3
			{2, 3, 0, 2}, // Element 2: neighbors on faces 0,1,2; boundary on face 3
			{3, 2, 1, 3}, // Element 3: neighbors on faces 0,1,2; boundary on face 3
		},
		EToF: [][]int{
			{0, 0, 0, 3}, // Element 0 face connections
			{0, 0, 0, 3}, // Element 1 face connections
			{0, 0, 0, 3}, // Element 2 face connections
			{0, 0, 0, 3}, // Element 3 face connections
		},
		Nfp:    6, // 6 points per face
		Nfaces: 4, // 4 faces per tetrahedron
	}

	// Create single partition containing all elements
	layout := &PartitionLayout{
		Partitions: []Partition{
			{
				ID:          0,
				Elements:    []int{0, 1, 2, 3},
				NumElements: 4,
				MaxElements: 4,
			},
		},
		KpartMax:      4,
		TotalElements: 4,
		NumPartitions: 1,
		EToP:          []int{0, 0, 0, 0},
	}

	// Build face buffers
	faceBuffers, err := BuildPartitionFaceBuffers(layout, mesh)
	if err != nil {
		t.Fatalf("Failed to build face buffers: %v", err)
	}

	if len(faceBuffers) != 1 {
		t.Fatalf("Expected 1 face buffer, got %d", len(faceBuffers))
	}

	fb := faceBuffers[0]

	// Verify dimensions
	if fb.K != 4 {
		t.Errorf("Expected K=4, got %d", fb.K)
	}
	if fb.Nfaces != 4 {
		t.Errorf("Expected Nfaces=4, got %d", fb.Nfaces)
	}
	if fb.Nfp != 6 {
		t.Errorf("Expected Nfp=6, got %d", fb.Nfp)
	}

	// Check face index array size
	expectedSize := 4 * 4 // K * Nfaces
	if len(fb.FaceIndex) != expectedSize {
		t.Errorf("Expected FaceIndex size %d, got %d", expectedSize, len(fb.FaceIndex))
	}

	// Verify face classifications
	boundaryCount := 0
	interiorCount := 0
	for i, code := range fb.FaceIndex {
		elem := i / 4
		face := i % 4
		neighbor := mesh.EToE[elem][face]

		if neighbor == elem {
			// Should be boundary
			if code != BoundaryPlaceholder {
				t.Errorf("Element %d face %d should be boundary, got code %d", elem, face, code)
			}
			boundaryCount++
		} else {
			// Should be interior (positive value)
			if code <= 0 {
				t.Errorf("Element %d face %d should be interior, got code %d", elem, face, code)
			}
			interiorCount++
		}
	}

	// Each element has 3 interior faces and 1 boundary
	if boundaryCount != 4 {
		t.Errorf("Expected 4 boundary faces, got %d", boundaryCount)
	}
	if interiorCount != 12 {
		t.Errorf("Expected 12 interior faces, got %d", interiorCount)
	}
}

// TestBuildPartitionFaceBuffers_TwoPartitions tests remote face detection
func TestBuildPartitionFaceBuffers_TwoPartitions(t *testing.T) {
	// Create a 2-element mesh where elements are in different partitions
	mesh := &MeshData{
		EToE: [][]int{
			{0, 1, 0, 0}, // Element 0: neighbor 1 on face 1, boundaries elsewhere
			{1, 0, 1, 1}, // Element 1: neighbor 0 on face 1, boundaries elsewhere
		},
		EToF: [][]int{
			{0, 1, 2, 3}, // Element 0 faces
			{1, 0, 2, 3}, // Element 1 faces
		},
		Nfp:    3, // 3 points per face (linear)
		Nfaces: 4, // 4 faces per tet
	}

	// Create two partitions, one element each
	layout := &PartitionLayout{
		Partitions: []Partition{
			{
				ID:          0,
				Elements:    []int{0},
				NumElements: 1,
				MaxElements: 1,
			},
			{
				ID:          1,
				Elements:    []int{1},
				NumElements: 1,
				MaxElements: 1,
			},
		},
		KpartMax:      1,
		TotalElements: 2,
		NumPartitions: 2,
		EToP:          []int{0, 1}, // Element 0 in partition 0, element 1 in partition 1
	}

	// Build face buffers
	faceBuffers, err := BuildPartitionFaceBuffers(layout, mesh)
	if err != nil {
		t.Fatalf("Failed to build face buffers: %v", err)
	}

	if len(faceBuffers) != 2 {
		t.Fatalf("Expected 2 face buffers, got %d", len(faceBuffers))
	}

	// Check partition 0
	fb0 := faceBuffers[0]
	remoteCount0 := 0
	for i, code := range fb0.FaceIndex {
		face := i % 4
		if face == 1 { // Face 1 should be remote
			if code != RemoteFace {
				t.Errorf("Partition 0 face 1 should be remote, got code %d", code)
			}
			remoteCount0++
		}
	}
	if remoteCount0 != 1 {
		t.Errorf("Partition 0 should have 1 remote face, got %d", remoteCount0)
	}

	// Check partition 1
	fb1 := faceBuffers[1]
	remoteCount1 := 0
	for i, code := range fb1.FaceIndex {
		face := i % 4
		if face == 1 { // Face 1 should be remote
			if code != RemoteFace {
				t.Errorf("Partition 1 face 1 should be remote, got code %d", code)
			}
			remoteCount1++
		}
	}
	if remoteCount1 != 1 {
		t.Errorf("Partition 1 should have 1 remote face, got %d", remoteCount1)
	}

	// Verify communication buffers
	if fb0.SendBufferSize != 3 { // 1 face * 3 points
		t.Errorf("Partition 0 send buffer size should be 3, got %d", fb0.SendBufferSize)
	}
	if fb1.RecvBufferSize != 3 {
		t.Errorf("Partition 1 recv buffer size should be 3, got %d", fb1.RecvBufferSize)
	}
}

// TestApplyBCOverlay tests boundary condition overlay phase
func TestApplyBCOverlay(t *testing.T) {
	// Create face index with boundary placeholders
	faceIndex := []int32{
		100, BoundaryPlaceholder, 200, BoundaryPlaceholder, // Element 0
		300, 400, BoundaryPlaceholder, RemoteFace, // Element 1
	}
	nfaces := 4

	// Define BC data
	bcData := map[int]int32{
		1: 1, // Element 0, face 1: Wall BC
		3: 2, // Element 0, face 3: Outflow BC
		6: 3, // Element 1, face 2: Inflow BC
	}

	// Apply BC overlay
	err := applyBCOverlay(faceIndex, bcData, nfaces)
	if err != nil {
		t.Fatalf("Failed to apply BC overlay: %v", err)
	}

	// Check results
	expected := []int32{
		100, -1, 200, -2, // Element 0: interior, wall, interior, outflow
		300, 400, -3, RemoteFace, // Element 1: interior, interior, inflow, remote
	}

	for i, exp := range expected {
		if faceIndex[i] != exp {
			t.Errorf("Face %d: expected %d, got %d", i, exp, faceIndex[i])
		}
	}
}

// TestFindLocalElement tests element lookup within partition
func TestFindLocalElement(t *testing.T) {
	elements := []int{5, 10, 15, 20, 25}

	tests := []struct {
		global   int
		expected int
	}{
		{5, 0},
		{10, 1},
		{15, 2},
		{20, 3},
		{25, 4},
		{30, -1}, // Not found
	}

	for _, test := range tests {
		result := findLocalElement(test.global, elements)
		if result != test.expected {
			t.Errorf("findLocalElement(%d): expected %d, got %d",
				test.global, test.expected, result)
		}
	}
}

// TestPopulateFaceIndex tests face index population logic
func TestPopulateFaceIndex(t *testing.T) {
	// 3-element mesh: 0-1 connected, 2 isolated
	mesh := &MeshData{
		EToE: [][]int{
			{0, 1, 0, 0}, // Element 0
			{1, 0, 1, 1}, // Element 1
			{2, 2, 2, 2}, // Element 2 (all boundaries)
		},
		EToF: [][]int{
			{0, 0, 2, 3},
			{1, 1, 2, 3},
			{0, 1, 2, 3},
		},
		Nfp:    4,
		Nfaces: 4,
	}

	layout := &PartitionLayout{
		Partitions: []Partition{
			{
				ID:          0,
				Elements:    []int{0, 1, 2},
				NumElements: 3,
				MaxElements: 3,
			},
		},
		EToP: []int{0, 0, 0},
	}

	faceIndex := make([]int32, 12) // 3 elements * 4 faces
	for i := range faceIndex {
		faceIndex[i] = BoundaryPlaceholder
	}

	buffer := &PartitionBuffer{}

	err := populateFaceIndex(0, layout, mesh, faceIndex, buffer)
	if err != nil {
		t.Fatalf("Failed to populate face index: %v", err)
	}

	// Check element 0, face 1 (connects to element 1, face 1)
	elem0face1 := faceIndex[1]
	expectedP := int32(1*4*4 + 1*4) // element 1, face 1 start in M buffer
	if elem0face1 != expectedP {
		t.Errorf("Element 0 face 1: expected P index %d, got %d", expectedP, elem0face1)
	}

	// Check element 2 (all boundaries)
	for face := 0; face < 4; face++ {
		idx := 2*4 + face
		if faceIndex[idx] != BoundaryPlaceholder {
			t.Errorf("Element 2 face %d should be boundary, got %d", face, faceIndex[idx])
		}
	}
}

// TestCommunicationSymmetry tests that send/receive patterns match
func TestCommunicationSymmetry(t *testing.T) {
	// Create buffers with asymmetric communication (should fail)
	buffers := []*PartitionBuffer{
		{
			RemotePartitions: []RemotePartition{
				{PartitionID: 1, SendCount: 10, RecvCount: 10},
			},
		},
		{
			RemotePartitions: []RemotePartition{
				{PartitionID: 0, SendCount: 5, RecvCount: 10}, // Mismatch!
			},
		},
	}

	err := validateCommunicationSymmetry(buffers)
	if err == nil {
		t.Error("Expected validation to fail for asymmetric communication")
	}

	// Fix the asymmetry
	buffers[1].RemotePartitions[0].SendCount = 10
	buffers[1].RemotePartitions[0].RecvCount = 5

	err = validateCommunicationSymmetry(buffers)
	if err != nil {
		t.Errorf("Expected validation to pass for symmetric communication: %v", err)
	}
}

// Benchmark face buffer traversal pattern
func BenchmarkFaceTraversal(b *testing.B) {
	// Create a reasonably sized partition
	K := 1000
	Nfaces := 4
	Nfp := 10

	faceIndex := make([]int32, K*Nfaces)
	M := make([]float64, K*Nfaces*Nfp)
	flux := make([]float64, K*Nfaces*Nfp)

	// Mix of face types
	for i := range faceIndex {
		switch i % 4 {
		case 0:
			faceIndex[i] = int32(i * Nfp) // Interior
		case 1:
			faceIndex[i] = -1 // Wall BC
		case 2:
			faceIndex[i] = RemoteFace
		case 3:
			faceIndex[i] = -2 // Outflow BC
		}
	}

	// Initialize data
	for i := range M {
		M[i] = float64(i)
	}

	recvBuffer := make([]float64, K*Nfp) // Simplified: assume 1 remote face per element
	for i := range recvBuffer {
		recvBuffer[i] = float64(i) * 0.5
	}

	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		remoteCounter := 0

		// Traverse elements
		for elem := 0; elem < K; elem++ {
			// Traverse faces
			for face := 0; face < Nfaces; face++ {
				faceCode := faceIndex[face+elem*Nfaces]

				// Traverse points
				for p := 0; p < Nfp; p++ {
					mIdx := elem*Nfaces*Nfp + face*Nfp + p
					mVal := M[mIdx]

					var pVal float64
					if faceCode > 0 {
						// Interior face
						pVal = M[faceCode+int32(p)]
					} else if faceCode == RemoteFace {
						// Remote face
						pVal = recvBuffer[remoteCounter+p]
					} else {
						// Boundary: simple BC
						pVal = -mVal // Wall reflection
					}

					// Simple flux computation
					flux[mIdx] = 0.5 * (mVal + pVal)
				}

				// Advance remote counter if needed
				if faceCode == RemoteFace {
					remoteCounter += Nfp
				}
			}
		}
	}
}
