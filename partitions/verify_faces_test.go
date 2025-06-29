package partitions

import (
	"testing"
)

// TestVerifyMeshConfiguration tests our understanding of the mesh structure
func TestVerifyMeshConfiguration(t *testing.T) {
	// Let's verify what the mesh configuration actually produces
	mesh := &MeshData{
		EToE: [][]int{
			{0, 1, 2, 0}, // Element 0
			{1, 0, 3, 1}, // Element 1
			{2, 3, 0, 2}, // Element 2
			{3, 2, 1, 3}, // Element 3
		},
		EToF: [][]int{
			{0, 1, 2, 3},
			{0, 1, 2, 3},
			{0, 1, 2, 3},
			{0, 1, 2, 3},
		},
		Nfp:    6,
		Nfaces: 4,
	}

	// Count boundaries manually
	boundaryCount := 0
	interiorCount := 0

	for elem := 0; elem < 4; elem++ {
		for face := 0; face < 4; face++ {
			neighbor := mesh.EToE[elem][face]
			if neighbor == elem {
				boundaryCount++
				t.Logf("Element %d, face %d: BOUNDARY", elem, face)
			} else {
				interiorCount++
				t.Logf("Element %d, face %d: connects to element %d", elem, face, neighbor)
			}
		}
	}

	t.Logf("Total boundaries: %d", boundaryCount)
	t.Logf("Total interior: %d", interiorCount)

	// Based on the EToE array:
	// Element 0: faces 0,3 are boundaries (self-reference)
	// Element 1: faces 0,3 are boundaries
	// Element 2: faces 0,3 are boundaries
	// Element 3: faces 0,3 are boundaries
	// Total: 8 boundaries, 8 interior faces

	if boundaryCount != 8 {
		t.Errorf("Expected 8 boundary faces based on EToE, got %d", boundaryCount)
	}
	if interiorCount != 8 {
		t.Errorf("Expected 8 interior faces based on EToE, got %d", interiorCount)
	}
}
