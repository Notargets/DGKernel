package utils

import (
	"fmt"
	"testing"
)

// Helper function to build connectivity arrays (EToE, EToF) from mesh
func buildConnectivity(K int, EToV [][]int) (EToE, EToF [][]int) {
	// Initialize with self-connections
	EToE = make([][]int, K)
	EToF = make([][]int, K)
	for e := 0; e < K; e++ {
		EToE[e] = make([]int, 4)
		EToF[e] = make([]int, 4)
		for f := 0; f < 4; f++ {
			EToE[e][f] = e // Self-connection by default
			EToF[e][f] = f
		}
	}

	// Face definitions for tetrahedron (which 3 vertices form each face)
	faceVertices := [][]int{
		{0, 1, 2}, // Face 0
		{0, 1, 3}, // Face 1
		{1, 2, 3}, // Face 2
		{0, 2, 3}, // Face 3
	}

	// Build face signatures for all elements
	type faceSignature struct {
		v1, v2, v3 int
		elem       int
		face       int
	}

	faceMap := make(map[string]faceSignature)

	// Process each element's faces
	for e := 0; e < K; e++ {
		for f := 0; f < 4; f++ {
			// Get the three vertices of this face
			v := make([]int, 3)
			for i := 0; i < 3; i++ {
				v[i] = EToV[e][faceVertices[f][i]]
			}

			// Sort vertices to create canonical face signature
			if v[0] > v[1] {
				v[0], v[1] = v[1], v[0]
			}
			if v[1] > v[2] {
				v[1], v[2] = v[2], v[1]
			}
			if v[0] > v[1] {
				v[0], v[1] = v[1], v[0]
			}

			key := fmt.Sprintf("%d-%d-%d", v[0], v[1], v[2])

			// Check if this face already exists
			if existing, found := faceMap[key]; found {
				// Found matching face - connect them
				EToE[e][f] = existing.elem
				EToF[e][f] = existing.face
				EToE[existing.elem][existing.face] = e
				EToF[existing.elem][existing.face] = f
			} else {
				// New face - add to map
				faceMap[key] = faceSignature{v[0], v[1], v[2], e, f}
			}
		}
	}

	return EToE, EToF
}

// Helper function to build VmapP from connectivity
func buildVmapP(K, Nfaces, Nfp, Np int, EToE, EToF [][]int) []int {
	VmapP := make([]int, K*Nfaces*Nfp)

	// For each face point
	for e := 0; e < K; e++ {
		for f := 0; f < Nfaces; f++ {
			neighborElem := EToE[e][f]
			neighborFace := EToF[e][f]

			for fp := 0; fp < Nfp; fp++ {
				idx := e*Nfaces*Nfp + f*Nfp + fp

				// Map to corresponding node in neighbor element
				// For simplicity, use a direct mapping scheme
				// In reality, this would depend on face orientation
				neighborNode := neighborElem*Np + (neighborFace*Nfp+fp)%Np
				VmapP[idx] = neighborNode
			}
		}
	}

	return VmapP
}

// Helper function to simulate face data exchange
func simulateFaceExchange(fc *FaceConnector, solutions [][]float64) ([][]float64, error) {
	// Create buffers
	pickBuffers := make([][][]float64, fc.NumPartitions)
	placeBuffers := make([][][]float64, fc.NumPartitions)
	pBuffers := make([][]float64, fc.NumPartitions)

	// Initialize buffers
	for p := 0; p < fc.NumPartitions; p++ {
		pickBuffers[p] = make([][]float64, fc.NumPartitions)
		placeBuffers[p] = make([][]float64, fc.NumPartitions)
		pBuffers[p] = make([]float64, fc.ElemsPerPartition[p]*fc.Nfaces*fc.Nfp)

		for q := 0; q < fc.NumPartitions; q++ {
			pickSize := len(fc.GetPickIndices(p, q))
			pickBuffers[p][q] = make([]float64, pickSize)
		}
	}

	// Phase 1: Pick - gather from solution using pick indices
	for p := 0; p < fc.NumPartitions; p++ {
		for q := 0; q < fc.NumPartitions; q++ {
			pickIndices := fc.GetPickIndices(p, q)
			for i, idx := range pickIndices {
				if idx >= len(solutions[p]) {
					return nil, fmt.Errorf("pick index %d out of bounds for partition %d", idx, p)
				}
				pickBuffers[p][q][i] = solutions[p][idx]
			}
		}
	}

	// Phase 2: Exchange - copy pick buffers to place buffers
	for p := 0; p < fc.NumPartitions; p++ {
		for q := 0; q < fc.NumPartitions; q++ {
			placeBuffers[q][p] = pickBuffers[p][q]
		}
	}

	// Phase 3: Place - scatter into P buffers using place indices
	for p := 0; p < fc.NumPartitions; p++ {
		for q := 0; q < fc.NumPartitions; q++ {
			placeIndices := fc.GetPlaceIndices(p, q)
			for i, idx := range placeIndices {
				if idx >= len(pBuffers[p]) {
					return nil, fmt.Errorf("place index %d out of bounds for partition %d P buffer", idx, p)
				}
				pBuffers[p][idx] = placeBuffers[p][q][i]
			}
		}
	}

	return pBuffers, nil
}

// Helper to convert mesh to EToV format
func meshToEToV(mesh CompleteMesh) [][]int {
	var EToV [][]int

	for _, elemSet := range mesh.Elements {
		if elemSet.Type == Tet {
			for _, elem := range elemSet.Elements {
				tetConn := make([]int, 4)
				for i, nodeName := range elem {
					tetConn[i] = mesh.Nodes.NodeMap[nodeName]
				}
				EToV = append(EToV, tetConn)
			}
		}
	}

	return EToV
}

// TestFaceConnector_CubeMeshUnpartitioned tests connectivity for unpartitioned cube mesh
// Verifies that face exchanges work correctly when all elements are in one partition
func TestFaceConnector_CubeMeshUnpartitioned(t *testing.T) {
	tm := GetStandardTestMeshes()

	// Convert mesh to connectivity arrays
	EToV := meshToEToV(tm.CubeMesh)
	K := len(EToV)
	Nfaces := 4
	Nfp := 3
	Np := 4

	// Build connectivity
	EToE, EToF := buildConnectivity(K, EToV)

	// Build VmapP
	VmapP := buildVmapP(K, Nfaces, Nfp, Np, EToE, EToF)

	// All elements in partition 0
	EToP := make([]int, K)
	for i := range EToP {
		EToP[i] = 0
	}

	// Create FaceConnector
	fc, err := NewFaceConnector(K, Nfaces, Nfp, Np, VmapP, EToP)
	if err != nil {
		t.Fatalf("Failed to create FaceConnector: %v", err)
	}

	// Create solution where each node value equals its element ID
	solution := make([][]float64, 1) // One partition
	solution[0] = make([]float64, K*Np)
	for e := 0; e < K; e++ {
		for n := 0; n < Np; n++ {
			solution[0][e*Np+n] = float64(e)
		}
	}

	// Simulate face exchange
	pBuffers, err := simulateFaceExchange(fc, solution)
	if err != nil {
		t.Fatalf("Failed to simulate face exchange: %v", err)
	}

	// Verify results
	t.Run("BoundaryFaces", func(t *testing.T) {
		// Count boundary faces (should be 12 for a cube = 6 faces * 2 triangles)
		boundaryCount := 0
		for e := 0; e < K; e++ {
			for f := 0; f < Nfaces; f++ {
				if EToE[e][f] == e {
					boundaryCount++

					// Verify boundary face points have their own element ID
					for fp := 0; fp < Nfp; fp++ {
						idx := e*Nfaces*Nfp + f*Nfp + fp
						expected := float64(e)
						actual := pBuffers[0][idx]
						if actual != expected {
							t.Errorf("Boundary face point [elem %d, face %d, fp %d]: expected %f, got %f",
								e, f, fp, expected, actual)
						}
					}
				}
			}
		}

		if boundaryCount != 12 {
			t.Errorf("Expected 12 boundary faces, got %d", boundaryCount)
		}
	})

	t.Run("InteriorFaces", func(t *testing.T) {
		// Verify interior faces have neighbor element IDs
		interiorCount := 0
		for e := 0; e < K; e++ {
			for f := 0; f < Nfaces; f++ {
				neighborElem := EToE[e][f]
				if neighborElem != e {
					interiorCount++

					// Verify face points have neighbor element ID
					for fp := 0; fp < Nfp; fp++ {
						idx := e*Nfaces*Nfp + f*Nfp + fp
						expected := float64(neighborElem)
						actual := pBuffers[0][idx]
						if actual != expected {
							t.Errorf("Interior face point [elem %d→%d, face %d, fp %d]: expected %f, got %f",
								e, neighborElem, f, fp, expected, actual)
						}
					}
				}
			}
		}

		// Each interior face is counted twice (once from each side)
		t.Logf("Found %d interior face connections", interiorCount)

		if interiorCount == 0 {
			t.Error("No interior faces found - cube mesh should have interior connections")
		}
	})

	// Verify conservation
	if err := fc.Verify(); err != nil {
		t.Errorf("Verification failed: %v", err)
	}
}

// TestFaceConnector_CubeMeshPartitioned tests connectivity for partitioned cube mesh
// Verifies cross-partition face exchanges work correctly
func TestFaceConnector_CubeMeshPartitioned(t *testing.T) {
	tm := GetStandardTestMeshes()

	// Convert mesh to connectivity arrays
	EToV := meshToEToV(tm.CubeMesh)
	K := len(EToV)
	Nfaces := 4
	Nfp := 3
	Np := 4

	// Build connectivity
	EToE, EToF := buildConnectivity(K, EToV)

	// Build VmapP
	VmapP := buildVmapP(K, Nfaces, Nfp, Np, EToE, EToF)

	// Partition elements: 0,1,2 in partition 0; 3,4,5 in partition 1
	EToP := []int{0, 0, 0, 1, 1, 1}

	// Create FaceConnector
	fc, err := NewFaceConnector(K, Nfaces, Nfp, Np, VmapP, EToP)
	if err != nil {
		t.Fatalf("Failed to create FaceConnector: %v", err)
	}

	// Create solution where each node value equals its global element ID
	solutions := make([][]float64, 2)    // Two partitions
	solutions[0] = make([]float64, 3*Np) // Partition 0: elements 0,1,2
	solutions[1] = make([]float64, 3*Np) // Partition 1: elements 3,4,5

	// Fill with global element IDs
	for e := 0; e < 3; e++ {
		for n := 0; n < Np; n++ {
			solutions[0][e*Np+n] = float64(e)     // Elements 0,1,2
			solutions[1][e*Np+n] = float64(e + 3) // Elements 3,4,5
		}
	}

	// Simulate face exchange
	pBuffers, err := simulateFaceExchange(fc, solutions)
	if err != nil {
		t.Fatalf("Failed to simulate face exchange: %v", err)
	}

	// Verify results for each partition
	for p := 0; p < 2; p++ {
		t.Run(fmt.Sprintf("Partition%d", p), func(t *testing.T) {
			// Check each element in this partition
			for localElem := 0; localElem < 3; localElem++ {
				globalElem := fc.LocalToGlobalElem[p][localElem]

				for f := 0; f < Nfaces; f++ {
					neighborElem := EToE[globalElem][f]

					// Check all face points
					for fp := 0; fp < Nfp; fp++ {
						idx := localElem*Nfaces*Nfp + f*Nfp + fp
						expected := float64(neighborElem)
						actual := pBuffers[p][idx]

						if actual != expected {
							neighborPart := EToP[neighborElem]
							t.Errorf("Face point [part %d, elem %d(g%d)→%d(p%d), face %d, fp %d]: expected %f, got %f",
								p, localElem, globalElem, neighborElem, neighborPart, f, fp, expected, actual)
						}
					}
				}
			}
		})
	}

	// Test specific cross-partition connections
	t.Run("CrossPartitionExchange", func(t *testing.T) {
		// Count cross-partition faces
		crossPartCount := 0
		for e := 0; e < K; e++ {
			for f := 0; f < Nfaces; f++ {
				neighbor := EToE[e][f]
				if neighbor != e && EToP[e] != EToP[neighbor] {
					crossPartCount++
				}
			}
		}

		t.Logf("Found %d cross-partition face connections", crossPartCount)

		if crossPartCount == 0 {
			t.Error("No cross-partition faces found - partitioned mesh should have cross-partition connections")
		}

		// Verify pick/place indices for cross-partition communication
		pick01 := fc.GetPickIndices(0, 1)
		pick10 := fc.GetPickIndices(1, 0)

		if len(pick01) == 0 {
			t.Error("No pick indices from partition 0 to 1")
		}
		if len(pick10) == 0 {
			t.Error("No pick indices from partition 1 to 0")
		}

		t.Logf("Partition 0→1: %d values, Partition 1→0: %d values", len(pick01), len(pick10))
	})

	// Verify conservation
	if err := fc.Verify(); err != nil {
		t.Errorf("Verification failed: %v", err)
	}
}

// TestFaceConnector_VerifyConnectivityMath tests mathematical properties of connectivity
func TestFaceConnector_VerifyConnectivityMath(t *testing.T) {
	tm := GetStandardTestMeshes()

	// Use a simple two-tet mesh for clear verification
	EToV := meshToEToV(tm.TwoTetMesh)
	K := len(EToV)
	Nfaces := 4
	Nfp := 3
	Np := 4

	// Build connectivity
	EToE, EToF := buildConnectivity(K, EToV)

	// Build VmapP
	VmapP := buildVmapP(K, Nfaces, Nfp, Np, EToE, EToF)

	// Both elements in same partition for simplicity
	EToP := []int{0, 0}

	// Create FaceConnector
	_, err := NewFaceConnector(K, Nfaces, Nfp, Np, VmapP, EToP)
	if err != nil {
		t.Fatalf("Failed to create FaceConnector: %v", err)
	}

	// Test 1: Reciprocity of connections
	t.Run("Reciprocity", func(t *testing.T) {
		// If element 0 face 2 connects to element 1 face 1,
		// then element 1 face 1 must connect to element 0 face 2
		for e := 0; e < K; e++ {
			for f := 0; f < Nfaces; f++ {
				neighbor := EToE[e][f]
				neighborFace := EToF[e][f]

				// Check reverse connection
				if EToE[neighbor][neighborFace] != e {
					t.Errorf("Reciprocity broken: elem %d face %d → elem %d face %d, but reverse points to elem %d",
						e, f, neighbor, neighborFace, EToE[neighbor][neighborFace])
				}
				if EToF[neighbor][neighborFace] != f {
					t.Errorf("Reciprocity broken: elem %d face %d → elem %d face %d, but reverse points to face %d",
						e, f, neighbor, neighborFace, EToF[neighbor][neighborFace])
				}
			}
		}
	})

	// Test 2: Each element has exactly 4 faces
	t.Run("FaceCount", func(t *testing.T) {
		for e := 0; e < K; e++ {
			if len(EToE[e]) != 4 || len(EToF[e]) != 4 {
				t.Errorf("Element %d: wrong face count", e)
			}
		}
	})

	// Test 3: Face indices are valid
	t.Run("ValidFaceIndices", func(t *testing.T) {
		for e := 0; e < K; e++ {
			for f := 0; f < Nfaces; f++ {
				if EToF[e][f] < 0 || EToF[e][f] >= Nfaces {
					t.Errorf("Invalid face index: elem %d face %d → face %d", e, f, EToF[e][f])
				}
			}
		}
	})
}
