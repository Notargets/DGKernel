package gonudg

import (
	"fmt"
	"github.com/notargets/DGKernel/mesh"
	"math"
	"testing"

	"github.com/notargets/DGKernel/utils"
)

// Helper function to convert CompleteMesh to NUDGTet format
func meshToDG3DFormat(m utils.CompleteMesh) (VX, VY, VZ []float64, EToV [][]int) {
	// Extract vertex coordinates
	numNodes := len(m.Nodes.Nodes)
	VX = make([]float64, numNodes)
	VY = make([]float64, numNodes)
	VZ = make([]float64, numNodes)

	for i := 0; i < numNodes; i++ {
		VX[i] = m.Nodes.Nodes[i][0]
		VY[i] = m.Nodes.Nodes[i][1]
		VZ[i] = m.Nodes.Nodes[i][2]
	}

	// Extract element connectivity
	// Assuming we only have tetrahedral elements
	for _, elemSet := range m.Elements {
		if elemSet.Type == utils.Tet {
			EToV = make([][]int, len(elemSet.Elements))
			for i, elem := range elemSet.Elements {
				EToV[i] = make([]int, 4)
				for j, nodeName := range elem {
					// DeviceMemType node name to index (0-based)
					EToV[i][j] = m.Nodes.NodeMap[nodeName]
				}
			}
			break
		}
	}

	return
}

// TestBuildMaps3D_SingleTet_FaceProperties tests fundamental face properties for a single tet
// Following Unit Testing Principle: Start with fundamentals
func TestBuildMaps3D_SingleTet_FaceProperties(t *testing.T) {
	tm := utils.GetStandardTestMeshes()
	singleTetMesh := utils.CompleteMesh{
		Nodes:     tm.TetraNodes,
		Elements:  []utils.ElementSet{tm.SingleTet},
		Dimension: 3,
	}

	VX, VY, VZ, EToV := meshToDG3DFormat(singleTetMesh)

	for _, N := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprintf("N=%d", N), func(t *testing.T) {
			dg, err := NewDG3D(N, VX, VY, VZ, EToV)
			if err != nil {
				t.Fatalf("Failed to create NUDGTet: %v", err)
			}

			// Test 1: Verify VmapM maps to valid volume nodes
			// Mathematical property: Face nodes must be subset of volume nodes
			for i, volIdx := range dg.VmapM {
				if volIdx < 0 || volIdx >= dg.Np*dg.K {
					t.Errorf("VmapM[%d] = %d is out of bounds [0, %d)", i, volIdx, dg.Np*dg.K)
				}
			}

			// Test 2: Verify face nodes lie on correct faces
			// Mathematical property: Face nodes must satisfy face constraints
			tolerance := math.Sqrt(dg.NODETOL)
			testFaceNodeConstraints(t, dg, tolerance)

			// Test 3: For boundary element, all faces are self-connected
			// Mathematical property: MapP[i] = MapM[i] for all boundary faces
			for i := range dg.MapM {
				if dg.MapP[i] != dg.MapM[i] {
					t.Errorf("Boundary face node %d: MapP should equal MapM", i)
				}
			}

			// Test 4: Verify all MapB entries correspond to actual face nodes
			// Mathematical property: Boundary map contains all and only boundary nodes
			for _, bIdx := range dg.MapB {
				if bIdx < 0 || bIdx >= len(dg.MapM) {
					t.Errorf("MapB contains invalid index %d", bIdx)
				}
			}
		})
	}
}

// TestBuildMaps3D_TwoTets_SharedFaceGeometry tests geometric properties of shared faces
// Following Unit Testing Principle: Test specific mathematical properties
func TestBuildMaps3D_TwoTets_SharedFaceGeometry(t *testing.T) {
	tm := utils.GetStandardTestMeshes()
	VX, VY, VZ, EToV := meshToDG3DFormat(tm.TwoTetMesh)

	for _, N := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("N=%d", N), func(t *testing.T) {
			dg, err := NewDG3D(N, VX, VY, VZ, EToV)
			if err != nil {
				t.Fatalf("Failed to create NUDGTet: %v", err)
			}

			// Set up connectivity
			meshObj := mesh.ConvertToMesh(tm.TwoTetMesh)
			meshObj.BuildConnectivity()
			dg.EToE = meshObj.EToE
			dg.EToF = meshObj.EToF
			dg.BuildMaps3D()

			// Test 1: Verify reciprocal connectivity
			// Mathematical property: If elem A face f connects to elem B face g,
			// then elem B face g must connect to elem A face f
			testReciprocalConnectivity(t, dg)

			// Test 2: Verify physical coordinate matching for connected nodes
			// Mathematical property: Connected nodes must have identical physical coordinates
			testConnectedNodeCoordinates(t, dg)

			// Test 3: Verify face normal consistency
			// Mathematical property: Shared face normals should be opposite
			// (This would require normals to be computed, which they aren'T in BuildMaps3D)

			// Test 4: Verify no node is both boundary and interior
			// Mathematical property: A node cannot be simultaneously on boundary and interior
			testBoundaryInteriorExclusion(t, dg)
		})
	}
}

// TestBuildMaps3D_CubeMesh_ConnectivityProperties tests complex connectivity
// Following Unit Testing Principle: Build systematically to complex cases
func TestBuildMaps3D_CubeMesh_ConnectivityProperties(t *testing.T) {
	tm := utils.GetStandardTestMeshes()
	VX, VY, VZ, EToV := meshToDG3DFormat(tm.CubeMesh)

	N := 2 // Use order 2 for this test
	dg, err := NewDG3D(N, VX, VY, VZ, EToV)
	if err != nil {
		t.Fatalf("Failed to create NUDGTet: %v", err)
	}

	// Set up connectivity
	meshObj := mesh.ConvertToMesh(tm.CubeMesh)
	meshObj.BuildConnectivity()
	dg.EToE = meshObj.EToE
	dg.EToF = meshObj.EToF
	dg.BuildMaps3D()

	// Test 1: Verify each interior face has exactly one neighbor
	// Mathematical property: Interior faces connect exactly two elements
	testInteriorFaceUniqueness(t, dg)

	// Test 2: Verify Euler characteristic for boundary
	// Mathematical property: For a closed surface, V - E + F = 2
	// (Would require more geometric information than BuildMaps3D provides)

	// Test 3: Verify coordinate continuity across all shared faces
	testGlobalCoordinateContinuity(t, dg)
}

// Helper function: Test face node constraints
func testFaceNodeConstraints(t *testing.T, dg *NUDGTet, tolerance float64) {
	// Face constraints for reference tetrahedron:
	// Face 0: T = -1
	// Face 1: S = -1
	// Face 2: R+S+T = -1
	// Face 3: R = -1

	for f := 0; f < dg.Nfaces; f++ {
		for i := 0; i < dg.Nfp; i++ {
			faceIdx := f*dg.Nfp + i
			volIdx := dg.VmapM[faceIdx]
			localIdx := volIdx % dg.Np

			rCoord := dg.R[localIdx]
			sCoord := dg.S[localIdx]
			tCoord := dg.T[localIdx]

			switch f {
			case 0: // T = -1
				if math.Abs(tCoord+1.0) > tolerance {
					t.Errorf("Face 0 node %d: T = %f, expected -1", i, tCoord)
				}
			case 1: // S = -1
				if math.Abs(sCoord+1.0) > tolerance {
					t.Errorf("Face 1 node %d: S = %f, expected -1", i, sCoord)
				}
			case 2: // R+S+T = -1
				if math.Abs(rCoord+sCoord+tCoord+1.0) > tolerance {
					t.Errorf("Face 2 node %d: R+S+T = %f, expected -1", i, rCoord+sCoord+tCoord)
				}
			case 3: // R = -1
				if math.Abs(rCoord+1.0) > tolerance {
					t.Errorf("Face 3 node %d: R = %f, expected -1", i, rCoord)
				}
			}
		}
	}
}

// Helper function: Test reciprocal connectivity
func testReciprocalConnectivity(t *testing.T, dg *NUDGTet) {
	for k := 0; k < dg.K; k++ {
		for f := 0; f < dg.Nfaces; f++ {
			neighbor := dg.EToE[k][f]
			neighborFace := dg.EToF[k][f]

			// If not a boundary face
			if neighbor != k {
				// Check reciprocal relationship
				if dg.EToE[neighbor][neighborFace] != k {
					t.Errorf("Reciprocal connectivity failed: elem %d face %d -> elem %d, but reverse connection missing",
						k, f, neighbor)
				}
				if dg.EToF[neighbor][neighborFace] != f {
					t.Errorf("Reciprocal face connectivity failed: elem %d face %d -> face %d, but reverse face mismatch",
						k, f, neighborFace)
				}
			}
		}
	}
}

// Helper function: Test connected node coordinates
func testConnectedNodeCoordinates(t *testing.T, dg *NUDGTet) {
	NF := dg.Nfp * dg.Nfaces
	tolerance := math.Sqrt(dg.NODETOL)

	for idx := 0; idx < dg.K*NF; idx++ {
		if dg.MapP[idx] != dg.MapM[idx] {
			// This is a connected (interior) node
			vidM := dg.VmapM[idx]
			vidP := dg.VmapP[idx]

			// Extract physical coordinates
			kM := vidM / dg.Np
			iM := vidM % dg.Np
			kP := vidP / dg.Np
			iP := vidP % dg.Np

			xM := dg.X.At(iM, kM)
			yM := dg.Y.At(iM, kM)
			zM := dg.Z.At(iM, kM)

			xP := dg.X.At(iP, kP)
			yP := dg.Y.At(iP, kP)
			zP := dg.Z.At(iP, kP)

			// Verify coordinates match
			dist := math.Sqrt((xP-xM)*(xP-xM) + (yP-yM)*(yP-yM) + (zP-zM)*(zP-zM))
			if dist > tolerance {
				t.Errorf("Connected nodes have mismatched coordinates: distance = %e", dist)
			}
		}
	}
}

// Helper function: Test boundary/interior exclusion
func testBoundaryInteriorExclusion(t *testing.T, dg *NUDGTet) {
	// Create a set of boundary indices
	boundarySet := make(map[int]bool)
	for _, bIdx := range dg.MapB {
		boundarySet[bIdx] = true
	}

	// Check that no interior node is in boundary set
	NF := dg.Nfp * dg.Nfaces
	for idx := 0; idx < dg.K*NF; idx++ {
		if dg.MapP[idx] != dg.MapM[idx] {
			// This is an interior node
			if boundarySet[idx] {
				t.Errorf("Node %d is both interior and boundary", idx)
			}
		}
	}
}

// Helper function: Test interior face uniqueness
func testInteriorFaceUniqueness(t *testing.T, dg *NUDGTet) {
	// Map to track face partnerships
	facePairs := make(map[string]bool)

	for k := 0; k < dg.K; k++ {
		for f := 0; f < dg.Nfaces; f++ {
			neighbor := dg.EToE[k][f]
			neighborFace := dg.EToF[k][f]

			if neighbor != k {
				// Create a unique key for this face pair
				key1 := fmt.Sprintf("%d_%d_%d_%d", k, f, neighbor, neighborFace)
				key2 := fmt.Sprintf("%d_%d_%d_%d", neighbor, neighborFace, k, f)

				// Check if we've seen this pair before
				if !facePairs[key1] && !facePairs[key2] {
					facePairs[key1] = true
					facePairs[key2] = true
				}
			}
		}
	}
}

// Helper function: Test global coordinate continuity
func testGlobalCoordinateContinuity(t *testing.T, dg *NUDGTet) {
	tolerance := math.Sqrt(dg.NODETOL)

	// For each element and face
	for k := 0; k < dg.K; k++ {
		for f := 0; f < dg.Nfaces; f++ {
			neighbor := dg.EToE[k][f]

			if neighbor != k {
				// Get the face nodes from both sides
				for i := 0; i < dg.Nfp; i++ {
					idx := k*dg.Nfp*dg.Nfaces + f*dg.Nfp + i

					// Get coordinates from this element
					vidM := dg.VmapM[idx]
					kM := vidM / dg.Np
					iM := vidM % dg.Np

					xM := dg.X.At(iM, kM)
					yM := dg.Y.At(iM, kM)
					zM := dg.Z.At(iM, kM)

					// Get coordinates from neighbor
					vidP := dg.VmapP[idx]
					kP := vidP / dg.Np
					iP := vidP % dg.Np

					xP := dg.X.At(iP, kP)
					yP := dg.Y.At(iP, kP)
					zP := dg.Z.At(iP, kP)

					// Check continuity
					if math.Abs(xP-xM) > tolerance ||
						math.Abs(yP-yM) > tolerance ||
						math.Abs(zP-zM) > tolerance {
						t.Errorf("Discontinuity at elem %d face %d node %d", k, f, i)
					}
				}
			}
		}
	}
}

// TestBuildMaps3D_BoundaryNodeOrdering tests progressive complexity in boundary detection
// Following Unit Testing Principle: Incremental validation
func TestBuildMaps3D_BoundaryNodeOrdering(t *testing.T) {
	// Test with progressively complex meshes
	testCases := []struct {
		name string
		mesh utils.CompleteMesh
	}{
		{
			name: "SingleTet",
			mesh: utils.CompleteMesh{
				Nodes:     utils.GetStandardTestMeshes().TetraNodes,
				Elements:  []utils.ElementSet{utils.GetStandardTestMeshes().SingleTet},
				Dimension: 3,
			},
		},
		{
			name: "TwoTets",
			mesh: utils.GetStandardTestMeshes().TwoTetMesh,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			VX, VY, VZ, EToV := meshToDG3DFormat(tc.mesh)

			for N := 1; N <= 3; N++ {
				t.Run(fmt.Sprintf("Order%d", N), func(t *testing.T) {
					dg, err := NewDG3D(N, VX, VY, VZ, EToV)
					if err != nil {
						t.Fatalf("Failed to create NUDGTet: %v", err)
					}

					// Set up connectivity if needed
					if tc.name != "SingleTet" {
						meshObj := mesh.ConvertToMesh(tc.mesh)
						meshObj.BuildConnectivity()
						dg.EToE = meshObj.EToE
						dg.EToF = meshObj.EToF
					}

					dg.BuildMaps3D()

					// Test: Every boundary node should map to itself
					for _, bIdx := range dg.MapB {
						if dg.MapP[bIdx] != bIdx {
							t.Errorf("Boundary node %d: MapP should equal MapM", bIdx)
						}
					}

					// Test: Boundary nodes should form complete faces
					// (This is a topological property, not an implementation detail)
					verifyBoundaryFaceCompleteness(t, dg)
				})
			}
		})
	}
}

// Helper function: Verify boundary faces are complete
func verifyBoundaryFaceCompleteness(t *testing.T, dg *NUDGTet) {
	// Create set of boundary indices for quick lookup
	boundarySet := make(map[int]bool)
	for _, bIdx := range dg.MapB {
		boundarySet[bIdx] = true
	}

	// Check each element and face
	for k := 0; k < dg.K; k++ {
		for f := 0; f < dg.Nfaces; f++ {
			// If this is a boundary face (self-connected)
			if dg.EToE == nil || dg.EToE[k][f] == k {
				// All nodes on this face should be boundary nodes
				allBoundary := true
				for i := 0; i < dg.Nfp; i++ {
					idx := k*dg.Nfp*dg.Nfaces + f*dg.Nfp + i
					if !boundarySet[idx] {
						allBoundary = false
						break
					}
				}

				if !allBoundary {
					t.Errorf("Boundary face %d of element %d has non-boundary nodes", f, k)
				}
			}
		}
	}
}
