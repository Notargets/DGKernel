package utils

import (
	"fmt"
	"math"
)

// TestMeshes provides a collection of standard test meshes that can be used
// across different file format readers (Gmsh, Gambit, SU2)
type TestMeshes struct {
	// Node definitions
	CubeNodes    NodeSet
	TetraNodes   NodeSet
	PyramidNodes NodeSet

	// Element definitions
	SingleTet     ElementSet
	SingleHex     ElementSet
	SinglePrism   ElementSet
	SinglePyramid ElementSet

	// Complete mesh definitions
	TwoTetMesh CompleteMesh
	MixedMesh  CompleteMesh
	CubeMesh   CompleteMesh
}

// NodeSet represents a set of nodes with their coordinates
type NodeSet struct {
	Nodes     [][]float64    // Coordinates [N][3]
	NodeMap   map[string]int // Logical name -> array index
	NodeIDMap map[string]int // Logical name -> node ID (1-based)
}

// ElementSet represents a set of elements with connectivity
type ElementSet struct {
	Type       GeometryType
	Elements   [][]string     // Connectivity using logical node names
	Properties []ElementProps // Additional properties per element
}

// ElementProps holds additional element properties
type ElementProps struct {
	PhysicalTag  int
	GeometricTag int
	PartitionTag int
}

// CompleteMesh represents a complete mesh with nodes and elements
type CompleteMesh struct {
	Nodes       NodeSet
	Elements    []ElementSet
	Dimension   int
	BoundingBox [2][3]float64 // Min and max coordinates
}

// GetStandardTestMeshes returns a set of standard test meshes
func GetStandardTestMeshes() *TestMeshes {
	tm := &TestMeshes{}

	// Initialize standard node sets
	tm.CubeNodes = createCubeNodes()
	tm.TetraNodes = createTetraNodes()
	tm.PyramidNodes = createPyramidNodes()

	// Initialize standard element sets
	tm.SingleTet = createSingleTet()
	tm.SingleHex = createSingleHex()
	tm.SinglePrism = createSinglePrism()
	tm.SinglePyramid = createSinglePyramid()

	// Initialize complete meshes
	tm.TwoTetMesh = createTwoTetMesh()
	tm.MixedMesh = createMixedMesh()
	tm.CubeMesh = createCubeMesh()

	return tm
}

// Node set creators

func createCubeNodes() NodeSet {
	nodes := [][]float64{
		{0, 0, 0}, // 0: origin
		{1, 0, 0}, // 1: x
		{1, 1, 0}, // 2: xy
		{0, 1, 0}, // 3: y
		{0, 0, 1}, // 4: z
		{1, 0, 1}, // 5: xz
		{1, 1, 1}, // 6: xyz
		{0, 1, 1}, // 7: yz
		// Additional nodes for mixed elements
		{0.5, 0, 0},     // 8: mid_bottom_x
		{0, 0.5, 0},     // 9: mid_bottom_y
		{0, 0, 0.5},     // 10: mid_left_z
		{0.5, 0.5, 0},   // 11: center_bottom
		{0.5, 0.5, 1},   // 12: center_top
		{0.5, 0, 0.5},   // 13: mid_front_xz
		{0, 0.5, 0.5},   // 14: mid_left_yz
		{0.5, 0.5, 0.5}, // 15: center
	}

	nodeMap := map[string]int{
		"origin": 0, "x": 1, "xy": 2, "y": 3,
		"z": 4, "xz": 5, "xyz": 6, "yz": 7,
		"mid_bottom_x": 8, "mid_bottom_y": 9, "mid_left_z": 10,
		"center_bottom": 11, "center_top": 12,
		"mid_front_xz": 13, "mid_left_yz": 14, "center": 15,
	}

	// Node IDs are 1-based
	nodeIDMap := make(map[string]int)
	for name, idx := range nodeMap {
		nodeIDMap[name] = idx + 1
	}

	return NodeSet{
		Nodes:     nodes,
		NodeMap:   nodeMap,
		NodeIDMap: nodeIDMap,
	}
}

func createTetraNodes() NodeSet {
	// Standard tetrahedron with vertices at:
	// (0,0,0), (1,0,0), (0,1,0), (0,0,1)
	nodes := [][]float64{
		{0, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		// Mid-edge nodes for higher order
		{0.5, 0, 0},   // 4: edge 0-1
		{0.5, 0.5, 0}, // 5: edge 1-2
		{0, 0.5, 0},   // 6: edge 0-2
		{0.5, 0, 0.5}, // 7: edge 1-3
		{0, 0.5, 0.5}, // 8: edge 2-3
		{0, 0, 0.5},   // 9: edge 0-3
	}

	nodeMap := map[string]int{
		"v0": 0, "v1": 1, "v2": 2, "v3": 3,
		"e01": 4, "e12": 5, "e02": 6,
		"e13": 7, "e23": 8, "e03": 9,
	}

	nodeIDMap := make(map[string]int)
	for name, idx := range nodeMap {
		nodeIDMap[name] = idx + 1
	}

	return NodeSet{
		Nodes:     nodes,
		NodeMap:   nodeMap,
		NodeIDMap: nodeIDMap,
	}
}

func createPyramidNodes() NodeSet {
	// Standard pyramid with square base and apex
	nodes := [][]float64{
		{0, 0, 0},     // 0: base corner 1
		{1, 0, 0},     // 1: base corner 2
		{1, 1, 0},     // 2: base corner 3
		{0, 1, 0},     // 3: base corner 4
		{0.5, 0.5, 1}, // 4: apex
		// Mid-edge nodes for higher order
		{0.5, 0, 0},       // 5: base edge 0-1
		{1, 0.5, 0},       // 6: base edge 1-2
		{0.5, 1, 0},       // 7: base edge 2-3
		{0, 0.5, 0},       // 8: base edge 3-0
		{0.25, 0.25, 0.5}, // 9: edge to apex from 0
		{0.75, 0.25, 0.5}, // 10: edge to apex from 1
		{0.75, 0.75, 0.5}, // 11: edge to apex from 2
		{0.25, 0.75, 0.5}, // 12: edge to apex from 3
		{0.5, 0.5, 0},     // 13: base center
	}

	nodeMap := map[string]int{
		"base0": 0, "base1": 1, "base2": 2, "base3": 3, "apex": 4,
		"base_e01": 5, "base_e12": 6, "base_e23": 7, "base_e30": 8,
		"apex_e0": 9, "apex_e1": 10, "apex_e2": 11, "apex_e3": 12,
		"base_center": 13,
	}

	nodeIDMap := make(map[string]int)
	for name, idx := range nodeMap {
		nodeIDMap[name] = idx + 1
	}

	return NodeSet{
		Nodes:     nodes,
		NodeMap:   nodeMap,
		NodeIDMap: nodeIDMap,
	}
}

// Element set creators

func createSingleTet() ElementSet {
	return ElementSet{
		Type: Tet,
		Elements: [][]string{
			{"v0", "v1", "v2", "v3"},
		},
		Properties: []ElementProps{
			{PhysicalTag: 1, GeometricTag: 1},
		},
	}
}

func createSingleHex() ElementSet {
	return ElementSet{
		Type: Hex,
		Elements: [][]string{
			{"origin", "x", "xy", "y", "z", "xz", "xyz", "yz"},
		},
		Properties: []ElementProps{
			{PhysicalTag: 1, GeometricTag: 1},
		},
	}
}

func createSinglePrism() ElementSet {
	return ElementSet{
		Type: Prism,
		Elements: [][]string{
			{"origin", "x", "y", "z", "xz", "yz"},
		},
		Properties: []ElementProps{
			{PhysicalTag: 1, GeometricTag: 1},
		},
	}
}

func createSinglePyramid() ElementSet {
	return ElementSet{
		Type: Pyramid,
		Elements: [][]string{
			{"base0", "base1", "base2", "base3", "apex"},
		},
		Properties: []ElementProps{
			{PhysicalTag: 1, GeometricTag: 1},
		},
	}
}

// Complete mesh creators

func createTwoTetMesh() CompleteMesh {
	// Two tetrahedra sharing a face
	nodes := NodeSet{
		Nodes: [][]float64{
			{0, 0, 0},
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
			{1, 1, 1},
		},
		NodeMap: map[string]int{
			"v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4,
		},
	}

	nodes.NodeIDMap = make(map[string]int)
	for name, idx := range nodes.NodeMap {
		nodes.NodeIDMap[name] = idx + 1
	}

	elements := []ElementSet{
		{
			Type: Tet,
			Elements: [][]string{
				{"v0", "v1", "v2", "v3"},
				{"v1", "v2", "v3", "v4"},
			},
			Properties: []ElementProps{
				{PhysicalTag: 1, GeometricTag: 1},
				{PhysicalTag: 1, GeometricTag: 1},
			},
		},
	}

	return CompleteMesh{
		Nodes:     nodes,
		Elements:  elements,
		Dimension: 3,
		BoundingBox: [2][3]float64{
			{0, 0, 0},
			{1, 1, 1},
		},
	}
}

func createMixedMesh() CompleteMesh {
	// A mesh with one of each 3D element type
	nodes := createCubeNodes()

	elements := []ElementSet{
		{
			Type: Tet,
			Elements: [][]string{
				{"origin", "x", "y", "z"},
				{"x", "xy", "y", "center"},
			},
		},
		{
			Type: Hex,
			Elements: [][]string{
				{"origin", "x", "xy", "y", "z", "xz", "xyz", "yz"},
			},
		},
		{
			Type: Prism,
			Elements: [][]string{
				{"origin", "x", "y", "z", "xz", "yz"},
			},
		},
		{
			Type: Pyramid,
			Elements: [][]string{
				{"origin", "x", "xy", "y", "center"},
			},
		},
	}

	// Set properties for all elements
	for i := range elements {
		for range elements[i].Elements {
			elements[i].Properties = append(elements[i].Properties,
				ElementProps{PhysicalTag: 10, GeometricTag: 1})
		}
	}

	return CompleteMesh{
		Nodes:     nodes,
		Elements:  elements,
		Dimension: 3,
		BoundingBox: [2][3]float64{
			{0, 0, 0},
			{1, 1, 1},
		},
	}
}

func createCubeMesh() CompleteMesh {
	// A cube meshed with 6 tetrahedra using a systematic decomposition
	// This uses the "diagonal" method where all tets share the diagonal from origin to xyz
	nodes := createCubeNodes()

	// Using only the 8 corner vertices (no center node needed for this decomposition)
	// The diagonal from origin (0,0,0) to xyz (1,1,1) is shared by all 6 tetrahedra
	// Each tet is formed by the diagonal plus one edge of the cube
	elements := []ElementSet{
		{
			Type: Tet,
			Elements: [][]string{
				// Bottom face tets (z=0 plane)
				{"origin", "x", "xy", "xyz"}, // Front edge of bottom face
				{"origin", "xy", "y", "xyz"}, // Back edge of bottom face

				// Top face tets (z=1 plane)
				{"origin", "z", "xz", "xyz"}, // Front edge projection
				{"origin", "yz", "z", "xyz"}, // Back edge projection

				// Middle tets (connecting bottom to top)
				{"origin", "xz", "x", "xyz"}, // Right face contribution (swapped x and xz)
				{"origin", "y", "yz", "xyz"}, // Left face contribution
			},
			Properties: []ElementProps{
				{PhysicalTag: 1, GeometricTag: 1},
				{PhysicalTag: 1, GeometricTag: 1},
				{PhysicalTag: 1, GeometricTag: 1},
				{PhysicalTag: 1, GeometricTag: 1},
				{PhysicalTag: 1, GeometricTag: 1},
				{PhysicalTag: 1, GeometricTag: 1},
			},
		},
	}

	return CompleteMesh{
		Nodes:     nodes,
		Elements:  elements,
		Dimension: 3,
		BoundingBox: [2][3]float64{
			{0, 0, 0},
			{1, 1, 1},
		},
	}
}

// Validation helpers

// ValidateNodeCoordinates checks if node coordinates match expected values
func ValidateNodeCoordinates(nodes [][]float64, expected [][]float64, tolerance float64) error {
	if len(nodes) != len(expected) {
		return fmt.Errorf("node count mismatch: got %d, expected %d", len(nodes), len(expected))
	}

	for i := range nodes {
		for j := 0; j < 3; j++ {
			diff := math.Abs(nodes[i][j] - expected[i][j])
			if diff > tolerance {
				return fmt.Errorf("node %d coord %d: got %f, expected %f (diff %f > tol %f)",
					i, j, nodes[i][j], expected[i][j], diff, tolerance)
			}
		}
	}

	return nil
}

// ValidateElementConnectivity checks if element connectivity matches expected
func ValidateElementConnectivity(elements [][]int, expected [][]int) error {
	if len(elements) != len(expected) {
		return fmt.Errorf("element count mismatch: got %d, expected %d", len(elements), len(expected))
	}

	for i := range elements {
		if len(elements[i]) != len(expected[i]) {
			return fmt.Errorf("element %d node count mismatch: got %d, expected %d",
				i, len(elements[i]), len(expected[i]))
		}

		for j := range elements[i] {
			if elements[i][j] != expected[i][j] {
				return fmt.Errorf("element %d node %d: got %d, expected %d",
					i, j, elements[i][j], expected[i][j])
			}
		}
	}

	return nil
}

// Geometric validation helpers

// ComputeTetVolume computes the volume of a tetrahedron given 4 vertices
func ComputeTetVolume(v0, v1, v2, v3 []float64) float64 {
	// Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
	d1 := []float64{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}
	d2 := []float64{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}
	d3 := []float64{v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]}

	// Compute determinant
	det := d1[0]*(d2[1]*d3[2]-d2[2]*d3[1]) -
		d1[1]*(d2[0]*d3[2]-d2[2]*d3[0]) +
		d1[2]*(d2[0]*d3[1]-d2[1]*d3[0])

	return math.Abs(det) / 6.0
}

// ComputeTriangleArea computes the area of a triangle given 3 vertices
func ComputeTriangleArea(v0, v1, v2 []float64) float64 {
	// Area = |cross(v1-v0, v2-v0)| / 2
	d1 := []float64{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}
	d2 := []float64{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}

	// Cross product
	cross := []float64{
		d1[1]*d2[2] - d1[2]*d2[1],
		d1[2]*d2[0] - d1[0]*d2[2],
		d1[0]*d2[1] - d1[1]*d2[0],
	}

	// Magnitude
	mag := math.Sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2])
	return mag / 2.0
}

// GetTetFaces returns the 4 triangular faces of a tetrahedron
// Each face is represented as 3 vertex indices in consistent orientation
func GetTetFaces(tet []int) [4][3]int {
	// Standard tetrahedron face connectivity
	// Faces are oriented so normals point outward
	return [4][3]int{
		{tet[0], tet[2], tet[1]}, // Face 0: opposite vertex 3
		{tet[0], tet[1], tet[3]}, // Face 1: opposite vertex 2
		{tet[0], tet[3], tet[2]}, // Face 2: opposite vertex 1
		{tet[1], tet[2], tet[3]}, // Face 3: opposite vertex 0
	}
}

// FaceKey represents a triangular face for comparison
type FaceKey struct {
	V0, V1, V2 int
}

// NewFaceKey creates a normalized face key (smallest vertex first)
func NewFaceKey(v0, v1, v2 int) FaceKey {
	// Sort vertices to create a canonical representation
	if v0 > v1 {
		v0, v1 = v1, v0
	}
	if v0 > v2 {
		v0, v2 = v2, v0
	}
	if v1 > v2 {
		v1, v2 = v2, v1
	}
	return FaceKey{v0, v1, v2}
}
