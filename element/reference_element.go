package element

import (
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
)

// Dimensionality represents the spatial dimension of an element
type Dimensionality uint8

const (
	D0 Dimensionality = iota // 0D elements (points)
	D1                       // 1D elements (lines, edges)
	D2                       // 2D elements (triangles, quadrilaterals)
	D3                       // 3D elements (tetrahedra, hexahedra, etc.)
)

// ElementProperties contains metadata describing an element type
type ElementProperties struct {
	Name       string             // Full descriptive name (e.g., "Lagrange Tetrahedron Order 3")
	ShortName  string             // Abbreviated name (e.g., "Tet3")
	Type       utils.GeometryType // Element shape
	Order      int                // Polynomial order
	Np         int                // Total number of nodes/points in element
	NFp        int                // Number of nodes per face
	NEp        int                // Number of nodes per edge
	NVp        int                // Number of vertex nodes (equals number of vertices)
	NIp        int                // Number of strictly interior nodes
	NFaces     int                // Number of faces in each element
	NEdges     int                // Number of edges in each element
	Dimensions Dimensionality     // Spatial dimension (1D, 2D, or 3D)
}

// ReferenceGeometry defines the layout of nodes in reference space [-1,1]^d
type ReferenceGeometry struct {
	// Node coordinates in reference space
	// For 3D: all three are used; for 2D: only R,S; for 1D: only R
	R, S, T []float64 // Length Np each

	// Node classification by topological entity
	VertexPoints   []int   // Indices of nodes located at vertices
	EdgePoints     [][]int // [edge_num][point_indices] - nodes on each edge
	FacePoints     [][]int // [face_num][point_indices] - nodes on each face
	InteriorPoints []int   // Indices of nodes strictly inside the element
}

// NodalModalMatrices contains transformation matrices between nodal and modal representations
type NodalModalMatrices struct {
	V    mat.Matrix // Vandermonde matrix: modal to nodal transformation [Np × Np]
	Vinv mat.Matrix // Inverse Vandermonde: nodal to modal transformation [Np × Np]
	M    mat.Matrix // Mass matrix in nodal space [Np × Np]
	Minv mat.Matrix // Inverse mass matrix [Np × Np]
}

// ReferenceOperators contains differential operators in reference space [-1,1]^d
type ReferenceOperators struct {
	// Differentiation matrices in reference coordinates
	Dr mat.Matrix // Derivative with respect to r [Np × Np]
	Ds mat.Matrix // Derivative with respect to s [Np × Np]
	Dt mat.Matrix // Derivative with respect to t [Np × Np] (3D only)

	// Surface-to-volume lifting operator
	// Maps face values to volume contribution [Np × (NFaces*NFp)]
	LIFT mat.Matrix
}

// ReferenceElement defines element properties and operators in reference space
// This interface is implemented once per element type (e.g., TetP3, TriP2)
type ReferenceElement interface {
	// Element metadata and properties
	GetProperties() ElementProperties

	// Node distribution in reference space
	GetReferenceGeometry() ReferenceGeometry

	// Transformation between nodal and modal bases
	GetNodalModal() NodalModalMatrices

	// Differential operators in reference space
	GetReferenceOperators() ReferenceOperators
}
