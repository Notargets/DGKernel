package element

import "gonum.org/v1/gonum/mat"

type Dimensionality uint8

const (
	D1 Dimensionality = iota
	D2
	D3
)

type ElementGeometry uint8

const (
	Tet ElementGeometry = iota
	Hex
	Prism
	Pyramid
	Tri
	Rectangle
	Line
)

type Element struct {
	Name         func() string
	ShortName    func() string
	GeometryType func() ElementGeometry
	Order        func() int
	Np           func() int // Number of defining geometric points
	NFp          func() int // Number of face points
	NEp          func() int // Number of edge points
	NVp          func() int // Number of vertex points
	NIp          func() int // Number of interior points
	Dimensions   func() Dimensionality
	// Reference Geometry Definition
	R func() []float64
	S func() []float64
	T func() []float64
	// JacobianMatrix returns the coordinate transformation Jacobian ∂x/∂ξ
	// as a [9, Np] matrix where column i contains the 3×3 Jacobian for point i,
	// stored in row-major order:
	//   jmat.At(0,i) = ∂x/∂ξ    jmat.At(1,i) = ∂y/∂ξ    jmat.At(2,i) = ∂z/∂ξ
	//   jmat.At(3,i) = ∂x/∂η    jmat.At(4,i) = ∂y/∂η    jmat.At(5,i) = ∂z/∂η
	//   jmat.At(6,i) = ∂x/∂ζ    jmat.At(7,i) = ∂y/∂ζ    jmat.At(8,i) = ∂z/∂ζ
	// For 2D elements, returns [4, Np] with only the 2×2 components.
	JacobianMatrix func() mat.Matrix
	// Point classification by geometric location
	VertexPoints   func() []int   // Indices into the Np points that are at vertices
	EdgePoints     func() [][]int // [edge_num][point_indices] - points on each edge
	FacePoints     func() [][]int // [face_num][point_indices] - points on each face
	InteriorPoints func() []int   // Points strictly inside the element
	// Nodal / Modal matrices
	V, Vinv func() mat.Matrix
	M, Minv func() mat.Matrix
	// Basic operators
	Dr, Ds, Dt func() mat.Matrix
	LIFT       func() mat.Matrix
}
