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

type Element interface {
	Name() string
	ShortName() string
	GeometryType() ElementGeometry
	Order() int
	Np() int  // Number of defining geometric points
	NFp() int // Number of face points
	NEp() int // Number of edge points
	NVp() int // Number of vertex points
	NIp() int // Number of interior points
	Dimensions() Dimensionality

	// Reference Geometry Definition
	R() []float64
	S() []float64
	T() []float64

	// JacobianMatrix returns the coordinate transformation Jacobian ∂x/∂ξ
	// as a [9, Np] matrix where column i contains the 3×3 Jacobian for point i,
	// stored in row-major order:
	//   jmat.At(0,i) = ∂x/∂ξ    jmat.At(1,i) = ∂y/∂ξ    jmat.At(2,i) = ∂z/∂ξ
	//   jmat.At(3,i) = ∂x/∂η    jmat.At(4,i) = ∂y/∂η    jmat.At(5,i) = ∂z/∂η
	//   jmat.At(6,i) = ∂x/∂ζ    jmat.At(7,i) = ∂y/∂ζ    jmat.At(8,i) = ∂z/∂ζ
	// For 2D elements, returns [4, Np] with only the 2×2 components.
	JacobianMatrix() mat.Matrix

	// Point classification by geometric location
	VertexPoints() []int   // Indices into the Np points that are at vertices
	EdgePoints() [][]int   // [edge_num][point_indices] - points on each edge
	FacePoints() [][]int   // [face_num][point_indices] - points on each face
	InteriorPoints() []int // Points strictly inside the element

	// Nodal / Modal matrices
	V() mat.Matrix
	Vinv() mat.Matrix
	M() mat.Matrix
	Minv() mat.Matrix

	// Basic operators
	Dr() mat.Matrix
	Ds() mat.Matrix
	Dt() mat.Matrix
	LIFT() mat.Matrix
}
