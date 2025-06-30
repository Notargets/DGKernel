package gonudg

import (
	"gonum.org/v1/gonum/mat"
)

// DG3D represents a 3D discontinuous Galerkin solver
type DG3D struct {
	// Polynomial order
	N       int
	NODETOL float64

	// Number of nodes and faces
	Np     int // Number of nodes per element
	Nfp    int // Number of nodes per face
	Nfaces int // Number of faces per element (4 for tetrahedra)

	// Node coordinates in reference element
	R, S, T []float64

	// Vandermonde matrices
	V    *mat.Dense // Vandermonde matrix
	Vinv *mat.Dense // Inverse Vandermonde matrix

	// Mass matrix
	MassMatrix *mat.Dense

	// Differentiation matrices
	Dr, Ds, Dt *mat.Dense

	// Lift matrix
	LIFT *mat.Dense

	// Face masks - indices of nodes on each face
	Fmask [][]int

	// Physical coordinates
	X, Y, Z *mat.Dense

	// Mesh information
	K          int       // Number of elements
	VX, VY, VZ []float64 // Vertex coordinates
	EToV       [][]int   // Element to vertex connectivity

	// Geometric factors
	J          *mat.Dense // Jacobian
	Rx, Ry, Rz *mat.Dense // Metric terms
	Sx, Sy, Sz *mat.Dense
	Tx, Ty, Tz *mat.Dense

	// Face normals and surface Jacobian
	Nx, Ny, Nz *mat.Dense // Face normals
	SJ         *mat.Dense // Surface Jacobian
	Fscale     *mat.Dense // Face scaling

	// Connectivity
	EToE       [][]int // Element to element
	EToF       [][]int // Element to face
	VmapM      []int   // Vertex map minus
	VmapP      []int   // Vertex map plus
	MapP, MapM []int
	MapB       []int   // Boundary nodes
	VmapB      []int   // Boundary vertex map
	FmaskF     [][]int // Flattened face mask

	// Surface coordinates
	FX, FY, FZ *mat.Dense // Face coordinates
}
