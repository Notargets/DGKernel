package gonudg

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// NUDGTet represents a 3D discontinuous Galerkin solver
type NUDGTet struct {
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
	Fmask [][]int // [Nfp][Nfaces]

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
	MapB       []int // Boundary nodes
	VmapB      []int // Boundary vertex map

	// Surface coordinates
	Fx, Fy, Fz *mat.Dense // Face coordinates
}

// NewDG3D creates and initializes a new 3D DG solver
func NewDG3D(N int, VX, VY, VZ []float64, EToV [][]int) (*NUDGTet, error) {
	dg := &NUDGTet{
		N:       N,
		VX:      VX,
		VY:      VY,
		VZ:      VZ,
		EToV:    EToV,
		K:       len(EToV),
		NODETOL: 1e-7,
		Nfaces:  4,
	}

	// Initialize
	if err := dg.StartUp3D(); err != nil {
		return nil, err
	}

	return dg, nil
}

// StartUp3D initializes the 3D DG operators
func (dg *NUDGTet) StartUp3D() error {
	// Definition of constants
	dg.Np = (dg.N + 1) * (dg.N + 2) * (dg.N + 3) / 6
	dg.Nfp = (dg.N + 1) * (dg.N + 2) / 2

	// Compute nodal set
	x1, y1, z1 := Nodes3D(dg.N)
	dg.R, dg.S, dg.T = XYZtoRST(x1, y1, z1)

	// Build reference element matrices
	dg.V = Vandermonde3D(dg.N, dg.R, dg.S, dg.T)

	// Compute V inverse
	dg.Vinv = mat.NewDense(dg.Np, dg.Np, nil)
	err := dg.Vinv.Inverse(dg.V)
	if err != nil {
		return fmt.Errorf("failed to invert Vandermonde matrix: %v", err)
	}

	// Compute mass matrix: M = (V^T * V)^{-1} = V^{-T} * V^{-1}
	dg.MassMatrix = mat.NewDense(dg.Np, dg.Np, nil)
	dg.MassMatrix.Mul(dg.Vinv.T(), dg.Vinv)

	// Build differentiation matrices
	dg.Dr, dg.Ds, dg.Dt = Dmatrices3D(dg.N, dg.R, dg.S, dg.T, dg.V)

	// Build coordinates of all the nodes
	dg.X = mat.NewDense(dg.Np, dg.K, nil)
	dg.Y = mat.NewDense(dg.Np, dg.K, nil)
	dg.Z = mat.NewDense(dg.Np, dg.K, nil)

	// Map from reference to physical elements
	for k := 0; k < dg.K; k++ {
		va := dg.EToV[k][0]
		vb := dg.EToV[k][1]
		vc := dg.EToV[k][2]
		vd := dg.EToV[k][3]

		for i := 0; i < dg.Np; i++ {
			dg.X.Set(i, k, 0.5*(-(1.0+dg.R[i]+dg.S[i]+dg.T[i])*dg.VX[va]+
				(1.0+dg.R[i])*dg.VX[vb]+
				(1.0+dg.S[i])*dg.VX[vc]+
				(1.0+dg.T[i])*dg.VX[vd]))

			dg.Y.Set(i, k, 0.5*(-(1.0+dg.R[i]+dg.S[i]+dg.T[i])*dg.VY[va]+
				(1.0+dg.R[i])*dg.VY[vb]+
				(1.0+dg.S[i])*dg.VY[vc]+
				(1.0+dg.T[i])*dg.VY[vd]))

			dg.Z.Set(i, k, 0.5*(-(1.0+dg.R[i]+dg.S[i]+dg.T[i])*dg.VZ[va]+
				(1.0+dg.R[i])*dg.VZ[vb]+
				(1.0+dg.S[i])*dg.VZ[vc]+
				(1.0+dg.T[i])*dg.VZ[vd]))
		}
	}

	// Find all the nodes that lie on each face
	dg.BuildFmask()

	// Extract face coordinates
	dg.ExtractFaceCoordinates()

	// Create surface integral terms
	if err := dg.Lift3D(); err != nil {
		return fmt.Errorf("Lift3D failed: %v", err)
	}

	// Calculate geometric factors and normals
	if err := dg.Normals3D(); err != nil {
		return fmt.Errorf("Normals3D failed: %v", err)
	}

	// Compute Fscale = SJ ./ J(Fmask,:)
	dg.ComputeFscale()

	// Build connectivity matrix
	dg.tiConnect3D()

	// Build connectivity maps
	dg.BuildMaps3D()

	return nil
}

// BuildFmask finds all nodes that lie on each face
func (dg *NUDGTet) BuildFmask() {
	dg.Fmask = make([][]int, 4)

	// Face 1: T = -1
	for i := 0; i < dg.Np; i++ {
		if math.Abs(1.0+dg.T[i]) < dg.NODETOL {
			dg.Fmask[0] = append(dg.Fmask[0], i)
		}
	}

	// Face 2: S = -1
	for i := 0; i < dg.Np; i++ {
		if math.Abs(1.0+dg.S[i]) < dg.NODETOL {
			dg.Fmask[1] = append(dg.Fmask[1], i)
		}
	}

	// Face 3: R+S+T = -1
	for i := 0; i < dg.Np; i++ {
		if math.Abs(1.0+dg.R[i]+dg.S[i]+dg.T[i]) < dg.NODETOL {
			dg.Fmask[2] = append(dg.Fmask[2], i)
		}
	}

	// Face 4: R = -1
	for i := 0; i < dg.Np; i++ {
		if math.Abs(1.0+dg.R[i]) < dg.NODETOL {
			dg.Fmask[3] = append(dg.Fmask[3], i)
		}
	}
}

// ExtractFaceCoordinates extracts coordinates at face nodes
func (dg *NUDGTet) ExtractFaceCoordinates() {
	// Create face coordinate matrices
	dg.Fx = mat.NewDense(dg.Nfp*4, dg.K, nil)
	dg.Fy = mat.NewDense(dg.Nfp*4, dg.K, nil)
	dg.Fz = mat.NewDense(dg.Nfp*4, dg.K, nil)

	// Extract coordinates for each face
	for face := 0; face < 4; face++ {
		for k := 0; k < dg.K; k++ {
			for i, nodeIdx := range dg.Fmask[face] {
				row := face*dg.Nfp + i
				dg.Fx.Set(row, k, dg.X.At(nodeIdx, k))
				dg.Fy.Set(row, k, dg.Y.At(nodeIdx, k))
				dg.Fz.Set(row, k, dg.Z.At(nodeIdx, k))
			}
		}
	}
}
