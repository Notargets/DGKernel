package element

import "gonum.org/v1/gonum/mat"

// GeometricTransform maps between reference space [-1,1]^d and physical space
// All data is stored for the entire mesh (K elements) in column-major format
type GeometricTransform struct {
	// Components of the inverse Jacobian matrix (∂ξ/∂x terms)
	// Dimension: [Np × K] where:
	//   - Np: number of points per element (or 1 for affine elements)
	//   - K: total number of elements in the mesh
	//   - Column k contains metric data for element k
	//   - Row i contains metric data for node i within each element
	Rx, Ry, Rz mat.Matrix // ∂r/∂x, ∂r/∂y, ∂r/∂z (Rz used in 3D only)
	Sx, Sy, Sz mat.Matrix // ∂s/∂x, ∂s/∂y, ∂s/∂z (Sz used in 3D only)
	Tx, Ty, Tz mat.Matrix // ∂t/∂x, ∂t/∂y, ∂t/∂z (3D only)

	// Jacobian determinant |∂(x,y,z)/∂(r,s,t)|
	// Dimension: [Np × K] where Np=1 for affine elements
	// Used for integration: ∫_Ω f dV = ∫_Ω̂ f |J| dr ds dt
	J mat.Matrix

	// Optimization flags
	IsAffine []bool // Length K: true if element k has constant metric terms
}

// SurfaceGeometry contains geometric information for element faces
// Used for numerical flux computations and boundary conditions
type SurfaceGeometry struct {
	// Unit outward normal vectors at face quadrature points
	// Dimension: [NFaces*NFp × K] where rows are ordered as:
	//   [face0_point0, face0_point1, ..., face0_pointNfp-1,
	//    face1_point0, face1_point1, ..., face1_pointNfp-1, ...]
	Nx, Ny, Nz mat.Matrix // Nz used in 3D only

	// Surface Jacobian (physical face area / reference face area)
	// Dimension: [NFaces*NFp × K]
	// Includes the transformation from reference face to physical face
	SJ mat.Matrix

	// Relative surface Jacobian for interior faces
	// Dimension: [NFaces*NFp × K]
	// FScale = SJ_neighbor / SJ_self at matching quadrature points
	// Used to ensure conservation in numerical fluxes
	FScale mat.Matrix
}

type MeshProperties struct {
	NumElements int
	NumVertices int
	NumFaces    int
}

// MeshElement represents a physical element with complete geometric information
// This combines reference element properties with physical space metrics
type MeshElement interface {
	GetMeshProperties() MeshProperties

	// Access to reference space properties and operators
	GetReferenceElement() ReferenceElement

	// Geometric transformation from reference to physical space
	GetGeometricTransform() GeometricTransform

	// Surface geometry for flux computations
	GetSurfaceGeometry() SurfaceGeometry

	String() string // Summary of key stats
}

// ElementConnectivity defines mesh topology and boundary conditions
type ElementConnectivity struct {
	// Element-to-element connectivity
	EToE mat.Matrix // [K × NFaces] Element k, face f connects to element EToE[k,f]
	EToF mat.Matrix // [K × NFaces] Element k, face f connects to face EToF[k,f] of neighbor

	// Boundary condition markers
	BCType mat.Matrix // [K × NFaces] Boundary type for each face (-1 for interior faces)
}
