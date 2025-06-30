package integration

import (
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/gocfd/DG3D/mesh/readers"
	"github.com/notargets/gocfd/DG3D/tetrahedra/tetelement"
	"gonum.org/v1/gonum/mat"
)

type TetNudg struct {
	*tetelement.Element3D
}

// GetReferenceElement returns reference element properties and operators
func (t TetNudg) GetReferenceElement() element.ReferenceElement {
	return &tetReferenceElement{t.Element3D}
}

// GetGeometricTransform returns transformation from reference to physical space
func (t TetNudg) GetGeometricTransform() element.GeometricTransform {
	dg := t.Element3D.DG3D
	return element.GeometricTransform{
		Rx:       dg.Rx,
		Ry:       dg.Ry,
		Rz:       dg.Rz,
		Sx:       dg.Sx,
		Sy:       dg.Sy,
		Sz:       dg.Sz,
		Tx:       dg.Tx,
		Ty:       dg.Ty,
		Tz:       dg.Tz,
		J:        dg.J,
		IsAffine: nil, // Could be computed if needed
	}
}

// GetSurfaceGeometry returns face geometry for flux computations
func (t TetNudg) GetSurfaceGeometry() element.SurfaceGeometry {
	dg := t.Element3D.DG3D
	return element.SurfaceGeometry{
		Nx:     dg.Nx,
		Ny:     dg.Ny,
		Nz:     dg.Nz,
		SJ:     dg.SJ,
		FScale: dg.Fscale,
	}
}

// GetProperties returns element metadata
func (t TetNudg) GetProperties() element.ElementProperties {
	dg := t.Element3D.DG3D
	return element.ElementProperties{
		Name:       "Lagrange Tetrahedron Order " + string(rune('0'+dg.N)),
		ShortName:  "Tet" + string(rune('0'+dg.N)),
		Type:       element.Tet,
		Order:      dg.N,
		Np:         dg.Np,
		NFp:        dg.Nfp,
		NEp:        0, // Would need to compute from order
		NVp:        4, // Tetrahedron has 4 vertices
		NIp:        0, // Would need to compute interior points
		Dimensions: element.D3,
	}
}

// GetReferenceGeometry returns node layout in reference space
func (t TetNudg) GetReferenceGeometry() element.ReferenceGeometry {
	dg := t.Element3D.DG3D

	// Convert vectors to slices
	r := make([]float64, dg.Np)
	s := make([]float64, dg.Np)
	tt := make([]float64, dg.Np)

	for i := 0; i < dg.Np; i++ {
		r[i] = dg.R[i]
		s[i] = dg.S[i]
		tt[i] = dg.T[i]
	}

	// TODO: Properly classify nodes by topological entity
	// For now, return empty classifications
	return element.ReferenceGeometry{
		R:              r,
		S:              s,
		T:              tt,
		VertexPoints:   nil, // Would need to identify vertex nodes
		EdgePoints:     nil, // Would need to identify edge nodes
		FacePoints:     nil, // Would need to identify face nodes
		InteriorPoints: nil, // Would need to identify interior nodes
	}
}

// GetNodalModal returns transformation matrices
func (t TetNudg) GetNodalModal() element.NodalModalMatrices {
	dg := t.Element3D.DG3D

	// Compute mass matrix if not available
	var M mat.Matrix
	if !dg.V.IsEmpty() {
		VT := dg.V.T()
		M = mat.NewDense(dg.Np, dg.Np, nil)
		M.(*mat.Dense).Mul(dg.V, VT)
	}

	// Compute Minv if we have M
	var Minv mat.Matrix
	if M != nil {
		Minv = mat.NewDense(dg.Np, dg.Np, nil)
		err := Minv.(*mat.Dense).Inverse(M)
		if err != nil {
			Minv = nil
		}
	}

	return element.NodalModalMatrices{
		V:    dg.V,
		Vinv: nil, // Not stored in DG3D
		M:    M,
		Minv: Minv,
	}
}

// GetReferenceOperators returns differential operators
func (t TetNudg) GetReferenceOperators() element.ReferenceOperators {
	dg := t.Element3D.DG3D
	return element.ReferenceOperators{
		Dr:   dg.Dr,
		Ds:   dg.Ds,
		Dt:   dg.Dt,
		LIFT: dg.LIFT,
	}
}

// NewTetNudg creates a new TetNudg from a mesh file
func NewTetNudg(order int, meshfile string) (tn *TetNudg) {
	msh, err := readers.ReadMeshFile(meshfile)
	if err != nil {
		panic(err)
	}
	el3d, err := tetelement.NewElement3DFromMesh(order, msh)
	if err != nil {
		panic(err)
	}
	tn = &TetNudg{Element3D: el3d}
	return
}

// tetReferenceElement implements element.ReferenceElement interface
type tetReferenceElement struct {
	*tetelement.Element3D
}

func (t *tetReferenceElement) GetProperties() element.ElementProperties {
	dg := t.Element3D.DG3D
	return element.ElementProperties{
		Name:       "Lagrange Tetrahedron Order " + string(rune('0'+dg.N)),
		ShortName:  "Tet" + string(rune('0'+dg.N)),
		Type:       element.Tet,
		Order:      dg.N,
		Np:         dg.Np,
		NFp:        dg.Nfp,
		NEp:        0, // Would need to compute from order
		NVp:        4, // Tetrahedron has 4 vertices
		NIp:        0, // Would need to compute interior points
		Dimensions: element.D3,
	}
}

func (t *tetReferenceElement) GetReferenceGeometry() element.ReferenceGeometry {
	dg := t.Element3D.DG3D

	// Convert vectors to slices
	r := make([]float64, dg.Np)
	s := make([]float64, dg.Np)
	tt := make([]float64, dg.Np)

	for i := 0; i < dg.Np; i++ {
		r[i] = dg.R[i]
		s[i] = dg.S[i]
		tt[i] = dg.T[i]
	}

	// TODO: Properly classify nodes by topological entity
	return element.ReferenceGeometry{
		R:              r,
		S:              s,
		T:              tt,
		VertexPoints:   nil,
		EdgePoints:     nil,
		FacePoints:     nil,
		InteriorPoints: nil,
	}
}

func (t *tetReferenceElement) GetNodalModal() element.NodalModalMatrices {
	dg := t.Element3D.DG3D

	// Compute mass matrix if not available
	var M mat.Matrix
	if !dg.V.IsEmpty() {
		VT := dg.V.T()
		M = mat.NewDense(dg.Np, dg.Np, nil)
		M.(*mat.Dense).Mul(dg.V, VT)
	}

	// Compute Minv if we have M
	var Minv mat.Matrix
	if M != nil {
		Minv = mat.NewDense(dg.Np, dg.Np, nil)
		err := Minv.(*mat.Dense).Inverse(M)
		if err != nil {
			Minv = nil
		}
	}

	return element.NodalModalMatrices{
		V:    dg.V,
		Vinv: nil,
		M:    M,
		Minv: Minv,
	}
}

func (t *tetReferenceElement) GetReferenceOperators() element.ReferenceOperators {
	dg := t.Element3D.DG3D
	return element.ReferenceOperators{
		Dr:   dg.Dr,
		Ds:   dg.Ds,
		Dt:   dg.Dt,
		LIFT: dg.LIFT,
	}
}
