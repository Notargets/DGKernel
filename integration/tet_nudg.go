package integration

import (
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/gocfd/DG3D/mesh/readers"
	"github.com/notargets/gocfd/DG3D/tetrahedra/tetelement"
)

type TetNudg struct {
	*tetelement.Element3D
}

func (t TetNudg) GetReferenceElement() element.ReferenceElement {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GetGeometricTransform() element.GeometricTransform {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GetSurfaceGeometry() element.SurfaceGeometry {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GetProperties() element.ElementProperties {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GetReferenceGeometry() element.ReferenceGeometry {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GetNodalModal() element.NodalModalMatrices {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GetReferenceOperators() element.ReferenceOperators {
	// TODO implement me
	panic("implement me")
}

func NewTetNudg(order int, meshfile string) (tn *TetNudg) {
	msh, err := readers.ReadMeshFile(meshfile)
	if err != nil {
		panic(err)
	}
	tn.Element3D, err = tetelement.NewElement3DFromMesh(order, msh)
	if err != nil {
		panic(err)
	}
	return
}
