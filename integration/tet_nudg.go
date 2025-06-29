package integration

import (
	"github.com/notargets/gocfd/DG3D/mesh/readers"
	"github.com/notargets/gocfd/DG3D/tetrahedra/tetelement"
)

type TetNudg struct {
	*tetelement.Element3D
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
