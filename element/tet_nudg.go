package element

import (
	"github.com/notargets/gocfd/DG3D/mesh/readers"
	"github.com/notargets/gocfd/DG3D/tetrahedra/tetelement"
	"gonum.org/v1/gonum/mat"
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

func (t TetNudg) Name() string {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) ShortName() string {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) GeometryType() ElementGeometry {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Order() int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Np() int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) NFp() int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) NEp() int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) NVp() int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) NIp() int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Dimensions() Dimensionality {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) R() []float64 {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) S() []float64 {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) T() []float64 {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) JacobianMatrix() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) VertexPoints() []int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) EdgePoints() [][]int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) FacePoints() [][]int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) InteriorPoints() []int {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) V() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Vinv() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) M() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Minv() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Dr() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Ds() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) Dt() mat.Matrix {
	// TODO implement me
	panic("implement me")
}

func (t TetNudg) LIFT() mat.Matrix {
	// TODO implement me
	panic("implement me")
}
