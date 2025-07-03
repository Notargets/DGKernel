package tetnudg

import (
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/DGKernel/utils"
)

type DXYZ struct {
	*TetNudgMesh
	Dir utils.DirectionType
}

func NewDXYZ(dg *TetNudgMesh, dir utils.DirectionType) (dxyz *DXYZ) {
	dxyz = &DXYZ{
		TetNudgMesh: dg,
		Dir:         dir,
	}
	return
}

func (dxyz *DXYZ) GetArguments() []element.Argument {
	// TODO implement me
	panic("implement me")
}

func (dxyz *DXYZ) GetKernelSource() string {
	// TODO implement me
	panic("implement me")
}
