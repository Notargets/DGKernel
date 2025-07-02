package tetnudg

import (
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/DGKernel/utils"
)

type DRST struct {
	*TetNudgMesh
	Dir utils.DirectionType
}

func NewDRST(dg *TetNudgMesh, dir utils.DirectionType) (drst *DRST) {
	drst = &DRST{
		TetNudgMesh: dg,
		Dir:         dir,
	}
	return
}

func (drst *DRST) GetMacro() string {
	// TODO implement me
	panic("implement me")
}

func (drst *DRST) GetArguments() []element.Argument {
	// TODO implement me
	panic("implement me")
}

func (drst *DRST) GetKernelSource() string {
	// TODO implement me
	panic("implement me")
}
