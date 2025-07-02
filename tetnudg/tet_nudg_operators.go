package tetnudg

import "github.com/notargets/DGKernel/element"

type DXYZ struct {
	*TetNudgMesh
	Dir DirectionType
}

type DirectionType uint8

const (
	XDIR DirectionType = iota
	YDIR
	ZDIR
)

func NewDXYZ(dg *TetNudgMesh, dir DirectionType) (dxyz *DXYZ) {
	dxyz = &DXYZ{
		TetNudgMesh: dg,
		Dir:         dir,
	}
	return
}

func (dxyz *DXYZ) GetMacro() string {
	// TODO implement me
	panic("implement me")
}

func (dxyz *DXYZ) GetArguments() []element.Argument {
	// TODO implement me
	panic("implement me")
}

func (dxyz *DXYZ) GetKernelSource() string {
	// TODO implement me
	panic("implement me")
}
