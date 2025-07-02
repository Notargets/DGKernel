package tetnudg

import (
	"github.com/notargets/DGKernel/element"
)

type RefMatrixMacros struct {
	*TetNudgMesh
}

func (dg *TetNudgMesh) NewRefMatrixMacros() *RefMatrixMacros {
	return &RefMatrixMacros{TetNudgMesh: dg}
}

func (rmm *RefMatrixMacros) GetMacro() string {
	// This returns all base reference element matrices as macros with static
	// initializers. This should be placed in the kernel global area as a
	// preamble
	return element.GenerateMatrixMacros(rmm.GetReferenceElement())
}

func (rmm *RefMatrixMacros) GetArguments() []element.Argument {
	// TODO implement me
	panic("implement me")
}

func (rmm *RefMatrixMacros) GetKernelSource() string {
	return ""
}
