package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/element"
	"testing"
)

func TestNewTetNudgMesh(t *testing.T) {
	tm := NewTetNudgMesh(1, "cube-partitioned.neu")
	fmt.Printf("%s", tm.String())
	fmt.Println(element.GenerateMatrixMacros(tm.GetReferenceElement()))
}
