package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/element"
	"testing"
)

func TestNewTetNudgMesh(t *testing.T) {
	tm := NewTetNudgMesh(1, "cube-partitioned.neu")
	fmt.Printf("%s", tm.String())
	for name, _ := range element.GetRefMatrices(tm) {
		fmt.Printf("%s\n", name)
	}
}
