package tetnudg

import (
	"fmt"
	"testing"
)

func TestNewTetNudgMesh(t *testing.T) {
	tm := NewTetNudgMesh(1, "cube-partitioned.neu")
	fmt.Printf("%s", tm.String())
	rmm := tm.NewRefMatrixMacros()
	fmt.Println(rmm.GetMacro())
}
