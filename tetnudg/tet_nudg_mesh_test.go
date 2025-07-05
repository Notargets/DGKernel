package tetnudg

import (
	"fmt"
	"testing"
)

func TestNewTetNudgMesh(t *testing.T) {
	tm := NewTetNudgMesh(1, "cube-partitioned.neu")
	fmt.Printf("%s", tm.String())
	for name, _ := range tm.GetRefMatrices() {
		fmt.Printf("%s\n", name)
	}
}
