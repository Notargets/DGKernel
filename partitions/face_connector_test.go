package partitions

import (
	"github.com/notargets/DGKernel/tetnudg"
	"github.com/notargets/DGKernel/utils"
	"testing"
)

func TestFaceConnector_CubeMeshFile(t *testing.T) {
	meshfile := "../mesh/cube-partitioned.neu"
	tn := tetnudg.NewTetNudgMesh(1, meshfile)
	props := tn.GetProperties()
	fc, err := utils.NewFaceConnector(tn.Mesh.NumElements, props.NFaces,
		props.NFp, props.Np, tn.VmapP, tn.Mesh.EToP)
	if err != nil {
		t.Fatalf("Failed to create FaceConnector: %v", err)
	}
	_ = fc
}
