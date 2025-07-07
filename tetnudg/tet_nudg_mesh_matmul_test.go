package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestTetNudgMatmul(t *testing.T) {
	order := 1
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Initialize test data
	// Ur := make([]float64, totalNodes)
	Ur := mat.NewDense(Np, Ktot, nil)
	U := mat.NewDense(Np, Ktot, nil)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			U.Set(j, K, 2.*float64(j%10))
		}
	}

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*runner.ParamBuilder, 0, len(matrices)+2)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, runner.Input(name).Bind(mat).ToMatrix())
	}

	// Add array parameters
	params = append(params,
		runner.Input("U").Bind(U).CopyTo(),
		runner.Output("Ur").Bind(Ur).CopyBack(),
	)

	// Define kernel with all parameters
	kernelName := "differentiate"
	err := kp.DefineKernel(kernelName, params...)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get signature and build kernel
	signature, _ := kp.GetKernelSignature(kernelName)
	kernelSource := fmt.Sprintf(`
@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
    }
}
`, kernelName, signature, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	// Execute differentiation
	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}
	expected := make([]float64, totalNodes)
	for i := range expected {
		expected[i] = 1.
	}

	if order == 1 {
		assert.InDeltaSlicef(t, expected, Ur.RawMatrix().Data, 1.e-8, "")
	}
}
