package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/stretchr/testify/assert"
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
	Ur := make([]float64, totalNodes)
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = 2. * float64(i%10)
	}

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*runner.ParamBuilder, 0, len(matrices)+2)

	// Add matrices as parameters
	for name, mat := range matrices {
		fmt.Printf("%s\n", name)
		params = append(params, runner.Input(name).Bind(mat).ToMatrix())
	}

	// Add array parameters
	params = append(params,
		runner.Input("U").Bind(U).CopyTo(),
		runner.Output("Ur").Bind(Ur).CopyBack(),
	)

	// Define kernel with all parameters
	err := kp.DefineKernel("differentiate", params...)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get signature and build kernel
	signature, _ := kp.GetKernelSignature("differentiate")
	kernelSource := fmt.Sprintf(`
@kernel void differentiate(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
    }
}
`, signature, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	// Execute differentiation
	err = kp.RunKernel("differentiate")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	expected := make([]float64, totalNodes)
	for i := range expected {
		expected[i] = 1.
	}

	if order == 1 {
		assert.InDeltaSlicef(t, expected, Ur, 1.e-8, "")
	}
}
