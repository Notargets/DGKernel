package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"testing"
)

func TestTetNudgPhysicalDerivative(t *testing.T) {
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

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*runner.ParamBuilder, 0, len(matrices)+2)

	// Add matrices as parameters
	for name, mat := range matrices {
		fmt.Printf("%s\n", name)
		params = append(params, runner.Input(name).Bind(mat).ToMatrix())
	}

	Dx := make([]float64, totalNodes)
	// Add array parameters
	params = append(params,
		runner.Input("Rx").Bind(tn.Rx.RawMatrix().Data).CopyTo(),
		runner.Input("Sx").Bind(tn.Rx.RawMatrix().Data).CopyTo(),
		runner.Input("Tx").Bind(tn.Rx.RawMatrix().Data).CopyTo(),
		runner.Output("Dx").Bind(Dx).CopyBack(),
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
        real_t* Dx = Dx_PART(part);
        const real_t* Rx = Rx_PART(part);
        const real_t* Sx = Sx_PART(part);
        const real_t* Tx = Tx_PART(part);
        MATMUL_Dr_%s(Rx, Dx, K[part]);
        MATMUL_ADD_Ds_%s(Sx, Dx, K[part]);
        MATMUL_ADD_Dt_%s(Tx, Dx, K[part]);
    }
}
`, signature, props.ShortName, props.ShortName, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	// Execute differentiation
	err = kp.RunKernel("differentiate")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}
	// fmt.Println(Dx)
}
