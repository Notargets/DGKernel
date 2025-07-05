package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"testing"
)

func TestTetNudgPhysicalDerivative(t *testing.T) {
	order := 4
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
	Dy := make([]float64, totalNodes)
	Dz := make([]float64, totalNodes)
	// Add array parameters
	params = append(params,
		runner.Input("Rx").Bind(tn.Rx.RawMatrix().Data).CopyTo(),
		runner.Input("Sx").Bind(tn.Sx.RawMatrix().Data).CopyTo(),
		runner.Input("Tx").Bind(tn.Tx.RawMatrix().Data).CopyTo(),
		runner.Input("Ry").Bind(tn.Ry.RawMatrix().Data).CopyTo(),
		runner.Input("Sy").Bind(tn.Sy.RawMatrix().Data).CopyTo(),
		runner.Input("Ty").Bind(tn.Ty.RawMatrix().Data).CopyTo(),
		runner.Input("Rz").Bind(tn.Rz.RawMatrix().Data).CopyTo(),
		runner.Input("Sz").Bind(tn.Sz.RawMatrix().Data).CopyTo(),
		runner.Input("Tz").Bind(tn.Tz.RawMatrix().Data).CopyTo(),
		runner.Output("Dx").Bind(Dx).CopyBack(),
		runner.Output("Dy").Bind(Dy).CopyBack(),
		runner.Output("Dz").Bind(Dz).CopyBack(),
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

        real_t* Dy = Dy_PART(part);
        const real_t* Ry = Ry_PART(part);
        const real_t* Sy = Sy_PART(part);
        const real_t* Ty = Ty_PART(part);
        MATMUL_Dr_%s(Ry, Dy, K[part]);
        MATMUL_ADD_Ds_%s(Sy, Dy, K[part]);
        MATMUL_ADD_Dt_%s(Ty, Dy, K[part]);

        real_t* Dz = Dz_PART(part);
        const real_t* Rz = Rz_PART(part);
        const real_t* Sz = Sz_PART(part);
        const real_t* Tz = Tz_PART(part);
        MATMUL_Dr_%s(Rz, Dz, K[part]);
        MATMUL_ADD_Ds_%s(Sz, Dz, K[part]);
        MATMUL_ADD_Dt_%s(Tz, Dz, K[part]);
    }
}
`, signature,
		props.ShortName, props.ShortName, props.ShortName,
		props.ShortName, props.ShortName, props.ShortName,
		props.ShortName, props.ShortName, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	// Execute differentiation
	err = kp.RunKernel("differentiate")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}
	fmt.Println(Dx[:10])
	fmt.Println(Dy[:10])
	fmt.Println(Dz[:10])
}
