package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/stretchr/testify/assert"
	"testing"
	"unsafe"
)

func TestTetNudgMatmul(t *testing.T) {
	tn := NewTetNudgMesh(1, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot

	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	for name, mat := range element.GetRefMatrices(tn) {
		fmt.Printf("%s\n", name)
		// kp.AddStaticMatrix(name, mat)
		kp.AddDeviceMatrix(name, mat)
	}

	// Allocate arrays
	specs := []builder.ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: builder.Float64, Alignment: builder.NoAlignment},
		{Name: "Ur", Size: int64(totalNodes * 8), DataType: builder.Float64,
			Alignment: builder.NoAlignment, IsOutput: true},
	}
	err := kp.AllocateDeviceMatrices()
	if err != nil {
		t.Fatalf("Failed to allocate device matrices: %v", err)
	}

	err = kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize test data
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = 2. * float64(i%10)
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Kernel using differentiation matrix
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void differentiate(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* Ur = Ur_PART(part);
		MATMUL_Dr_NudgTet1(U, Ur, K[part]);
	}
}
`, Np, kp.GenerateKernelSignature())

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute differentiation
	err = kp.RunKernel("differentiate", "U", "Ur")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results make sense (not checking exact values, just sanity)
	result := make([]float64, totalNodes)
	kp.GetMemory("Ur").CopyTo(unsafe.Pointer(&result[0]), int64(totalNodes*8))

	expected := make([]float64, totalNodes)
	for i := range expected {
		expected[i] = 1.
	}

	assert.InDeltaSlicef(t, expected, result, 1.e-8, "")
}
