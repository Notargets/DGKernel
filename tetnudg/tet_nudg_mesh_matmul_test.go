package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
	"unsafe"
)

func TestTetNudgMatmul(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 4
	k := []int{5, 10}
	totalNodes := 15 * np

	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	tn := NewTetNudgMesh(1, "cube-partitioned.neu")
	for name, mat := range element.GetRefMatrices(tn) {
		fmt.Printf("%s\n", name)
		kp.AddStaticMatrix(name, mat)
	}

	// Allocate arrays
	specs := []builder.ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: builder.Float64, Alignment: builder.NoAlignment},
		{Name: "Ur", Size: int64(totalNodes * 8), DataType: builder.Float64, Alignment: builder.NoAlignment},
	}
	err := kp.AllocateArrays(specs)
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
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* Ur_global,
	const int_t* Ur_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* Ur = Ur_PART(part);
		MATMUL_Dr_NudgTet1(U, Ur, K[part]);
	}
}
`, np)

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

	// Check that we got non-zero results
	hasNonZero := false
	for i := 0; i < totalNodes; i++ {
		if math.Abs(result[i]) > 1e-10 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Differentiation produced all zeros")
	} else {
		fmt.Println(U)
		fmt.Println(result)
	}
}
