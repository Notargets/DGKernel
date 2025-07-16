package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/notargets/gocca"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
	"time"
)

func TestTetNudgMatCopy(t *testing.T) {
	order := 2
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+4)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	U := mat.NewDense(Np, Ktot, nil)
	fOrder := float64(order)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			x, y, z := tn.X.At(j, K), tn.Y.At(j, K), tn.Z.At(j, K)
			xP, yP, zP := math.Pow(x, fOrder), math.Pow(y, fOrder), math.Pow(z, fOrder)
			U.Set(j, K, xP+yP+zP)
		}
	}

	Dx := make([]float64, totalNodes)
	// Add array parameters
	params = append(params,
		builder.Input("U").Bind(U),
		builder.Input("Rx").Bind(tn.Rx),
		builder.Output("Dx").Bind(Dx),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
	)

	// Phase 1: Define bindings
	err := kp.DefineBindings(params...)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 2: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 3: Configure kernel for execution
	kernelName := "differentiate"
	config, err := kp.ConfigureKernel(kernelName,
		kp.Param("U").CopyTo(),
		kp.Param("Rx").CopyTo(),
		kp.Param("Dx").CopyBack(),
		kp.Param("Dr_"+props.ShortName), // Reference the matrix
		kp.Param("Ur"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature and build kernel
	signature, err := config.GetSignature(kp)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const double* U = U_PART(part);
        double* Ur = Ur_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);

        double* Dx = Dx_PART(part);
        const double* Rx = Rx_PART(part);
		for (int k = 0; k < KpartMax; ++k; @inner) {
			if (k < K[part]) {
				for (int n = 0; n < NP; ++n) {
					int i = n + k*NP;
					Dx[i] = Rx[i]*Ur[i];
				}
			}
		}
    }
}
`, tn.Np, kernelName, signature, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Calculate expected result on host
	var Ur mat.Dense
	Ur.Mul(tn.Dr, U)
	DxH := make([]float64, totalNodes)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			i := j + K*Np
			DxH[i] = tn.Rx.At(j, K) * Ur.At(j, K)
		}
	}

	assert.InDeltaSlicef(t, DxH, Dx, 1.e-8, "")
}

func CalculatePhysicalDerivative32(t *testing.T, device *gocca.OCCADevice,
	splits []int, tn *TetNudgMesh,
	U, Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz []mat.Matrix) (
	DuDx, DuDy, DuDz []mat.Matrix, elapsed time.Duration) {

	var (
		Np, K, Ktot, totalNodes int
	)
	for _, m := range U {
		Np, K = m.Dims()
		totalNodes += Np * K
		Ktot += K
	}

	kp := runner.NewRunner(device, builder.Config{
		K: splits,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	props := tn.GetProperties()
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+16)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	DuDx = splitMatrix(splits, mat.NewDense(Np, Ktot, nil))
	DuDy = splitMatrix(splits, mat.NewDense(Np, Ktot, nil))
	DuDz = splitMatrix(splits, mat.NewDense(Np, Ktot, nil))

	// Add array parameters with float32 conversion
	params = append(params,
		builder.Input("U").Bind(U),
		builder.Input("Rx").Bind(Rx).Convert(builder.Float32),
		builder.Input("Sx").Bind(Sx).Convert(builder.Float32),
		builder.Input("Tx").Bind(Tx).Convert(builder.Float32),
		builder.Input("Ry").Bind(Ry).Convert(builder.Float32),
		builder.Input("Sy").Bind(Sy).Convert(builder.Float32),
		builder.Input("Ty").Bind(Ty).Convert(builder.Float32),
		builder.Input("Rz").Bind(Rz).Convert(builder.Float32),
		builder.Input("Sz").Bind(Sz).Convert(builder.Float32),
		builder.Input("Tz").Bind(Tz).Convert(builder.Float32),
		builder.Output("DuDx").Bind(DuDx).Convert(builder.Float32),
		builder.Output("DuDy").Bind(DuDy).Convert(builder.Float32),
		builder.Output("DuDz").Bind(DuDz).Convert(builder.Float32),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Us").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Ut").Type(builder.Float64).Size(totalNodes),
	)

	// Phase 1: Define bindings
	err := kp.DefineBindings(params...)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 2: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 3: Configure kernel
	kernelName := "differentiate"
	configParams := []*runner.ParamConfig{
		kp.Param("U").CopyTo(),
		kp.Param("Rx").CopyTo(),
		kp.Param("Sx").CopyTo(),
		kp.Param("Tx").CopyTo(),
		kp.Param("Ry").CopyTo(),
		kp.Param("Sy").CopyTo(),
		kp.Param("Ty").CopyTo(),
		kp.Param("Rz").CopyTo(),
		kp.Param("Sz").CopyTo(),
		kp.Param("Tz").CopyTo(),
		kp.Param("DuDx").CopyBack(),
		kp.Param("DuDy").CopyBack(),
		kp.Param("DuDz").CopyBack(),
		kp.Param("Dr_" + props.ShortName),
		kp.Param("Ds_" + props.ShortName),
		kp.Param("Dt_" + props.ShortName),
		kp.Param("Ur"),
		kp.Param("Us"),
		kp.Param("Ut"),
	}

	config, err := kp.ConfigureKernel(kernelName, configParams...)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature and build kernel
	signature, err := config.GetSignature(kp)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const double* U = U_PART(part);
        double* Ur = Ur_PART(part);
        double* Us = Us_PART(part);
        double* Ut = Ut_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
        MATMUL_Ds_%s(U, Us, K[part]);
        MATMUL_Dt_%s(U, Ut, K[part]);

        double* DuDx = DuDx_PART(part);
        double* DuDy = DuDy_PART(part);
        double* DuDz = DuDz_PART(part);
        const float* Rx = Rx_PART(part);
        const float* Sx = Sx_PART(part);
        const float* Tx = Tx_PART(part);
        const float* Ry = Ry_PART(part);
        const float* Sy = Sy_PART(part);
        const float* Ty = Ty_PART(part);
        const float* Rz = Rz_PART(part);
        const float* Sz = Sz_PART(part);
        const float* Tz = Tz_PART(part);
		// Single partition means we can safely use KpartMax as K[part]
        // ************************************************************
		// *** This computation is happening in column-major format ***
        // ************************************************************
		for (int i = 0; i < NP*KpartMax; ++i; @inner) {
			DuDx[i] = Rx[i]*Ur[i] + Sx[i]*Us[i] + Tx[i]*Ut[i];
			DuDy[i] = Ry[i]*Ur[i] + Sy[i]*Us[i] + Ty[i]*Ut[i];
			DuDz[i] = Rz[i]*Ur[i] + Sz[i]*Us[i] + Tz[i]*Ut[i];
		}
        // ************************************************************
		// ** When DuDx... are copied back to host they r transposed **
        // ************************************************************
    }
}
`, tn.Np, kernelName, signature,
		props.ShortName, props.ShortName, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Time the execution
	start := time.Now()
	err = kp.ExecuteKernel(kernelName)
	elapsed = time.Since(start)

	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	return
}

// Helper functions from the original test file

func CreateTestSolutionPolynomial(tn *TetNudgMesh) (U *mat.Dense) {
	var (
		order = tn.N
		Np    = tn.Np
		Ktot  = tn.K
	)
	U = mat.NewDense(Np, Ktot, nil)
	fOrder := float64(order)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			x, y, z := tn.X.At(j, K), tn.Y.At(j, K), tn.Z.At(j, K)
			xP, yP, zP := math.Pow(x, fOrder), math.Pow(y, fOrder), math.Pow(z, fOrder)
			U.Set(j, K, xP+yP+zP)
		}
	}
	return
}

func CreateTestSolutionPolynomialDerivative(tn *TetNudgMesh) (
	DuDx, DuDy, DuDz *mat.Dense) {
	var (
		order = tn.N
		Np    = tn.Np
		Ktot  = tn.K
	)
	DuDx = mat.NewDense(Np, Ktot, nil)
	DuDy = mat.NewDense(Np, Ktot, nil)
	DuDz = mat.NewDense(Np, Ktot, nil)
	fOrder := float64(order)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			x, y, z := tn.X.At(j, K), tn.Y.At(j, K), tn.Z.At(j, K)
			dxP := float64(fOrder) * math.Pow(x, fOrder-1)
			dyP := float64(fOrder) * math.Pow(y, fOrder-1)
			dzP := float64(fOrder) * math.Pow(z, fOrder-1)
			DuDx.Set(j, K, dxP)
			DuDy.Set(j, K, dyP)
			DuDz.Set(j, K, dzP)
		}
	}
	return
}

func calcPhysicalDerivative(U, Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz, Dr, Ds, Dt mat.Matrix) (DuDx, DuDy, DuDz []float64) {
	rows, cols := U.Dims()
	var (
		Ur, Us, Ut mat.Dense
	)
	Ur.Mul(Dr, U)
	Us.Mul(Ds, U)
	Ut.Mul(Dt, U)
	totalNodes := rows * cols
	DuDx = make([]float64, totalNodes)
	DuDy = make([]float64, totalNodes)
	DuDz = make([]float64, totalNodes)
	for K := 0; K < cols; K++ {
		for j := 0; j < rows; j++ {
			i := j + K*rows
			DuDx[i] = Rx.At(j, K)*Ur.At(j, K) + Sx.At(j, K)*Us.At(j, K) + Tx.At(j, K)*Ut.At(j, K)
			DuDy[i] = Ry.At(j, K)*Ur.At(j, K) + Sy.At(j, K)*Us.At(j, K) + Ty.At(j, K)*Ut.At(j, K)
			DuDz[i] = Rz.At(j, K)*Ur.At(j, K) + Sz.At(j, K)*Us.At(j, K) + Tz.At(j, K)*Ut.At(j, K)
		}
	}
	return
}

func getMatrixData(m mat.Matrix) []float64 {
	switch v := m.(type) {
	case *mat.Dense:
		return v.RawMatrix().Data

	case *mat.VecDense:
		return v.RawVector().Data

	case *mat.SymDense:
		return v.RawSymmetric().Data

	case *mat.TriDense:
		return v.RawTriangular().Data

	default:
		dense := mat.DenseCopyOf(m)
		return dense.RawMatrix().Data
	}
}

func splitMatrix(splits []int, matrix mat.Matrix) (splitMat []mat.Matrix) {
	// Total of two data copies used in this transform, resulting in a split
	var (
		rows, _ = matrix.Dims()
	)
	// This makes one copy of the matrix data within getMatrixData after .T()
	// The splitData slice of slices stores addresses of split regions
	splitData := splitSlice(splits, getMatrixData(matrix.T()))
	splitMat = make([]mat.Matrix, len(splits))
	for i, K := range splits {
		// This makes another copy of the data after .T()
		// The end result of this is a matrix split on columns in row-major form
		splitMat[i] = mat.DenseCopyOf(mat.NewDense(K, rows, splitData[i]).T())
	}
	return
}

func splitSlice(splits []int, slice []float64) (splitSlice [][]float64) {
	var (
		total int
	)
	for _, k := range splits {
		total += k
	}
	if len(slice)%total != 0 {
		fmt.Printf("len(slice) = %d, total = %d\n", len(slice), total)
		panic("split slice is not a multiple of len(slice)")
	}
	stride := len(slice) / total
	splitSlice = make([][]float64, len(splits))
	var iii int
	for i, K := range splits {
		// This avoids a copy by just storing the addresses
		splitSlice[i] = slice[iii : iii+K*stride]
		iii += K * stride
	}
	return
}

func printMatrix(m mat.Matrix) (out string) {
	rows, cols := m.Dims()

	for i := 0; i < rows; i++ {
		out += fmt.Sprintf("[")
		for j := 0; j < cols; j++ {
			out += fmt.Sprintf("%8.3f", m.At(i, j))
			if j < cols-1 {
				out += fmt.Sprintf(" ")
			}
		}
		out += fmt.Sprintf("]\n")
	}
	return
}

func PrintMatrix(name string, m mat.Matrix) {
	fmt.Printf("%s:\n%s", name, printMatrix(m))
}

func TestSplitMatrix(t *testing.T) {
	/*
		We start with a 2,5 rank matrix, with each column sequential:
		[   1.000    3.000    5.000    7.000    9.000]
		[   2.000    4.000    6.000    8.000   10.000]

		The split version should look like two pieces of the same matrix:
		Mat 0
		[   1.000    3.000]
		[   2.000    4.000]
		Mat 1
		[   5.000    7.000    9.000]
		[   6.000    8.000   10.000]
	*/
	kk := []int{2, 3}
	mm := mat.NewDense(2, 5, []float64{1, 3, 5, 7, 9, 2, 4, 6, 8, 10})
	splitMs := splitMatrix(kk, mm)
	if testing.Verbose() {
		fmt.Printf("%s", printMatrix(mm))
		// printMatrix(mm.T())
		for i, m := range splitMs {
			fmt.Printf("Mat %d\n", i)
			fmt.Printf("%s", printMatrix(m))
		}
	}

	assert.InDeltaSlicef(t, splitMs[0].(*mat.Dense).RawMatrix().Data,
		[]float64{1, 3, 2, 4}, 1.e-8, "")
	assert.InDeltaSlicef(t, splitMs[1].(*mat.Dense).RawMatrix().Data,
		[]float64{5, 7, 9, 6, 8, 10}, 1.e-8, "")
}

// Benchmark and performance testing functions

func TestBenchmarkKnownMatrixOperations(t *testing.T) {
	if !testing.Verbose() {
		return
	}

	N := []int{2, 3, 4}
	K := []int{1000, 2000, 4000, 6000, 10000}
	deviceTypes := []string{"CUDA", "OpenMP"}

	for _, deviceType := range deviceTypes {
		fmt.Printf("\n\n === Testing with %s Device ===\n", deviceType)

		var device *gocca.OCCADevice
		if deviceType == "CUDA" {
			device = utils.CreateTestDevice(true) // Force CUDA
		} else {
			device = utils.CreateTestDevice() // Default (OpenMP)
		}

		if device.Mode() != deviceType {
			fmt.Printf("Skipping %s tests - device not available\n", deviceType)
			continue
		}
		defer device.Free()

		var results []struct {
			N, K       int
			Time       time.Duration
			Throughput float64 // GB/s
		}

		for _, n := range N {
			for _, k := range K {
				tn := NewTetNudgMesh(n, "cube-partitioned.neu")
				fmt.Printf("\nTesting N=%d, K=%d\n", n, k)

				// Adjust K to match mesh
				actualK := tn.K
				if actualK != k {
					fmt.Printf("  Adjusted K from %d to %d to match mesh\n", k, actualK)
					k = actualK
				}

				// Create test data
				U := splitMatrix([]int{k}, CreateTestSolutionPolynomial(tn))
				Rx := splitMatrix([]int{k}, tn.Rx)
				Ry := splitMatrix([]int{k}, tn.Ry)
				Rz := splitMatrix([]int{k}, tn.Rz)
				Sx := splitMatrix([]int{k}, tn.Sx)
				Sy := splitMatrix([]int{k}, tn.Sy)
				Sz := splitMatrix([]int{k}, tn.Sz)
				Tx := splitMatrix([]int{k}, tn.Tx)
				Ty := splitMatrix([]int{k}, tn.Ty)
				Tz := splitMatrix([]int{k}, tn.Tz)

				// Run the calculation
				DuDx, DuDy, DuDz, elapsed := CalculatePhysicalDerivative(t, device,
					[]int{k}, tn, U, Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz)
				_, _ = DuDy, DuDz

				// Calculate throughput
				Np := tn.Np
				bytesPerElement := 8 // float64
				// Memory operations: Read U, Rx-Tz (10 arrays), Write DuDx-DuDz (3 arrays)
				totalBytes := int64(Np * k * bytesPerElement * 13)
				throughput := float64(totalBytes) / (elapsed.Seconds() * 1e9) // GB/s

				results = append(results, struct {
					N, K       int
					Time       time.Duration
					Throughput float64
				}{N: n, K: k, Time: elapsed, Throughput: throughput})

				fmt.Printf("  Time: %v, Throughput: %.2f GB/s\n", elapsed, throughput)

				// Verify correctness (spot check)
				if k > 0 && DuDx != nil && len(DuDx) > 0 {
					firstElem := DuDx[0].At(0, 0)
					fmt.Printf("  First element of DuDx: %f\n", firstElem)
				}
			}
		}

		// Print summary table
		fmt.Printf("\n%s Performance Summary:\n", deviceType)
		fmt.Printf("%-5s %-8s %-15s %-15s\n", "N", "K", "Time", "Throughput (GB/s)")
		fmt.Printf("%-5s %-8s %-15s %-15s\n", "---", "---", "----", "-----------------")
		for _, r := range results {
			fmt.Printf("%-5d %-8d %-15v %-15.2f\n", r.N, r.K, r.Time, r.Throughput)
		}
	}
}

func TestBenchmarkFloat32Performance(t *testing.T) {
	if !testing.Verbose() {
		return
	}

	order := 4
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	K := tn.K

	// Test with both device types
	deviceTypes := []bool{true, false} // true = CUDA, false = OpenMP

	for _, useCUDA := range deviceTypes {
		device := utils.CreateTestDevice(useCUDA)
		deviceName := device.Mode()
		defer device.Free()

		fmt.Printf("\n=== %s Device ===\n", deviceName)

		// Create test data
		U := splitMatrix([]int{K}, CreateTestSolutionPolynomial(tn))
		Rx := splitMatrix([]int{K}, tn.Rx)
		Ry := splitMatrix([]int{K}, tn.Ry)
		Rz := splitMatrix([]int{K}, tn.Rz)
		Sx := splitMatrix([]int{K}, tn.Sx)
		Sy := splitMatrix([]int{K}, tn.Sy)
		Sz := splitMatrix([]int{K}, tn.Sz)
		Tx := splitMatrix([]int{K}, tn.Tx)
		Ty := splitMatrix([]int{K}, tn.Ty)
		Tz := splitMatrix([]int{K}, tn.Tz)

		// Run float64 version
		_, _, _, elapsed64 := CalculatePhysicalDerivative(t, device,
			[]int{K}, tn, U, Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz)

		// Run float32 version
		_, _, _, elapsed32 := CalculatePhysicalDerivative32(t, device,
			[]int{K}, tn, U, Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz)

		fmt.Printf("Float64 time: %v\n", elapsed64)
		fmt.Printf("Float32 time: %v\n", elapsed32)
		fmt.Printf("Speedup: %.2fx\n", elapsed64.Seconds()/elapsed32.Seconds())
	}
}

func TestTetNudgMatCopyMatrixReturn(t *testing.T) {
	order := 1
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Calculate expected result on host
	U := CreateTestSolutionPolynomial(tn)
	var Ur mat.Dense
	Ur.Mul(tn.Dr, U)
	// DxH is the host calculated value to be compared with the device value
	DxH := mat.NewDense(Np, Ktot, nil)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			DxH.Set(j, K, Ur.At(j, K))
		}
	}

	// Collect all element matrices
	matrices := tn.GetRefMatrices()

	// Phase 1: Define all bindings (once)
	params := make([]*builder.ParamBuilder, 0, len(matrices)+4)

	// Add basic element matrices as parameters - these will be allocated as
	// device matrices
	for name, Mat := range matrices {
		fmt.Printf("Matrix name: %s\n", name)
		params = append(params, builder.Input(name).Bind(Mat).ToMatrix().Static())
	}

	// Add array parameters
	// IMPORTANT: We bind matrices without .ToMatrix() so they are treated as regular arrays
	// This means they WILL be automatically transposed during copy operations
	// Output matrix
	Dx := mat.NewDense(Np, Ktot, nil)
	Uw := U.T()
	params = append(params,
		// builder.Input("U").Bind(U),      // Will be transposed on CopyTo
		builder.Input("U").Bind(Uw),     // Will be transposed on CopyTo
		builder.Input("Rx").Bind(tn.Rx), // Will be transposed on CopyTo
		builder.Output("Dx").Bind(Dx),   // Will be transposed on CopyBack
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
	)

	// Define all bindings
	err := kp.DefineBindings(params...)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 2: Allocate device memory (once)
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 3: Configure kernel for execution
	kernelName := "differentiate"
	config, err := kp.ConfigureKernel(kernelName,
		kp.Param("U").CopyTo(),          // Copy and transpose to column-major
		kp.Param("Rx").CopyTo(),         // Copy and transpose to column-major
		kp.Param("Dx").CopyBack(),       // Copy back and transpose to row-major
		kp.Param("Dr_"+props.ShortName), // Bind the backing array for the MATMUL macro
		kp.Param("Ur"),                  // Temp array, no copy needed
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature and build kernel
	signature, err := config.GetSignature(kp)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const double* U = U_PART(part);
        double* Ur = Ur_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);

        double* Dx = Dx_PART(part);
        const double* Rx = Rx_PART(part);
        // ************************************************************
        // *** This computation is happening in column-major format ***
        // ************************************************************
        for (int k = 0; k < KpartMax; ++k; @inner) {
            if (k < K[part]) {  // Bounds check for partition size
        		for (int n = 0; n < NP; ++n) {
                	int i = n + k*NP;  // Column-major indexing
                    //Dx[i] = Rx[i]*Ur[i];
                    Dx[i] = Ur[i];
                }
            }
        }
        // ************************************************************
        // *** When Dx is copied back to host, it will be transposed **
        // ************************************************************
    }
}
`, tn.Np, kernelName, signature, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel with automatic memory operations
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}
}

func TestTetNudgPhysicalDerivative(t *testing.T) {
	order := 2
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+16)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	U := CreateTestSolutionPolynomial(tn)
	DuDxExpected, DuDyExpected, DuDzExpected :=
		CreateTestSolutionPolynomialDerivative(tn)

	// Use matrices for the host output to get the automatic conversion to
	// row-major storage format
	DuDx := mat.NewDense(Np, Ktot, nil)
	DuDy := mat.NewDense(Np, Ktot, nil)
	DuDz := mat.NewDense(Np, Ktot, nil)

	// Add array parameters
	params = append(params,
		builder.Input("U").Bind(U),
		builder.Input("Rx").Bind(tn.Rx),
		builder.Input("Sx").Bind(tn.Sx),
		builder.Input("Tx").Bind(tn.Tx),
		builder.Input("Ry").Bind(tn.Ry),
		builder.Input("Sy").Bind(tn.Sy),
		builder.Input("Ty").Bind(tn.Ty),
		builder.Input("Rz").Bind(tn.Rz),
		builder.Input("Sz").Bind(tn.Sz),
		builder.Input("Tz").Bind(tn.Tz),
		builder.Output("DuDx").Bind(DuDx),
		builder.Output("DuDy").Bind(DuDy),
		builder.Output("DuDz").Bind(DuDz),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Us").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Ut").Type(builder.Float64).Size(totalNodes),
	)

	// Phase 1: Define bindings
	err := kp.DefineBindings(params...)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 2: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 3: Configure kernel
	kernelName := "differentiate"
	configParams := []*runner.ParamConfig{
		kp.Param("U").CopyTo(),
		kp.Param("Rx").CopyTo(),
		kp.Param("Sx").CopyTo(),
		kp.Param("Tx").CopyTo(),
		kp.Param("Ry").CopyTo(),
		kp.Param("Sy").CopyTo(),
		kp.Param("Ty").CopyTo(),
		kp.Param("Rz").CopyTo(),
		kp.Param("Sz").CopyTo(),
		kp.Param("Tz").CopyTo(),
		kp.Param("DuDx").CopyBack(),
		kp.Param("DuDy").CopyBack(),
		kp.Param("DuDz").CopyBack(),
		kp.Param("Dr_" + props.ShortName),
		kp.Param("Ds_" + props.ShortName),
		kp.Param("Dt_" + props.ShortName),
		kp.Param("Ur"),
		kp.Param("Us"),
		kp.Param("Ut"),
	}

	config, err := kp.ConfigureKernel(kernelName, configParams...)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature and build kernel
	signature, err := config.GetSignature(kp)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const double* U = U_PART(part);
        double* Ur = Ur_PART(part);
        double* Us = Us_PART(part);
        double* Ut = Ut_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
        MATMUL_Ds_%s(U, Us, K[part]);
        MATMUL_Dt_%s(U, Ut, K[part]);

        double* DuDx = DuDx_PART(part);
        double* DuDy = DuDy_PART(part);
        double* DuDz = DuDz_PART(part);
        const double* Rx = Rx_PART(part);
        const double* Sx = Sx_PART(part);
        const double* Tx = Tx_PART(part);
        const double* Ry = Ry_PART(part);
        const double* Sy = Sy_PART(part);
        const double* Ty = Ty_PART(part);
        const double* Rz = Rz_PART(part);
        const double* Sz = Sz_PART(part);
        const double* Tz = Tz_PART(part);
        // ************************************************************
		// *** This computation is happening in column-major format ***
        // ************************************************************
		for (int k = 0; k < KpartMax; ++k; @inner) {
			if (k < K[part]) {
				for (int n = 0; n < NP; ++n) {
					int i = n + k*NP;
					DuDx[i] = Rx[i]*Ur[i] + Sx[i]*Us[i] + Tx[i]*Ut[i];
					DuDy[i] = Ry[i]*Ur[i] + Sy[i]*Us[i] + Ty[i]*Ut[i];
					DuDz[i] = Rz[i]*Ur[i] + Sz[i]*Us[i] + Tz[i]*Ut[i];
				}
			}
		}
        // ************************************************************
		// ** When DuDx... are copied back to host they r transposed **
        // ************************************************************
    }
}
`, tn.Np, kernelName, signature,
		props.ShortName, props.ShortName, props.ShortName)
	// for K := 0; K < cols; K++ {
	// 	for j := 0; j < rows; j++ {
	// 		i := j + K*rows
	// 		DuDx[i] = Rx.At(j, K)*Ur.At(j, K) + Sx.At(j, K)*Us.At(j, K) + Tx.At(j, K)*Ut.At(j, K)
	// 		DuDy[i] = Ry.At(j, K)*Ur.At(j, K) + Sy.At(j, K)*Us.At(j, K) + Ty.At(j, K)*Ut.At(j, K)
	// 		DuDz[i] = Rz.At(j, K)*Ur.At(j, K) + Sz.At(j, K)*Us.At(j, K) + Tz.At(j, K)*Ut.At(j, K)
	// 	}
	// }

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Calculate host reference
	// DuDxH, DuDyH, DuDzH := calcPhysicalDerivative(U, tn.Rx, tn.Ry, tn.Rz, tn.Sx,
	// 	tn.Sy, tn.Sz, tn.Tx, tn.Ty, tn.Tz, tn.Dr, tn.Ds, tn.Dt)
	// PrintMatrix("DuDxH", mat.NewDense(Np, Ktot, DuDxH))
	// PrintMatrix("DuDx", DuDx)
	// PrintMatrix("DuDxExpected", DuDxExpected)
	// os.Exit(1)
	// assert.InDeltaSlicef(t, DuDxExpected.RawMatrix().Data, DuDxH, 1.e-8, "")
	// assert.InDeltaSlicef(t, DuDyExpected.RawMatrix().Data, DuDyH, 1.e-8, "")
	// assert.InDeltaSlicef(t, DuDzExpected.RawMatrix().Data, DuDzH, 1.e-8, "")
	// fmt.Println("Host calculation of physical derivative validates")

	assert.InDeltaSlicef(t, DuDxExpected.RawMatrix().Data, DuDx.RawMatrix().Data, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDyExpected.RawMatrix().Data, DuDy.RawMatrix().Data, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDzExpected.RawMatrix().Data, DuDz.RawMatrix().Data, 1.e-8, "")
	fmt.Println("Device calculation of physical derivative validates")
}

func CalculatePhysicalDerivative(t *testing.T, device *gocca.OCCADevice,
	splits []int, tn *TetNudgMesh,
	U, Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz []mat.Matrix) (
	DuDx, DuDy, DuDz []mat.Matrix, elapsed time.Duration) {

	var (
		Np, K, Ktot, totalNodes int
	)
	for _, m := range U {
		Np, K = m.Dims()
		totalNodes += Np * K
		Ktot += K
	}

	kp := runner.NewRunner(device, builder.Config{
		K: splits,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	props := tn.GetProperties()
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+16)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	DuDx = splitMatrix(splits, mat.NewDense(Np, Ktot, nil))
	DuDy = splitMatrix(splits, mat.NewDense(Np, Ktot, nil))
	DuDz = splitMatrix(splits, mat.NewDense(Np, Ktot, nil))

	// Add array parameters
	params = append(params,
		builder.Input("U").Bind(U),
		builder.Input("Rx").Bind(Rx),
		builder.Input("Sx").Bind(Sx),
		builder.Input("Tx").Bind(Tx),
		builder.Input("Ry").Bind(Ry),
		builder.Input("Sy").Bind(Sy),
		builder.Input("Ty").Bind(Ty),
		builder.Input("Rz").Bind(Rz),
		builder.Input("Sz").Bind(Sz),
		builder.Input("Tz").Bind(Tz),
		builder.Output("DuDx").Bind(DuDx),
		builder.Output("DuDy").Bind(DuDy),
		builder.Output("DuDz").Bind(DuDz),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Us").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Ut").Type(builder.Float64).Size(totalNodes),
	)

	// Phase 1: Define bindings
	err := kp.DefineBindings(params...)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 2: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 3: Configure kernel
	kernelName := "differentiate"
	configParams := []*runner.ParamConfig{
		kp.Param("U").CopyTo(),
		kp.Param("Rx").CopyTo(),
		kp.Param("Sx").CopyTo(),
		kp.Param("Tx").CopyTo(),
		kp.Param("Ry").CopyTo(),
		kp.Param("Sy").CopyTo(),
		kp.Param("Ty").CopyTo(),
		kp.Param("Rz").CopyTo(),
		kp.Param("Sz").CopyTo(),
		kp.Param("Tz").CopyTo(),
		kp.Param("DuDx").CopyBack(),
		kp.Param("DuDy").CopyBack(),
		kp.Param("DuDz").CopyBack(),
		kp.Param("Dr_" + props.ShortName),
		kp.Param("Ds_" + props.ShortName),
		kp.Param("Dt_" + props.ShortName),
		kp.Param("Ur"),
		kp.Param("Us"),
		kp.Param("Ut"),
	}

	config, err := kp.ConfigureKernel(kernelName, configParams...)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature and build kernel
	signature, err := config.GetSignature(kp)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const double* U = U_PART(part);
        double* Ur = Ur_PART(part);
        double* Us = Us_PART(part);
        double* Ut = Ut_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
        MATMUL_Ds_%s(U, Us, K[part]);
        MATMUL_Dt_%s(U, Ut, K[part]);

        double* DuDx = DuDx_PART(part);
        double* DuDy = DuDy_PART(part);
        double* DuDz = DuDz_PART(part);
        const double* Rx = Rx_PART(part);
        const double* Sx = Sx_PART(part);
        const double* Tx = Tx_PART(part);
        const double* Ry = Ry_PART(part);
        const double* Sy = Sy_PART(part);
        const double* Ty = Ty_PART(part);
        const double* Rz = Rz_PART(part);
        const double* Sz = Sz_PART(part);
        const double* Tz = Tz_PART(part);
		// Single partition means we can safely use KpartMax as K[part]
        // ************************************************************
		// *** This computation is happening in column-major format ***
        // ************************************************************
		for (int i = 0; i < NP*KpartMax; ++i; @inner) {
			DuDx[i] = Rx[i]*Ur[i] + Sx[i]*Us[i] + Tx[i]*Ut[i];
			DuDy[i] = Ry[i]*Ur[i] + Sy[i]*Us[i] + Ty[i]*Ut[i];
			DuDz[i] = Rz[i]*Ur[i] + Sz[i]*Us[i] + Tz[i]*Ut[i];
		}
        // ************************************************************
		// ** When DuDx... are copied back to host they r transposed **
        // ************************************************************
    }
}
`, tn.Np, kernelName, signature,
		props.ShortName, props.ShortName, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Time the execution
	start := time.Now()
	err = kp.ExecuteKernel(kernelName)
	elapsed = time.Since(start)

	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	return
}
