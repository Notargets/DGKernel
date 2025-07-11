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
	"sort"
	"testing"
)

func TestTetNudgMatCopy(t *testing.T) {
	order := 2
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	// device := utils.CreateTestDevice(true)
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+2)

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
		builder.Input("U").Bind(U).CopyTo(),
		builder.Input("Rx").Bind(tn.Rx).CopyTo(),
		builder.Output("Dx").Bind(Dx).CopyBack(),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
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
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);

        real_t* Dx = Dx_PART(part);
        const real_t* Rx = Rx_PART(part);
		// Single partition means we can safely use KpartMax as K[part]
		for (int i = 0; i < NP*KpartMax; ++i; @inner) {
			Dx[i] = Rx[i]*Ur[i];
		}
    }
}
`, tn.Np, kernelName, signature, props.ShortName)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	// Execute differentiation
	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}
}

func setupArrays(Np, Ktot int) [][]float64 {
	U := make([]float64, Np*Ktot)
	Rx := make([]float64, Np*Ktot)
	Dx := make([]float64, Np*Ktot)
	for i := 0; i < Ktot*Np; i++ {
		U[i] = float64(i)
		Rx[i] = float64(i)
	}
	return [][]float64{U, Rx, Dx}
}

func TestTetNudgArrayReturn(t *testing.T) {
	Np := 10
	Ktot := 565
	for ii, name := range []string{"OpenMP", "CUDA"} {
		fmt.Printf("Running %s test\n", name)
		var device *gocca.OCCADevice
		if ii%2 == 0 {
			device = utils.CreateTestDevice()
		} else {
			device = utils.CreateTestDevice(true)
		}
		defer device.Free()

		arrays := setupArrays(Np, Ktot)
		U, Rx, Dx := arrays[0], arrays[1], arrays[2]

		kp := runner.NewRunner(device, builder.Config{
			K:         []int{Ktot},
			FloatType: builder.Float64,
		})
		defer kp.Free()

		// Add array parameters
		params := make([]*builder.ParamBuilder, 0)
		params = append(params,
			// Since U and Rx are matrices, they will be transposed on CopyTo()
			builder.Input("U").Bind(U).CopyTo(),
			builder.Input("Rx").Bind(Rx).CopyTo(),
			builder.Output("Dx").Bind(Dx).CopyBack(),
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
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Dx = Dx_PART(part);
        const real_t* Rx = Rx_PART(part);
        
        // Use nested loops to avoid exceeding CUDA thread block limits
        for (int n = 0; n < NP; ++n; @inner) {
            for (int k = 0; k < KpartMax; ++k) {
                int i = n + k*NP;  // Column-major indexing
                if (k < K[part]) {  // Bounds check for partition size
                    Dx[i] = Rx[i]*U[i];
                }
            }
        }
    }
}
`, Np, kernelName, signature)

		_, err = kp.BuildKernel(kernelSource, kernelName)
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}
		// Execute differentiation
		err = kp.RunKernel(kernelName)
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}

		// fmt.Println(Dx)
		var sum float64
		for i := 0; i < Np*Ktot; i++ {
			sum += Dx[i] / float64(Np*Ktot)
		}
		sum /= float64(Np * Ktot)
		fmt.Printf("Sum = %f\n", sum)
	}
}

func TestTetNudgMatCopyMatrixReturn(t *testing.T) {
	order := 2
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	// device := utils.CreateTestDevice(true)
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+2)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	// Use all host side matrix based computations.
	// Note that these result in a storage of the answers in row-major format
	U := mat.NewDense(Np, Ktot, nil)
	fOrder := float64(order)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			x, y, z := tn.X.At(j, K), tn.Y.At(j, K), tn.Z.At(j, K)
			xP, yP, zP := math.Pow(x, fOrder), math.Pow(y, fOrder), math.Pow(z, fOrder)
			U.Set(j, K, xP+yP+zP)
		}
	}
	var Ur mat.Dense
	Ur.Mul(tn.Dr, U)
	// DxH is the host calculated value to be compared with the device value
	DxH := mat.NewDense(Np, Ktot, nil)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			DxH.Set(j, K, tn.Rx.At(j, K)*Ur.At(j, K))
		}
	}

	Dx := mat.NewDense(Np, Ktot, nil)
	// Add array parameters
	params = append(params,
		// Since U and Rx are matrices, they will be transposed on CopyTo()
		builder.Input("U").Bind(U).CopyTo(),
		builder.Input("Rx").Bind(tn.Rx).CopyTo(),
		// Since Dx is a matrix, it will be transposed on CopyBack()
		builder.Output("Dx").Bind(Dx).CopyBack(),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
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
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);

        real_t* Dx = Dx_PART(part);
        const real_t* Rx = Rx_PART(part);
		// Single partition means we can safely use KpartMax as K[part]
        // ************************************************************
		// *** This computation is happening in column-major format ***
        // ************************************************************
        for (int n = 0; n < NP; ++n; @inner) {
            for (int k = 0; k < KpartMax; ++k) {
                int i = n + k*NP;  // Column-major indexing
                if (k < K[part]) {  // Bounds check for partition size
			// DxH.Set(j, K, tn.Rx.At(j, K)*Ur.At(j, K))
			Dx[i] = Rx[i]*Ur[i];
		} } }
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
	// Execute differentiation
	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Dx is now in row-major format and can be compared directly to the host
	// matrix result
	// fmt.Println(Dx)
	// os.Exit(1)
	assert.InDeltaSlicef(t, DxH.RawMatrix().Data, Dx.RawMatrix().Data, 1.e-8, "")
}

func TestTetNudgPhysicalDerivative(t *testing.T) {
	order := 4
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Np := tn.Np
	Ktot := tn.K
	totalNodes := Np * Ktot
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	// device := utils.CreateTestDevice(true)
	defer device.Free()

	k := []int{Ktot}
	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+2)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	U := mat.NewDense(Np, Ktot, nil)
	DuDxExpected := mat.NewDense(Np, Ktot, nil)
	DuDyExpected := mat.NewDense(Np, Ktot, nil)
	DuDzExpected := mat.NewDense(Np, Ktot, nil)
	fOrder := float64(order)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			x, y, z := tn.X.At(j, K), tn.Y.At(j, K), tn.Z.At(j, K)
			xP, yP, zP := math.Pow(x, fOrder), math.Pow(y, fOrder), math.Pow(z, fOrder)
			U.Set(j, K, xP+yP+zP)
			dxP := float64(fOrder) * math.Pow(x, fOrder-1)
			dyP := float64(fOrder) * math.Pow(y, fOrder-1)
			dzP := float64(fOrder) * math.Pow(z, fOrder-1)
			DuDxExpected.Set(j, K, dxP)
			DuDyExpected.Set(j, K, dyP)
			DuDzExpected.Set(j, K, dzP)
		}
	}

	// Use matrices for the host output to get the automatic conversion to
	// row-major storage format
	DuDx := mat.NewDense(Np, Ktot, nil)
	DuDy := mat.NewDense(Np, Ktot, nil)
	DuDz := mat.NewDense(Np, Ktot, nil)
	// Add array parameters
	params = append(params,
		builder.Input("U").Bind(U).CopyTo(),
		builder.Input("Rx").Bind(tn.Rx).CopyTo(),
		builder.Input("Sx").Bind(tn.Sx).CopyTo(),
		builder.Input("Tx").Bind(tn.Tx).CopyTo(),
		builder.Input("Ry").Bind(tn.Ry).CopyTo(),
		builder.Input("Sy").Bind(tn.Sy).CopyTo(),
		builder.Input("Ty").Bind(tn.Ty).CopyTo(),
		builder.Input("Rz").Bind(tn.Rz).CopyTo(),
		builder.Input("Sz").Bind(tn.Sz).CopyTo(),
		builder.Input("Tz").Bind(tn.Tz).CopyTo(),
		builder.Output("DuDx").Bind(DuDx).CopyBack(),
		builder.Output("DuDy").Bind(DuDy).CopyBack(),
		builder.Output("DuDz").Bind(DuDz).CopyBack(),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Us").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Ut").Type(builder.Float64).Size(totalNodes),
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
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        real_t* Us = Us_PART(part);
        real_t* Ut = Ut_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
        MATMUL_Ds_%s(U, Us, K[part]);
        MATMUL_Dt_%s(U, Ut, K[part]);

        real_t* DuDx = DuDx_PART(part);
        real_t* DuDy = DuDy_PART(part);
        real_t* DuDz = DuDz_PART(part);
        const real_t* Rx = Rx_PART(part);
        const real_t* Sx = Sx_PART(part);
        const real_t* Tx = Tx_PART(part);
        const real_t* Ry = Ry_PART(part);
        const real_t* Sy = Sy_PART(part);
        const real_t* Ty = Ty_PART(part);
        const real_t* Rz = Rz_PART(part);
        const real_t* Sz = Sz_PART(part);
        const real_t* Tz = Tz_PART(part);
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
	// Execute differentiation
	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	DuDxH, DuDyH, DuDzH := calcPhysicalDerivative(U, tn.Rx, tn.Ry, tn.Rz, tn.Sx,
		tn.Sy, tn.Sz, tn.Tx, tn.Ty, tn.Tz, tn.Dr, tn.Ds, tn.Dt)

	assert.InDeltaSlicef(t, DuDxExpected.RawMatrix().Data, DuDxH, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDyExpected.RawMatrix().Data, DuDyH, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDzExpected.RawMatrix().Data, DuDzH, 1.e-8, "")
	fmt.Println("Host calculation of physical derivative validates")

	assert.InDeltaSlicef(t, DuDxExpected.RawMatrix().Data, DuDx.RawMatrix().Data, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDyExpected.RawMatrix().Data, DuDy.RawMatrix().Data, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDzExpected.RawMatrix().Data, DuDz.RawMatrix().Data, 1.e-8, "")
	fmt.Println("Device calculation of physical derivative validates")
}

func TestTetNudgPhysicalDerivativePartitionedMesh(t *testing.T) {
	order := 4
	tn := NewTetNudgMesh(order, "cube-partitioned.neu")
	Ktot := tn.K
	Np := tn.Np
	totalNodes := Np * Ktot
	k := make([]int, 0)
	if len(tn.Mesh.EToP) != 0 {
		// Count elements per partition
		partitionCounts := make(map[int]int)
		maxPartID := -1
		for _, p := range tn.Mesh.EToP {
			if p >= 0 {
				partitionCounts[p]++
				if p > maxPartID {
					maxPartID = p
				}
			}
		}
		var pNames []int
		var elementSum int
		for p, count := range partitionCounts {
			pNames = append(pNames, p)
			elementSum += count
		}
		sort.Ints(pNames)
		fmt.Printf("Total elements: %d, Partition counts: %v\n",
			elementSum, partitionCounts)
		if elementSum != Ktot {
			panic("Wrong partition counts")
		}
		fmt.Printf("Partition names: %v\n", pNames)
		for _, n := range pNames {
			k = append(k, partitionCounts[n])
		}
	} else {
		k = []int{Ktot}
	}
	fmt.Printf("Partition K: %v\n", k)
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
	// device := utils.CreateTestDevice(true)
	defer device.Free()

	kp := runner.NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	// defer kp.Free()

	// Collect all matrices into param builders
	matrices := tn.GetRefMatrices()
	params := make([]*builder.ParamBuilder, 0, len(matrices)+2)

	// Add matrices as parameters
	for name, mat := range matrices {
		params = append(params, builder.Input(name).Bind(mat).ToMatrix())
	}

	Uw := mat.NewDense(Np, Ktot, nil)
	DuDxExpected := mat.NewDense(Np, Ktot, nil)
	DuDyExpected := mat.NewDense(Np, Ktot, nil)
	DuDzExpected := mat.NewDense(Np, Ktot, nil)
	fOrder := float64(order)
	for K := 0; K < Ktot; K++ {
		for j := 0; j < Np; j++ {
			x, y, z := tn.X.At(j, K), tn.Y.At(j, K), tn.Z.At(j, K)
			xP, yP, zP := math.Pow(x, fOrder), math.Pow(y, fOrder), math.Pow(z, fOrder)
			Uw.Set(j, K, xP+yP+zP)
			dxP := float64(fOrder) * math.Pow(x, fOrder-1)
			dyP := float64(fOrder) * math.Pow(y, fOrder-1)
			dzP := float64(fOrder) * math.Pow(z, fOrder-1)
			DuDxExpected.Set(j, K, dxP)
			DuDyExpected.Set(j, K, dyP)
			DuDzExpected.Set(j, K, dzP)
		}
	}

	// Use matrices for the host output to get the automatic conversion to
	// row-major storage format
	DuDx := splitMatrix(k, mat.NewDense(Np, Ktot, nil))
	DuDy := splitMatrix(k, mat.NewDense(Np, Ktot, nil))
	DuDz := splitMatrix(k, mat.NewDense(Np, Ktot, nil))
	U := splitMatrix(k, Uw)
	Rx, Ry, Rz := splitMatrix(k, tn.Rx), splitMatrix(k, tn.Ry), splitMatrix(k, tn.Rz)
	Sx, Sy, Sz := splitMatrix(k, tn.Sx), splitMatrix(k, tn.Sy), splitMatrix(k, tn.Sz)
	Tx, Ty, Tz := splitMatrix(k, tn.Tx), splitMatrix(k, tn.Ty), splitMatrix(k, tn.Tz)
	// Add array parameters
	params = append(params,
		builder.Input("U").Bind(U).CopyTo(),
		builder.Input("Rx").Bind(Rx).CopyTo(),
		builder.Input("Sx").Bind(Sx).CopyTo(),
		builder.Input("Tx").Bind(Tx).CopyTo(),
		builder.Input("Ry").Bind(Ry).CopyTo(),
		builder.Input("Sy").Bind(Sy).CopyTo(),
		builder.Input("Ty").Bind(Ty).CopyTo(),
		builder.Input("Rz").Bind(Rz).CopyTo(),
		builder.Input("Sz").Bind(Sz).CopyTo(),
		builder.Input("Tz").Bind(Tz).CopyTo(),
		builder.Output("DuDx").Bind(DuDx).CopyBack(),
		builder.Output("DuDy").Bind(DuDy).CopyBack(),
		builder.Output("DuDz").Bind(DuDz).CopyBack(),
		builder.Temp("Ur").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Us").Type(builder.Float64).Size(totalNodes),
		builder.Temp("Ut").Type(builder.Float64).Size(totalNodes),
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
#define NP %d

@kernel void %s(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        real_t* Us = Us_PART(part);
        real_t* Ut = Ut_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
        MATMUL_Ds_%s(U, Us, K[part]);
        MATMUL_Dt_%s(U, Ut, K[part]);

        real_t* DuDx = DuDx_PART(part);
        real_t* DuDy = DuDy_PART(part);
        real_t* DuDz = DuDz_PART(part);
        const real_t* Rx = Rx_PART(part);
        const real_t* Sx = Sx_PART(part);
        const real_t* Tx = Tx_PART(part);
        const real_t* Ry = Ry_PART(part);
        const real_t* Sy = Sy_PART(part);
        const real_t* Ty = Ty_PART(part);
        const real_t* Rz = Rz_PART(part);
        const real_t* Sz = Sz_PART(part);
        const real_t* Tz = Tz_PART(part);
        // ************************************************************
		// *** This computation is happening in column-major format ***
        // ************************************************************
		// Multiple partitions means we need to check bounds
        for (int n = 0; n < NP; ++n; @inner) {
            for (int k = 0; k < KpartMax; ++k) {
                int i = n + k*NP;  // Column-major indexing
                if (k < K[part]) {  // Bounds check for partition size
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

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	// Execute differentiation
	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	DuDxH, DuDyH, DuDzH := calcPhysicalDerivative(Uw, tn.Rx, tn.Ry, tn.Rz, tn.Sx,
		tn.Sy, tn.Sz, tn.Tx, tn.Ty, tn.Tz, tn.Dr, tn.Ds, tn.Dt)

	assert.InDeltaSlicef(t, DuDxExpected.RawMatrix().Data, DuDxH, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDyExpected.RawMatrix().Data, DuDyH, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDzExpected.RawMatrix().Data, DuDzH, 1.e-8, "")
	fmt.Println("Host calculation of physical derivative validates")

	DuDxXp := splitMatrix(k, DuDxExpected)
	DuDyXp := splitMatrix(k, DuDyExpected)
	DuDzXp := splitMatrix(k, DuDzExpected)
	for i := range DuDxXp {
		assert.InDeltaSlicef(t, mat.DenseCopyOf(DuDzXp[i]).RawMatrix().Data,
			mat.DenseCopyOf(DuDz[i]).RawMatrix().Data, 1.e-8, "")
		assert.InDeltaSlicef(t, mat.DenseCopyOf(DuDxXp[i]).RawMatrix().Data,
			mat.DenseCopyOf(DuDx[i]).RawMatrix().Data, 1.e-8, "")
		assert.InDeltaSlicef(t, mat.DenseCopyOf(DuDyXp[i]).RawMatrix().Data,
			mat.DenseCopyOf(DuDy[i]).RawMatrix().Data, 1.e-8, "")
	}
	fmt.Println("Device calculation of physical derivative validates")
}

func calcPhysicalDerivative(UM, RxM, RyM, RzM, SxM, SyM, SzM, TxM, TyM,
	TzM, DrM, DsM, DtM *mat.Dense) (DuDx, DuDy, DuDz []float64) {
	var (
		Np, K      = UM.Dims()
		Ur, Us, Ut mat.Dense
	)
	// These matrix multiplies result in row-major format storage into Ur,Us,Ut
	Ur.Mul(DrM, UM)
	Us.Mul(DsM, UM)
	Ut.Mul(DtM, UM)
	DuDx = make([]float64, Np*K)
	DuDy = make([]float64, Np*K)
	DuDz = make([]float64, Np*K)
	// ************************************************************
	// **** This computation is happening in row-major format *****
	// ************************************************************
	for i, ur := range Ur.RawMatrix().Data {
		us, ut := Us.RawMatrix().Data[i], Ut.RawMatrix().Data[i]
		rx, ry, rz := RxM.RawMatrix().Data[i], RyM.RawMatrix().Data[i], RzM.RawMatrix().Data[i]
		sx, sy, sz := SxM.RawMatrix().Data[i], SyM.RawMatrix().Data[i], SzM.RawMatrix().Data[i]
		tx, ty, tz := TxM.RawMatrix().Data[i], TyM.RawMatrix().Data[i], TzM.RawMatrix().Data[i]
		DuDx[i] = rx*ur + sx*us + tx*ut
		DuDy[i] = ry*ur + sy*us + ty*ut
		DuDz[i] = rz*ur + sz*us + tz*ut
	}
	// ************************************************************
	// ************* DuDx... are in row-major format **************
	// ************************************************************
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
