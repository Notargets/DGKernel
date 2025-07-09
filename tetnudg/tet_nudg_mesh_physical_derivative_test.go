package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"math"
	"os"
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

func TestTetNudgMatCopyMatrixReturn(t *testing.T) {
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
		for (int i = 0; i < NP*KpartMax; ++i; @inner) {
			// DxH.Set(j, K, tn.Rx.At(j, K)*Ur.At(j, K))
			Dx[i] = Rx[i]*Ur[i];
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
	// Execute differentiation
	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Dx is now in row-major format and can be compared directly to the host
	// matrix result
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

func splitSlice(splits []int, slice []float64) (splitSlice [][]float64) {
	// TODO: This only workds for column-major strided data - I need to
	//  implement this for matrices anyway
	var (
		total int
	)
	for _, k := range splits {
		total += k
	}
	if total%len(slice) != 0 {
		panic("split slice is not a multiple of len(splits)")
	}
	stride := len(slice) / total
	splitSlice = make([][]float64, len(splits))
	var iii int
	for i, K := range splits {
		splitSlice[i] = make([]float64, K*stride)
		for ii := 0; ii < K*stride; ii++ {
			splitSlice[i][ii] = slice[iii]
			iii++
		}
	}
	return
}

func _TestTetNudgPhysicalDerivativePartitionedMesh(t *testing.T) {
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
		UU := make([]float64, elementSum)
		for i := range UU {
			UU[i] = float64(i)
		}
		UUU := splitSlice(k, UU)
		for i := range k {
			fmt.Printf("Partition %d: %v\n", i, UUU[i])
		}
		os.Exit(1)
	} else {
		k = []int{Ktot}
	}
	fmt.Printf("Partition K: %v\n", k)
	props := tn.GetProperties()

	device := utils.CreateTestDevice()
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
