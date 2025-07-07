package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"math"
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
			DxH.Set(j, K, Ur.At(j, K)*tn.Rx.At(j, K))
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

	DuDx := make([]float64, totalNodes)
	DuDy := make([]float64, totalNodes)
	DuDz := make([]float64, totalNodes)
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
		for (int i = 0; i < NP*KpartMax; ++i; @inner) {
			DuDx[i] = Rx[i]*Ur[i] + Sx[i]*Us[i] + Tx[i]*Ut[i];
			DuDy[i] = Ry[i]*Ur[i] + Sy[i]*Us[i] + Ty[i]*Ut[i];
			DuDz[i] = Rz[i]*Ur[i] + Sz[i]*Us[i] + Tz[i]*Ut[i];
		}
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

	// Because the data from the kernel is in column-major format, we need to
	// transpose the matrix to compare it to the host calculation
	DuDxT := mat.DenseCopyOf(mat.NewDense(Ktot, Np, DuDx).T()).RawMatrix().Data
	DuDyT := mat.DenseCopyOf(mat.NewDense(Ktot, Np, DuDy).T()).RawMatrix().Data
	DuDzT := mat.DenseCopyOf(mat.NewDense(Ktot, Np, DuDz).T()).RawMatrix().Data
	assert.InDeltaSlicef(t, DuDxExpected.RawMatrix().Data, DuDxT, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDyExpected.RawMatrix().Data, DuDyT, 1.e-8, "")
	assert.InDeltaSlicef(t, DuDzExpected.RawMatrix().Data, DuDzT, 1.e-8, "")
	fmt.Println("Device calculation of physical derivative validates")
}

func calcPhysicalDerivative(UM, RxM, RyM, RzM, SxM, SyM, SzM, TxM, TyM,
	TzM, DrM, DsM, DtM *mat.Dense) (DuDx, DuDy, DuDz []float64) {
	var (
		Np, K      = UM.Dims()
		Ur, Us, Ut mat.Dense
	)
	Ur.Mul(DrM, UM)
	Us.Mul(DsM, UM)
	Ut.Mul(DtM, UM)
	DuDx = make([]float64, Np*K)
	DuDy = make([]float64, Np*K)
	DuDz = make([]float64, Np*K)
	for i, ur := range Ur.RawMatrix().Data {
		us, ut := Us.RawMatrix().Data[i], Ut.RawMatrix().Data[i]
		rx, ry, rz := RxM.RawMatrix().Data[i], RyM.RawMatrix().Data[i], RzM.RawMatrix().Data[i]
		sx, sy, sz := SxM.RawMatrix().Data[i], SyM.RawMatrix().Data[i], SzM.RawMatrix().Data[i]
		tx, ty, tz := TxM.RawMatrix().Data[i], TyM.RawMatrix().Data[i], TzM.RawMatrix().Data[i]
		DuDx[i] = rx*ur + sx*us + tx*ut
		DuDy[i] = ry*ur + sy*us + ty*ut
		DuDz[i] = rz*ur + sz*us + tz*ut
	}
	return
}
