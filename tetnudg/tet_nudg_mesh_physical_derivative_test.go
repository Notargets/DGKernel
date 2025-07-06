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

func TestTetNudgPhysicalDerivative(t *testing.T) {
	// TODO: The issue this test has surfaced is that golang is ROW-MAJOR
	//  storage of matrices, where all numerical libraries are COLUMN-MAJOR
	//  We need to implement COLUMN-MAJOR in the DGKernel code because the
	//  partitions need to be stored contiguously for us to slice them.
	//  We will need to automatically transpose matrices when copying from
	//  the host to device to implement the required COLUMN-MAJOR format.
	//  We'll keep arrays unchanged,
	//  and make notes so that a user will need to implement COLUMN-MAJOR
	//  when flattening matrices into slices on the host before tranferring
	//  to the device. This will perform optimally on host,
	//  and allow for natural zero copy manipulation of partitions for
	//  parallelism.
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
		params = append(params, runner.Input(name).Bind(mat).ToMatrix())
	}

	U := make([]float64, totalNodes)
	UxExpected := make([]float64, totalNodes)
	UyExpected := make([]float64, totalNodes)
	UzExpected := make([]float64, totalNodes)
	fOrder := float64(order)
	for i := 0; i < totalNodes; i++ {
		x, y, z := tn.X.RawMatrix().Data[i], tn.Y.RawMatrix().Data[i], tn.Z.RawMatrix().Data[i]
		xP, yP, zP := math.Pow(x, fOrder), math.Pow(y, fOrder), math.Pow(z, fOrder)
		dxP := float64(fOrder) * math.Pow(x, fOrder-1)
		dyP := float64(fOrder) * math.Pow(y, fOrder-1)
		dzP := float64(fOrder) * math.Pow(z, fOrder-1)
		U[i] = xP + yP + zP
		UxExpected[i] = dxP
		UyExpected[i] = dyP
		UzExpected[i] = dzP
	}

	Dx := make([]float64, totalNodes)
	Dy := make([]float64, totalNodes)
	Dz := make([]float64, totalNodes)
	// Add array parameters
	params = append(params,
		runner.Input("U").Bind(U).CopyTo(),
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
		runner.Temp("Ur").Type(builder.Float64).Size(totalNodes),
		runner.Temp("Us").Type(builder.Float64).Size(totalNodes),
		runner.Temp("Ut").Type(builder.Float64).Size(totalNodes),
	)

	// Define kernel with all parameters
	err := kp.DefineKernel("differentiate", params...)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get signature and build kernel
	signature, _ := kp.GetKernelSignature("differentiate")
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void differentiate(%s) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* Ur = Ur_PART(part);
        real_t* Us = Us_PART(part);
        real_t* Ut = Ut_PART(part);
        MATMUL_Dr_%s(U, Ur, K[part]);
        MATMUL_Ds_%s(U, Us, K[part]);
        MATMUL_Dt_%s(U, Ut, K[part]);

        real_t* Dx = Dx_PART(part);
        real_t* Dy = Dy_PART(part);
        real_t* Dz = Dz_PART(part);
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
			Dx[i] = Rx[i]*Ur[i] + Sx[i]*Us[i] + Tx[i]*Ut[i];
			Dy[i] = Ry[i]*Ur[i] + Sy[i]*Us[i] + Ty[i]*Ut[i];
			Dz[i] = Rz[i]*Ur[i] + Sz[i]*Us[i] + Tz[i]*Ut[i];
		}
    }
}
`, tn.Np, signature,
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

	UM := mat.NewDense(tn.Np, tn.K, U)
	DxH, DyH, DzH := calcPhysicalDerivative(UM, tn.Rx, tn.Ry, tn.Rz, tn.Sx,
		tn.Sy, tn.Sz, tn.Tx, tn.Ty, tn.Tz, tn.Dr, tn.Ds, tn.Dt)

	_, _, _ = DxH, DyH, DzH
	assert.InDeltaSlicef(t, UxExpected, DxH, 1.e-8, "")
	assert.InDeltaSlicef(t, UyExpected, DyH, 1.e-8, "")
	assert.InDeltaSlicef(t, UzExpected, DzH, 1.e-8, "")
	fmt.Println("Host calculation of physical derivative validates")

	assert.InDeltaSlicef(t, UxExpected, Dx, 1.e-8, "")
}

func calcPhysicalDerivative(UM, RxM, RyM, RzM, SxM, SyM, SzM, TxM, TyM,
	TzM, DrM, DsM, DtM *mat.Dense) (Dx, Dy, Dz []float64) {
	var (
		Np, K      = UM.Dims()
		Ur, Us, Ut mat.Dense
	)
	Ur.Mul(DrM, UM)
	Us.Mul(DsM, UM)
	Ut.Mul(DtM, UM)
	Dx = make([]float64, Np*K)
	Dy = make([]float64, Np*K)
	Dz = make([]float64, Np*K)
	for i, ur := range Ur.RawMatrix().Data {
		us, ut := Us.RawMatrix().Data[i], Ut.RawMatrix().Data[i]
		rx, ry, rz := RxM.RawMatrix().Data[i], RyM.RawMatrix().Data[i], RzM.RawMatrix().Data[i]
		sx, sy, sz := SxM.RawMatrix().Data[i], SyM.RawMatrix().Data[i], SzM.RawMatrix().Data[i]
		tx, ty, tz := TxM.RawMatrix().Data[i], TyM.RawMatrix().Data[i], TzM.RawMatrix().Data[i]
		Dx[i] = rx*ur + sx*us + tx*ut
		Dy[i] = ry*ur + sy*us + ty*ut
		Dz[i] = rz*ur + sz*us + tz*ut
	}
	return
}
