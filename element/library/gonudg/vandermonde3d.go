package gonudg

// INDEXING NOTE: Original C++ code uses 1-based indexing to emulate Matlab behavior.
// This Go port uses standard 0-based indexing. Example conversions:
//   C++: sk = 1; V3D(All,sk) = ...    ->    Go: sk = 0; V3D.SetCol(sk, ...)
//   C++: Fmask[1] (first face)        ->    Go: Fmask[0] (first face)
// The indexing has been correctly translated throughout this port.

import (
	"gonum.org/v1/gonum/mat"
)

// Vandermonde3D initializes the 3D Vandermonde Matrix V_{ij} = phi_j(r_i, s_i, t_i)
// This is the 0-based index version of the C++ Vandermonde3D function
func Vandermonde3D(N int, r, s, t []float64) *mat.Dense {
	Np := len(r)
	Ncol := (N + 1) * (N + 2) * (N + 3) / 6

	// Initialize the Vandermonde matrix
	V3D := mat.NewDense(Np, Ncol, nil)

	// Transfer to (a,b,c) coordinates
	a, b, c := RSTtoABC(r, s, t)

	// Build the Vandermonde matrix
	sk := 0 // 0-based column index
	for i := 0; i <= N; i++ {
		for j := 0; j <= N-i; j++ {
			for k := 0; k <= N-i-j; k++ {
				// Evaluate basis function at all points
				col := Simplex3DP(a, b, c, i, j, k)

				// Copy to matrix column
				V3D.SetCol(sk, col)
				sk++
			}
		}
	}

	return V3D
}

// GradVandermonde3D builds the gradient Vandermonde matrices
// Returns Vr, Vs, Vt where (Vr)_{ij} = dphi_j/dr at point i
func GradVandermonde3D(N int, r, s, t []float64) (Vr, Vs, Vt *mat.Dense) {
	Np := len(r)
	Ncol := (N + 1) * (N + 2) * (N + 3) / 6

	// Initialize the gradient matrices
	Vr = mat.NewDense(Np, Ncol, nil)
	Vs = mat.NewDense(Np, Ncol, nil)
	Vt = mat.NewDense(Np, Ncol, nil)

	// Build the gradient Vandermonde matrices
	sk := 0 // 0-based column index
	for i := 0; i <= N; i++ {
		for j := 0; j <= N-i; j++ {
			for k := 0; k <= N-i-j; k++ {
				// Evaluate gradient of basis function at all points
				dr, ds, dt := GradSimplex3DP(r, s, t, i, j, k)

				// Copy to matrix columns
				Vr.SetCol(sk, dr)
				Vs.SetCol(sk, ds)
				Vt.SetCol(sk, dt)
				sk++
			}
		}
	}

	return Vr, Vs, Vt
}

// Dmatrices3D computes the differentiation matrices Dr, Ds, Dt
// Given the Vandermonde matrix V and points (R,S,T)
func Dmatrices3D(N int, r, s, t []float64, V *mat.Dense) (pDr, pDs, pDt *mat.Dense) {
	// Get gradient Vandermonde matrices
	Vr, Vs, Vt := GradVandermonde3D(N, r, s, t)

	// Compute V inverse
	// Vinv := V.InverseWithCheck()
	var Vinv mat.Dense
	err := Vinv.Inverse(V)
	if err != nil {
		panic(err)
	}

	// Dr = Vr * V^{-1}, etc.
	// Dr = Vr.Mul(Vinv)
	// Ds = Vs.Mul(Vinv)
	// Dt = Vt.Mul(Vinv)

	var Dr, Ds, Dt mat.Dense
	Dr.Mul(Vr, &Vinv)
	Ds.Mul(Vs, &Vinv)
	Dt.Mul(Vt, &Vinv)

	return &Dr, &Ds, &Dt
}
