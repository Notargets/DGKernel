package gonudg

import (
	"gonum.org/v1/gonum/mat"
)

// Vandermonde2D initializes the 2D Vandermonde Matrix using mat.Dense
// This is a version that returns *mat.Dense instead of utils.Matrix
func Vandermonde2D(N int, R, S []float64) *mat.Dense {
	Np := (N + 1) * (N + 2) / 2
	Nr := len(R)

	V2D := mat.NewDense(Nr, Np, nil)

	sk := 0
	for i := 0; i <= N; i++ {
		for j := 0; j <= (N - i); j++ {
			// Get the polynomial values for this mode
			P := Simplex2DP(R, S, i, j)

			// Set the column
			for row := 0; row < Nr; row++ {
				V2D.Set(row, sk, P[row])
			}
			sk++
		}
	}
	return V2D
}

// Simplex2DP evaluates 2D orthonormal polynomial on simplex at (R,
// S) of order (i,j)
// This version works with []float64 instead of utils.Vector
func Simplex2DP(R, S []float64, i, j int) []float64 {
	// DeviceMemType R,S to (a,b) coordinates
	a, b := RStoAB(R, S)

	Np := len(R)
	h1 := JacobiP(a, 0, 0, i)
	h2 := JacobiP(b, float64(2*i+1), 0, j)

	P := make([]float64, Np)
	sq2 := 1.4142135623730951 // sqrt(2)

	for ii := range h1 {
		tv1 := sq2 * h1[ii] * h2[ii]
		tv2 := 1.0
		if i > 0 {
			tv2 = pow(1-b[ii], i)
		}
		P[ii] = tv1 * tv2
	}
	return P
}

// RStoAB converts from (r,s) to (a,b) coordinates
// This version works with []float64 instead of utils.Vector
func RStoAB(R, S []float64) (a, b []float64) {
	Np := len(R)
	a = make([]float64, Np)
	b = make([]float64, Np)

	for n := 0; n < Np; n++ {
		if S[n] != 1 {
			a[n] = 2*(1+R[n])/(1-S[n]) - 1
		} else {
			a[n] = -1
		}
		b[n] = S[n]
	}
	return
}

// pow computes x^n for integer n
func pow(x float64, n int) float64 {
	if n == 0 {
		return 1.0
	}
	result := x
	for i := 1; i < n; i++ {
		result *= x
	}
	return result
}
