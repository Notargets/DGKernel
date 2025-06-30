package gonudg

// INDEXING NOTE: Original C++ code uses 1-based indexing to emulate Matlab behavior.
// This Go port uses standard 0-based indexing. Example conversions:
//   C++: sk = 1; V3D(All,sk) = ...    ->    Go: sk = 0; V3D.SetCol(sk, ...)
//   C++: Fmask[1] (first face)        ->    Go: Fmask[0] (first face)
// The indexing has been correctly translated throughout this port.

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// JacobiGL computes the Gauss-Lobatto quadrature points for Jacobi polynomials
// These are the zeros of (1-X^2)*P'_N^{alpha,beta}(X)
func JacobiGL(alpha, beta float64, N int) []float64 {
	if N == 0 {
		return []float64{0.0}
	}

	if N == 1 {
		return []float64{-1.0, 1.0}
	}

	// Get the interior Gauss-Jacobi points
	// FIXED: Use N-2 to match C++ implementation
	// This gives us N-1 interior points, plus 2 endpoints = N+1 total points
	xint, _ := JacobiGQ(alpha+1, beta+1, N-2)

	// Combine with endpoints
	x := make([]float64, N+1)
	x[0] = -1.0
	copy(x[1:N], xint) // Copy N-1 interior points
	x[N] = 1.0

	return x
}

func JacobiGQ(alpha, beta float64, N int) (X, W []float64) {
	var (
		x, w       []float64
		fac        float64
		h1, d0, d1 []float64
		VVr        *mat.Dense
	)
	if N == 0 {
		x = []float64{-(alpha - beta) / (alpha + beta + 2.)}
		w = []float64{2.}
		return x, w
	}

	h1 = make([]float64, N+1)
	for i := 0; i < N+1; i++ {
		h1[i] = 2*float64(i) + alpha + beta
	}

	// main diagonal: d0[i] = -(β²-α²)/((2i+α+β)*(2i+α+β+2))
	d0 = make([]float64, N+1)
	fac = (beta*beta - alpha*alpha)
	for i := 0; i < N+1; i++ {
		val := h1[i]
		d0[i] = fac / (val * (val + 2.))
	}

	// Handle division by zero
	eps := 1.e-16
	if alpha+beta < 10*eps {
		d0[0] = 0.
	}

	// 1st upper diagonal: diag(2./(h1(1:N)+2).*sqrt((1:N).*((1:N)+alpha+beta) .* ((1:N)+alpha).*((1:N)+beta)./(h1(1:N)+1)./(h1(1:N)+3)),1);
	// for (i=1; i<=N; ++i) { d1(i)=2.0/(h1(i)+2.0)*sqrt(i*(i+alpha+beta)*(i+alpha)*(i+beta)/(h1(i)+1)/(h1(i)+3.0)); }
	// var ip1 float64
	d1 = make([]float64, N)
	for i := 0; i < N; i++ {
		ip1 := float64(i + 1)
		val := h1[i]
		d1[i] = 2.0 / (val + 2.0) * math.Sqrt(
			ip1*(ip1+alpha+beta)*(ip1+alpha)*(ip1+beta)/(val+1)/(val+3),
		)
	}

	// JJ := utils.NewSymTriDiagonal(d0, d1)
	JJ := NewSymTriDiagonal(d0, d1)

	var eig mat.EigenSym
	ok := eig.Factorize(JJ, true)
	if !ok {
		panic("eigenvalue decomposition failed")
	}
	x = eig.Values(x)

	VVr = mat.NewDense(len(x), len(x), nil)
	eig.VectorsTo(VVr)
	W = VVr.RawRowView(0)
	// .POW(2).Scale(Gamma0(alpha, beta))
	for i := range W {
		W[i] *= W[i] * Gamma0(alpha, beta)
	}
	return x, W
}

func Gamma0(alpha, beta float64) float64 {
	ab1 := alpha + beta + 1.
	a1 := alpha + 1.
	b1 := beta + 1.
	return math.Gamma(a1) * math.Gamma(b1) * math.Pow(2, ab1) / ab1 / math.Gamma(ab1)
}

func Gamma1(alpha, beta float64) float64 {
	ab := alpha + beta
	a1 := alpha + 1.
	b1 := beta + 1.
	return a1 * b1 * Gamma0(alpha, beta) / (ab + 3.0)
}

func NewSymTriDiagonal(d0, d1 []float64) (Tri *mat.SymDense) {
	dd := make([]float64, len(d0)*len(d0))
	var p1, p2 int
	for j := 0; j < len(d0); j++ {
		for i := 0; i < len(d0); i++ {
			if i == j {
				dd[i+j*len(d0)] = d0[p1]
				p1++
				if i != len(d0)-1 {
					dd[+1+i+j*len(d0)] = d1[p2]
					p2++
				}
			}
		}
	}
	Tri = mat.NewSymDense(len(d0), dd)
	return
}
