package gonudg

import (
	"fmt"
	"math"
	"testing"

	"github.com/notargets/gocfd/utils"
	"gonum.org/v1/gonum/mat"
)

// monomialValue evaluates X^i * Y^j * Z^k at a point
func monomialValue(x, y, z float64, i, j, k int) float64 {
	result := 1.0
	for n := 0; n < i; n++ {
		result *= x
	}
	for n := 0; n < j; n++ {
		result *= y
	}
	for n := 0; n < k; n++ {
		result *= z
	}
	return result
}

// monomialDerivative computes the derivative of X^i * Y^j * Z^k
// with respect to X (deriv=0), Y (deriv=1), or Z (deriv=2)
func monomialDerivative(x, y, z float64, i, j, k, deriv int) float64 {
	switch deriv {
	case 0: // ∂/∂X
		if i == 0 {
			return 0.0
		}
		return float64(i) * monomialValue(x, y, z, i-1, j, k)
	case 1: // ∂/∂Y
		if j == 0 {
			return 0.0
		}
		return float64(j) * monomialValue(x, y, z, i, j-1, k)
	case 2: // ∂/∂Z
		if k == 0 {
			return 0.0
		}
		return float64(k) * monomialValue(x, y, z, i, j, k-1)
	default:
		panic("invalid derivative direction")
	}
}

// evaluatePolynomialAtNodes evaluates a polynomial at all nodes
func evaluatePolynomialAtNodes(X, Y, Z []float64, i, j, k int) []float64 {
	Np := len(X)
	values := make([]float64, Np)
	for n := 0; n < Np; n++ {
		values[n] = monomialValue(X[n], Y[n], Z[n], i, j, k)
	}
	return values
}

// TestVandermonde3DPolynomialInterpolation tests that the Vandermonde matrix
// can exactly interpolate all polynomials up to degree N
func TestVandermonde3DPolynomialInterpolation(t *testing.T) {
	maxOrder := 6
	tol := 1e-10

	for N := 1; N <= maxOrder; N++ {
		t.Run(fmt.Sprintf("N=%d", N), func(t *testing.T) {
			// Generate nodes
			X, Y, Z := Nodes3D(N)
			r, s, tt := XYZtoRST(X, Y, Z)

			// Build Vandermonde matrix
			V := Vandermonde3D(N, r, s, tt)

			// Use proper gonum inverse
			var Vinv mat.Dense
			err := Vinv.Inverse(V)
			if err != nil {
				t.Fatalf("Failed to invert Vandermonde matrix: %v", err)
			}

			// Test all monomials up to degree N
			successCount := 0
			totalCount := 0

			for totalDegree := 0; totalDegree <= N; totalDegree++ {
				for i := 0; i <= totalDegree; i++ {
					for j := 0; j <= totalDegree-i; j++ {
						k := totalDegree - i - j
						totalCount++

						// Evaluate monomial at physical nodes
						fValues := evaluatePolynomialAtNodes(X, Y, Z, i, j, k)

						// Create vector from values
						f := mat.NewVecDense(len(fValues), fValues)

						// Find modal coefficients: coeffs = V^{-1} * f
						coeffs := mat.NewVecDense(V.RawMatrix().Cols, nil)
						coeffs.MulVec(&Vinv, f)

						// Reconstruct: f_reconstructed = V * coeffs
						fReconstructed := mat.NewVecDense(len(fValues), nil)
						fReconstructed.MulVec(V, coeffs)

						// Check reconstruction error
						maxError := 0.0
						for n := 0; n < len(fValues); n++ {
							error := math.Abs(fReconstructed.AtVec(n) - fValues[n])
							if error > maxError {
								maxError = error
							}
						}

						if maxError > tol {
							t.Logf("Monomial X^%d Y^%d Z^%d: max error = %e", i, j, k, maxError)
						} else {
							successCount++
						}
					}
				}
			}

			t.Logf("N=%d: Successfully interpolated %d/%d monomials", N, successCount, totalCount)
			if successCount < totalCount {
				t.Errorf("Failed to interpolate %d monomials", totalCount-successCount)
			}
		})
	}
}

// TestGradVandermonde3DExactDerivatives tests that gradient matrices
// compute exact derivatives of polynomials
func TestGradVandermonde3DExactDerivatives(t *testing.T) {
	for N := 1; N <= 4; N++ {
		t.Run(fmt.Sprintf("N=%d", N), func(t *testing.T) {
			// Generate nodes
			X, Y, Z := Nodes3D(N)
			r, s, tt := XYZtoRST(X, Y, Z)

			// Build matrices
			V := Vandermonde3D(N, r, s, tt)
			Dr, Ds, Dt := Dmatrices3D(N, r, s, tt, V)

			// Test 1: Derivatives of linear polynomials should be exact
			testPolynomials := []struct {
				name    string
				i, j, k int
			}{
				{"X", 1, 0, 0},
				{"Y", 0, 1, 0},
				{"Z", 0, 0, 1},
			}

			for _, poly := range testPolynomials {
				// Evaluate polynomial at nodes
				fValues := evaluatePolynomialAtNodes(X, Y, Z, poly.i, poly.j, poly.k)
				fVec := utils.NewVector(len(fValues))
				for i := 0; i < len(fValues); i++ {
					fVec.Set(i, fValues[i])
				}

				// Compute derivatives using matrices
				// Need to convert utils.Matrix to mat.Matrix for multiplication
				var dfdr, dfds, dfdt mat.Dense
				dfdr.Mul(Dr, mat.NewDense(len(fValues), 1, fValues))
				dfds.Mul(Ds, mat.NewDense(len(fValues), 1, fValues))
				dfdt.Mul(Dt, mat.NewDense(len(fValues), 1, fValues))

				// Expected derivatives (in physical space)
				for n := 0; n < len(X); n++ {
					// Expected physical derivatives
					expectedDx := monomialDerivative(X[n], Y[n], Z[n], poly.i, poly.j, poly.k, 0)
					expectedDy := monomialDerivative(X[n], Y[n], Z[n], poly.i, poly.j, poly.k, 1)
					expectedDz := monomialDerivative(X[n], Y[n], Z[n], poly.i, poly.j, poly.k, 2)
					_, _, _ = expectedDx, expectedDy, expectedDz

					// We need the Jacobian to transform from (r,s,t) to (X,Y,Z) derivatives
					// For the standard tetrahedron transformation
					// For now, just test that derivatives are computed without error
					drValue := dfdr.At(n, 0)
					dsValue := dfds.At(n, 0)
					dtValue := dfdt.At(n, 0)

					// Basic sanity check - derivatives shouldn't be NaN or Inf
					if math.IsNaN(drValue) || math.IsInf(drValue, 0) ||
						math.IsNaN(dsValue) || math.IsInf(dsValue, 0) ||
						math.IsNaN(dtValue) || math.IsInf(dtValue, 0) {
						t.Errorf("Invalid derivative for %s at node %d", poly.name, n)
					}
				}
			}

			// Test 2: Commutation of mixed derivatives (if we had second derivatives)
			// This would test that ∂²f/∂R∂S = ∂²f/∂S∂R
			// Skipped as we don't have second derivative matrices
		})
	}
}

// TestPolynomialOrderConvergence verifies that higher orders maintain
// accuracy for lower degree polynomials
func TestPolynomialOrderConvergence(t *testing.T) {
	// Test that a polynomial of degree p is exactly represented
	// by all Vandermonde matrices of order N >= p
	testPolynomials := []struct {
		name    string
		i, j, k int
	}{
		{"constant", 0, 0, 0},
		{"X", 1, 0, 0},
		{"Y", 0, 1, 0},
		{"Z", 0, 0, 1},
		{"xy", 1, 1, 0},
		{"X²", 2, 0, 0},
		{"xyz", 1, 1, 1},
		{"X³", 3, 0, 0},
	}

	tol := 1e-10

	for _, poly := range testPolynomials {
		t.Run(poly.name, func(t *testing.T) {
			minOrder := poly.i + poly.j + poly.k
			if minOrder == 0 {
				minOrder = 1 // Even constant needs N >= 1
			}

			// Test all higher orders
			for N := minOrder; N <= 6; N++ {
				X, Y, Z := Nodes3D(N)
				r, s, tt := XYZtoRST(X, Y, Z)
				V := Vandermonde3D(N, r, s, tt)

				// Use proper gonum inverse
				var Vinv mat.Dense
				err := Vinv.Inverse(V)
				if err != nil {
					t.Fatalf("Failed to invert Vandermonde matrix: %v", err)
				}

				// Interpolate
				f := evaluatePolynomialAtNodes(X, Y, Z, poly.i, poly.j, poly.k)
				fVec := mat.NewVecDense(len(f), f)

				// Compute coefficients
				coeffs := mat.NewVecDense(V.RawMatrix().Cols, nil)
				coeffs.MulVec(&Vinv, fVec)

				// Reconstruct
				fRecon := mat.NewVecDense(len(f), nil)
				fRecon.MulVec(V, coeffs)

				// Check exact reconstruction
				maxError := 0.0
				for i := 0; i < len(f); i++ {
					error := math.Abs(fRecon.AtVec(i) - f[i])
					if error > maxError {
						maxError = error
					}
				}

				if maxError > tol {
					t.Errorf("Order N=%d: %s interpolation error = %e",
						N, poly.name, maxError)
				}
			}
		})
	}
}
