package gonudg

import (
	"math"
)

// JacobiP evaluates Jacobi polynomial of type (alpha,beta) at points x for order n
// This version works with []float64
func JacobiP(x []float64, alpha, beta float64, n int) []float64 {
	Np := len(x)
	P := make([]float64, Np)

	// Initial values P_0(x) and P_1(x)
	gamma0 := math.Pow(2, alpha+beta+1) / (alpha + beta + 1) *
		math.Gamma(alpha+1) * math.Gamma(beta+1) / math.Gamma(alpha+beta+1)

	for i := range P {
		P[i] = 1.0 / math.Sqrt(gamma0)
	}

	if n == 0 {
		return P
	}

	gamma1 := (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
	for i := range P {
		P[i] = ((alpha+beta+2)*x[i] + (alpha - beta)) / 2 / math.Sqrt(gamma1)
	}

	if n == 1 {
		return P
	}

	// Use recurrence relation for higher orders
	aold := 2.0 / (2.0 + alpha + beta) * math.Sqrt((alpha+1)*(beta+1)/(alpha+beta+3))

	for i := 1; i < n; i++ {
		h1 := 2*float64(i) + alpha + beta
		anew := 2.0 / (h1 + 2) * math.Sqrt((float64(i)+1)*(float64(i)+1+alpha+beta)*
			(float64(i)+1+alpha)*(float64(i)+1+beta)/(h1+1)/(h1+3))
		bnew := -(alpha*alpha - beta*beta) / h1 / (h1 + 2)

		// Update P values using recurrence
		Pold := make([]float64, Np)
		copy(Pold, P)

		for j := range P {
			P[j] = 1 / anew * (-aold*P[j] + (x[j]-bnew)*P[j])
		}

		aold = anew
	}

	return P
}

// JacobiPSingle evaluates Jacobi polynomial at a single point
func JacobiPSingle(x, alpha, beta float64, n int) float64 {
	xArr := []float64{x}
	result := JacobiP(xArr, alpha, beta, n)
	return result[0]
}

// GradJacobiP evaluates the derivative of the Jacobi polynomial of type (alpha,beta) at points x for order n
func GradJacobiP(x []float64, alpha, beta float64, n int) []float64 {
	Np := len(x)
	dP := make([]float64, Np)

	if n == 0 {
		// Derivative of constant is zero
		return dP
	}

	// Use the derivative formula: d/dx P_n^(a,b)(x) = sqrt(n(n+a+b+1)) * P_{n-1}^(a+1,b+1)(x)
	Ptemp := JacobiP(x, alpha+1, beta+1, n-1)
	for i := range dP {
		dP[i] = math.Sqrt(float64(n)*(float64(n)+alpha+beta+1)) * Ptemp[i]
	}

	return dP
}

// Gamma computes the gamma function
// For simplicity, we'll use a basic implementation
func Gamma(x float64) float64 {
	// This is a simplified version. In production, use math.Gamma when available
	// or a more sophisticated implementation
	if x == 1.0 {
		return 1.0
	}
	if x == 0.5 {
		return math.Sqrt(math.Pi)
	}
	// For integer values
	if x == math.Floor(x) && x > 0 {
		n := int(x)
		result := 1.0
		for i := 1; i < n; i++ {
			result *= float64(i)
		}
		return result
	}
	// For other values, use Stirling's approximation or other methods
	// This is simplified - in production use a proper gamma function
	return math.Sqrt(2*math.Pi/x) * math.Pow(x/math.E, x)
}
