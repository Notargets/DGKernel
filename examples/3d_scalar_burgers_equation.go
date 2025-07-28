package main

import (
	"fmt"
	"github.com/notargets/DGKernel/element"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/notargets/DGKernel/runner"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/tetnudg"
	"github.com/notargets/DGKernel/utils"
	"github.com/notargets/gocca"
)

// Simulation parameters
const (
	PolynomialOrder = 3
	FinalTime       = 0.1  // Before shock formation
	OutputInterval  = 0.01 // Output frequency
	CFL             = 0.1  // CFL number for stability
)

// Gaussian pulse parameters
const (
	GaussianAmplitude = 1.0 // Peak amplitude A
	PulseCenter       = 0.0 // Center location
)

// Output line parameters for validation
const (
	LineStart = -0.5
	LineEnd   = 0.5
	NumPoints = 100
)

func main() {
	// Create computational device
	device := utils.CreateTestDevice()
	defer device.Free()

	fmt.Printf("=== 3D Scalar Burgers Equation Solver ===\n")
	fmt.Printf("Polynomial Order: %d\n", PolynomialOrder)
	fmt.Printf("Final Time: %.3f\n", FinalTime)

	// Load tetrahedral mesh using DGKernel mesh readers
	meshFile := "mesh/cube-partitioned.neu"
	if _, err := os.Stat(meshFile); os.IsNotExist(err) {
		log.Fatalf("Mesh file not found: %s", meshFile)
	}

	fmt.Printf("Loading mesh: %s\n", meshFile)
	tn := tetnudg.NewTetNudgMesh(PolynomialOrder, meshFile)
	fmt.Printf("Mesh loaded: %d elements, %d nodes per element\n", tn.K, tn.Np)

	// Compute mesh characteristic length for Gaussian width
	hMin := computeMinimumEdgeLength(tn)
	gaussianWidth := 10.0 * hMin // σ >= 10h rule for good resolution
	fmt.Printf("Minimum edge length: %.6f\n", hMin)
	fmt.Printf("Gaussian width: %.6f\n", gaussianWidth)

	// Initialize solver
	solver, err := NewBurgers3DSolver(device, tn, gaussianWidth)
	if err != nil {
		log.Fatalf("Failed to create solver: %v", err)
	}
	defer solver.Free()

	// Run simulation
	fmt.Printf("\nStarting time integration...\n")
	err = solver.Solve()
	if err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}

	// Post-process results
	fmt.Printf("\nPost-processing results...\n")
	err = solver.OutputResults()
	if err != nil {
		log.Fatalf("Failed to output results: %v", err)
	}

	fmt.Printf("Simulation complete!\n")
	fmt.Printf("Results written to: results/burgers_3d_results.csv\n")
	fmt.Printf("To plot: cd results && gnuplot plot_burgers.gnu\n")
}

// Burgers3DSolver encapsulates the DG solver for 3D scalar Burgers equation
type Burgers3DSolver struct {
	device *gocca.OCCADevice
	runner *runner.Runner

	// Mesh and element data
	tetMesh *tetnudg.TetNudgMesh
	K       int // Number of elements
	Np      int // Nodes per element

	// Solution arrays
	u    []float64 // Current solution
	resu []float64 // RK residual
	rhsu []float64 // Right-hand side

	// Gaussian parameters
	sigma float64 // Gaussian width

	// Time stepping
	dt   float64 // Time step
	time float64 // Current time

	// RK5SSP4 coefficients (low-storage)
	rk4a []float64
	rk4b []float64

	// Current RK stage scalars
	rka float64
	rkb float64

	faceBuffer *FaceBuffer // Face connectivity for DG flux computation
}

func NewBurgers3DSolver(device *gocca.OCCADevice, tn *tetnudg.TetNudgMesh, sigma float64) (*Burgers3DSolver, error) {
	solver := &Burgers3DSolver{
		device:  device,
		tetMesh: tn,
		K:       tn.K,
		Np:      tn.Np,
		sigma:   sigma,
		time:    0.0,
	}

	// Initialize solution arrays
	totalNodes := solver.K * solver.Np
	solver.u = make([]float64, totalNodes)
	solver.resu = make([]float64, totalNodes)
	solver.rhsu = make([]float64, totalNodes)

	// Set RK5SSP4 coefficients (from project knowledge)
	solver.rk4a = []float64{
		0.0,
		-567301805773.0 / 1357537059087.0,
		-2404267990393.0 / 2016746695238.0,
		-3550918686646.0 / 2091501179385.0,
		-1275806237668.0 / 842570457699.0,
	}
	solver.rk4b = []float64{
		1432997174477.0 / 9575080441755.0,
		5161836677717.0 / 13612068292357.0,
		1720146321549.0 / 2090206949498.0,
		3134564353537.0 / 4481467310338.0,
		2277821191437.0 / 14882151754819.0,
	}

	// Initialize DGKernel runner
	err := solver.setupDGKernel()
	if err != nil {
		return nil, fmt.Errorf("failed to setup DGKernel: %w", err)
	}

	// Initialize face exchange
	err = solver.setupFaceExchange()
	if err != nil {
		return nil, fmt.Errorf("failed to setup face exchange: %w", err)
	}

	// Initialize solution
	solver.initializeGaussianPulse()

	// Compute stable time step
	solver.computeTimeStep()

	return solver, nil
}

func (s *Burgers3DSolver) setupDGKernel() error {
	// Create runner with single partition (non-partitioned example)
	s.runner = runner.NewRunner(s.device, builder.Config{
		K: []int{s.K},
	})

	// Get properties for matrix naming
	props := s.tetMesh.GetProperties()

	// Collect reference matrices from tetrahedral element
	refMatrices := s.tetMesh.GetRefMatrices()

	// Build parameter list starting with matrices
	params := make([]*builder.ParamBuilder, 0, len(refMatrices)+15)

	// Add all reference element matrices as static parameters
	for name, matrix := range refMatrices {
		params = append(params, builder.Input(name).Bind(matrix).ToMatrix().Static())
	}

	// Phase 1: Define bindings for all arrays and scalars needed
	params = append(params,
		// Solution arrays
		builder.InOut("u").Bind(s.u),
		builder.InOut("resu").Bind(s.resu),
		builder.Output("rhsu").Bind(s.rhsu),

		// Geometric factors from tetMesh (field arrays, not matrices)
		builder.Input("Rx").Bind(s.tetMesh.Rx),
		builder.Input("Ry").Bind(s.tetMesh.Ry),
		builder.Input("Rz").Bind(s.tetMesh.Rz),
		builder.Input("Sx").Bind(s.tetMesh.Sx),
		builder.Input("Sy").Bind(s.tetMesh.Sy),
		builder.Input("Sz").Bind(s.tetMesh.Sz),
		builder.Input("Tx").Bind(s.tetMesh.Tx),
		builder.Input("Ty").Bind(s.tetMesh.Ty),
		builder.Input("Tz").Bind(s.tetMesh.Tz),

		// RK stage scalars
		builder.Scalar("rka").Bind(s.rka),
		builder.Scalar("rkb").Bind(s.rkb),

		// Temporary arrays for kernel computation
		builder.Temp("ur").Type(builder.Float64).Size(s.K*s.Np),
		builder.Temp("us").Type(builder.Float64).Size(s.K*s.Np),
		builder.Temp("ut").Type(builder.Float64).Size(s.K*s.Np),
	)

	err := s.runner.DefineBindings(params...)
	if err != nil {
		return fmt.Errorf("DefineBindings failed: %w", err)
	}

	// Phase 1: Allocate device memory
	err = s.runner.AllocateDevice()
	if err != nil {
		return fmt.Errorf("AllocateDevice failed: %w", err)
	}

	// Phase 2: Configure kernels
	err = s.configureKernels(props)
	if err != nil {
		return fmt.Errorf("failed to configure kernels: %w", err)
	}

	return nil
}

func (s *Burgers3DSolver) configureKernels(props element.ElementProperties) error {
	// Configure RHS computation kernel
	_, err := s.runner.ConfigureKernel("computeRHS",
		s.runner.Param("u"),               // Input: current solution (already on device)
		s.runner.Param("rhsu").CopyBack(), // Output: RHS

		// Reference operators (static matrices, already embedded)
		s.runner.Param("Dr_"+props.ShortName),
		s.runner.Param("Ds_"+props.ShortName),
		s.runner.Param("Dt_"+props.ShortName),

		// Geometry (copy to device once)
		s.runner.Param("Rx").CopyTo(),
		s.runner.Param("Ry").CopyTo(),
		s.runner.Param("Rz").CopyTo(),
		s.runner.Param("Sx").CopyTo(),
		s.runner.Param("Sy").CopyTo(),
		s.runner.Param("Sz").CopyTo(),
		s.runner.Param("Tx").CopyTo(),
		s.runner.Param("Ty").CopyTo(),
		s.runner.Param("Tz").CopyTo(),

		// Temporary arrays
		s.runner.Param("ur"),
		s.runner.Param("us"),
		s.runner.Param("ut"),
	)
	if err != nil {
		return fmt.Errorf("failed to configure RHS kernel: %w", err)
	}

	// Configure update kernel for RK stages
	_, err = s.runner.ConfigureKernel("updateSolution",
		s.runner.Param("u").CopyTo().CopyBack(),    // InOut: solution
		s.runner.Param("resu").CopyTo().CopyBack(), // InOut: RK residual
		s.runner.Param("rhsu"),                     // Input: RHS (already on device)
		s.runner.Param("rka"),                      // Scalar: RK coefficient a
		s.runner.Param("rkb"),                      // Scalar: RK coefficient b
	)
	if err != nil {
		return fmt.Errorf("failed to configure update kernel: %w", err)
	}

	// Build kernels
	err = s.buildKernels(props)
	if err != nil {
		return fmt.Errorf("failed to build kernels: %w", err)
	}

	return nil
}

func (s *Burgers3DSolver) buildKernels(props element.ElementProperties) error {
	// Build RHS computation kernel
	rhsSignature, err := s.runner.GetKernelSignatureForConfig("computeRHS")
	if err != nil {
		return fmt.Errorf("failed to get RHS signature: %w", err)
	}

	rhsKernelSource := fmt.Sprintf(`
@kernel void computeRHS(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		// Get partition data pointers
		const double* u = u_PART(part);
		double* rhsu = rhsu_PART(part);
		double* ur = ur_PART(part);
		double* us = us_PART(part);
		double* ut = ut_PART(part);
		
		// Access geometric factors
		const double* Rx = Rx_PART(part);
		const double* Ry = Ry_PART(part);
		const double* Rz = Rz_PART(part);
		const double* Sx = Sx_PART(part);
		const double* Sy = Sy_PART(part);
		const double* Sz = Sz_PART(part);
		const double* Tx = Tx_PART(part);
		const double* Ty = Ty_PART(part);
		const double* Tz = Tz_PART(part);
		
		// Step 1: Compute reference space derivatives using matrix operations
		// ur = Dr * u, us = Ds * u, ut = Dt * u
        MATMUL_Dr_%s(u, ur, K[part]);
        MATMUL_Ds_%s(u, us, K[part]);
        MATMUL_Dt_%s(u, ut, K[part]);

		for (int elem = 0; elem < K[part]; ++elem; @inner) {
			// Element base index
			const int base = elem * %d;
			
			// Step 2: Transform to physical space and compute Burgers RHS
			for (int i = 0; i < %d; ++i) {
				const int idx = base + i;
				
				// Physical space derivatives: ux = rx*ur + sx*us + tx*ut, etc.
				double ux = Rx[idx] * ur[idx] + Sx[idx] * us[idx] + Tx[idx] * ut[idx];
				double uy = Ry[idx] * ur[idx] + Sy[idx] * us[idx] + Ty[idx] * ut[idx];
				double uz = Rz[idx] * ur[idx] + Sz[idx] * us[idx] + Tz[idx] * ut[idx];
				
				// Burgers equation RHS: -u * (∂u/∂x + ∂u/∂y + ∂u/∂z)
				rhsu[idx] = -u[idx] * (ux + uy + uz);
			}
		}
	}
}`, rhsSignature, // 1: %s - kernel signature
		props.ShortName, // 5: %s - Dr matrix name
		props.ShortName, // 6: %s - Ds matrix name
		props.ShortName, // 7: %s - Dt matrix name
		s.Np,            // 2: %d - Element base offset
		s.Np,            // 8: %d - Loop bounds for RHS
	)

	_, err = s.runner.BuildKernel(rhsKernelSource, "computeRHS")
	if err != nil {
		return fmt.Errorf("failed to build RHS kernel: %w", err)
	}

	// Build update kernel
	updateSignature, err := s.runner.GetKernelSignatureForConfig("updateSolution")
	if err != nil {
		return fmt.Errorf("failed to get update signature: %w", err)
	}

	updateKernelSource := fmt.Sprintf(`
@kernel void updateSolution(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* u = u_PART(part);
		double* resu = resu_PART(part);
		const double* rhsu = rhsu_PART(part);
		
		for (int elem = 0; elem < K[part]; ++elem; @inner) {
			const int base = elem * %d;
			
			for (int i = 0; i < %d; ++i) {
				const int idx = base + i;
				// Low-storage RK update: resu = rka*resu + rkb*rhsu
				resu[idx] = rka * resu[idx] + rkb * rhsu[idx];
				// Update solution: u = u + resu
				u[idx] = u[idx] + resu[idx];
			}
		}
	}
}`, updateSignature, s.Np, s.Np)

	_, err = s.runner.BuildKernel(updateKernelSource, "updateSolution")
	if err != nil {
		return fmt.Errorf("failed to build update kernel: %w", err)
	}

	return nil
}

func (s *Burgers3DSolver) initializeGaussianPulse() {
	fmt.Printf("Initializing Gaussian pulse (σ=%.6f, A=%.2f)...\n", s.sigma, GaussianAmplitude)

	for k := 0; k < s.K; k++ {
		for i := 0; i < s.Np; i++ {
			idx := k*s.Np + i

			// Get node coordinates from tetMesh
			x := s.tetMesh.X.At(i, k)
			y := s.tetMesh.Y.At(i, k)
			z := s.tetMesh.Z.At(i, k)

			// Gaussian initial condition: u₀ = A * exp(-r²/σ²)
			r2 := (x-PulseCenter)*(x-PulseCenter) +
				(y-PulseCenter)*(y-PulseCenter) +
				(z-PulseCenter)*(z-PulseCenter)

			s.u[idx] = GaussianAmplitude * math.Exp(-r2/(s.sigma*s.sigma))
		}
	}

	// Initialize residual to zero
	for i := range s.resu {
		s.resu[i] = 0.0
	}
}

func (s *Burgers3DSolver) computeTimeStep() {
	// Estimate maximum velocity (approximately the peak of Gaussian)
	maxVel := GaussianAmplitude

	// Compute minimum element size for 3D tetrahedra
	hMin := computeMinimumEdgeLength(s.tetMesh)

	// CFL condition: dt = CFL * h / |u_max|
	// s.dt = 0.01 * CFL * hMin / maxVel
	s.dt = CFL * hMin / maxVel

	fmt.Printf("Time step: dt = %.6f (CFL=%.2f, h_min=%.6f, u_max=%.2f)\n",
		s.dt, CFL, hMin, maxVel)
}

func (s *Burgers3DSolver) Solve() error {
	nSteps := int(math.Ceil(FinalTime / s.dt))

	fmt.Printf("Time integration: %d steps to t=%.3f\n", nSteps, FinalTime)

	// Copy initial solution to device
	err := s.runner.CopyToDevice("u")
	if err != nil {
		return fmt.Errorf("failed to copy initial solution: %w", err)
	}

	err = s.runner.CopyToDevice("resu")
	if err != nil {
		return fmt.Errorf("failed to copy initial residual: %w", err)
	}

	for step := 0; step < nSteps; step++ {
		// RK5SSP4 time stepping
		for stage := 0; stage < 5; stage++ {
			// Update scalar parameters for this RK stage
			s.updateRKCoefficients(stage)

			// Compute RHS
			err := s.runner.ExecuteKernel("computeRHS")
			if err != nil {
				return fmt.Errorf("RHS computation failed at step %d, stage %d: %w", step, stage, err)
			}

			// Update solution
			err = s.runner.ExecuteKernel("updateSolution")
			if err != nil {
				return fmt.Errorf("solution update failed at step %d, stage %d: %w", step, stage, err)
			}
		}

		s.time += s.dt

		// Progress output
		if step%100 == 0 || step == nSteps-1 {
			fmt.Printf("Step %d/%d, time = %.6f\n", step+1, nSteps, s.time)
		}
	}

	// Copy final solution back to host
	err = s.runner.CopyFromDevice("u")
	if err != nil {
		return fmt.Errorf("failed to copy final solution: %w", err)
	}

	return nil
}

func (s *Burgers3DSolver) updateRKCoefficients(stage int) {
	// Update RK coefficients for current stage
	s.rka = s.rk4a[stage]
	s.rkb = s.rk4b[stage] * s.dt

	// Update scalar bindings by copying to device
	// Note: For scalars that change during execution, we need to copy them
	s.runner.CopyToDevice("rka")
	s.runner.CopyToDevice("rkb")
}

func (s *Burgers3DSolver) OutputResults() error {
	// Create output directory
	err := os.MkdirAll("results", 0755)
	if err != nil {
		return fmt.Errorf("failed to create results directory: %w", err)
	}

	// Extract solution along a line for comparison with analytical solution
	linePoints, numericalSol, analyticalSol := s.extractLineData()

	// Write CSV file
	csvFile := filepath.Join("results", "burgers_3d_results.csv")
	err = s.writeCSV(csvFile, linePoints, numericalSol, analyticalSol)
	if err != nil {
		return fmt.Errorf("failed to write CSV: %w", err)
	}

	// Write gnuplot script
	gnuFile := filepath.Join("results", "plot_burgers.gnu")
	err = s.writeGnuplotScript(gnuFile)
	if err != nil {
		return fmt.Errorf("failed to write gnuplot script: %w", err)
	}

	// Compute and report error
	s.computeError(numericalSol, analyticalSol)

	return nil
}

func (s *Burgers3DSolver) extractLineData() ([]float64, []float64, []float64) {
	linePoints := make([]float64, NumPoints)
	numericalSol := make([]float64, NumPoints)
	analyticalSol := make([]float64, NumPoints)

	// Create line from (LineStart,0,0) to (LineEnd,0,0)
	for i := 0; i < NumPoints; i++ {
		x := LineStart + float64(i)*(LineEnd-LineStart)/float64(NumPoints-1)
		linePoints[i] = x

		// Interpolate numerical solution at this point
		numericalSol[i] = s.interpolateSolution(x, 0.0, 0.0)

		// Compute analytical solution (solve implicit equation iteratively)
		analyticalSol[i] = s.computeAnalyticalSolution(x, 0.0, 0.0, s.time)
	}

	return linePoints, numericalSol, analyticalSol
}

func (s *Burgers3DSolver) interpolateSolution(x, y, z float64) float64 {
	// Simple nearest-neighbor interpolation
	// In production, use proper finite element interpolation

	minDist := math.Inf(1)
	var value float64

	for k := 0; k < s.K; k++ {
		for i := 0; i < s.Np; i++ {
			xi := s.tetMesh.X.At(i, k)
			yi := s.tetMesh.Y.At(i, k)
			zi := s.tetMesh.Z.At(i, k)

			dist := math.Sqrt((x-xi)*(x-xi) + (y-yi)*(y-yi) + (z-zi)*(z-zi))
			if dist < minDist {
				minDist = dist
				value = s.u[k*s.Np+i]
			}
		}
	}

	return value
}

func (s *Burgers3DSolver) computeAnalyticalSolution(x, y, z, t float64) float64 {
	// Solve implicit equation: u₀ = A*exp(-((x-u₀*t)² + (y-u₀*t)² + (z-u₀*t)²)/σ²)
	// Using Newton's method

	// Initial guess
	u0 := GaussianAmplitude * math.Exp(-((x*x + y*y + z*z) / (s.sigma * s.sigma)))

	// Newton iteration
	for iter := 0; iter < 10; iter++ {
		// Current function value
		r2 := (x-u0*t)*(x-u0*t) + (y-u0*t)*(y-u0*t) + (z-u0*t)*(z-u0*t)
		f := u0 - GaussianAmplitude*math.Exp(-r2/(s.sigma*s.sigma))

		// Derivative
		expTerm := math.Exp(-r2 / (s.sigma * s.sigma))
		dfdu0 := 1.0 + GaussianAmplitude*expTerm*2.0*t*(x+y+z-3.0*u0*t)/(s.sigma*s.sigma)

		// Newton update
		if math.Abs(dfdu0) > 1e-12 {
			u0_new := u0 - f/dfdu0
			if math.Abs(u0_new-u0) < 1e-10 {
				break
			}
			u0 = u0_new
		}
	}

	return u0
}

func (s *Burgers3DSolver) writeCSV(filename string, x, numerical, analytical []float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write header
	fmt.Fprintf(file, "x,numerical,analytical,error\n")

	// Write data
	for i := 0; i < len(x); i++ {
		error := numerical[i] - analytical[i]
		fmt.Fprintf(file, "%.6f,%.6f,%.6f,%.6f\n", x[i], numerical[i], analytical[i], error)
	}

	return nil
}

func (s *Burgers3DSolver) writeGnuplotScript(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	script := `#!/usr/bin/gnuplot

set terminal png enhanced size 1200,800
set output 'burgers_3d_comparison.png'

set xlabel 'x'
set ylabel 'u(x,0,0,t)'
set title '3D Scalar Burgers Equation: Numerical vs Analytical Solution'
set grid

plot 'burgers_3d_results.csv' using 1:2 with lines linewidth 2 title 'Numerical (DG)', \
     'burgers_3d_results.csv' using 1:3 with lines linewidth 2 title 'Analytical', \
     'burgers_3d_results.csv' using 1:4 with lines linewidth 2 title 'Error'

set output 'burgers_3d_error.png'
set ylabel 'Error'
set title '3D Scalar Burgers Equation: Error Analysis'

plot 'burgers_3d_results.csv' using 1:4 with lines linewidth 2 title 'Numerical Error'
`

	fmt.Fprint(file, script)
	return nil
}

func (s *Burgers3DSolver) computeError(numerical, analytical []float64) {
	if len(numerical) != len(analytical) {
		fmt.Printf("Error: mismatched array lengths\n")
		return
	}

	var l2Error, maxError float64
	var l2Norm float64

	for i := 0; i < len(numerical); i++ {
		err := numerical[i] - analytical[i]
		l2Error += err * err
		l2Norm += analytical[i] * analytical[i]

		absErr := math.Abs(err)
		if absErr > maxError {
			maxError = absErr
		}
	}

	l2Error = math.Sqrt(l2Error)
	l2Norm = math.Sqrt(l2Norm)
	relativeL2 := l2Error / l2Norm

	fmt.Printf("\nError Analysis:\n")
	fmt.Printf("L2 Error: %.6e\n", l2Error)
	fmt.Printf("Relative L2 Error: %.6e (%.3f%%)\n", relativeL2, relativeL2*100)
	fmt.Printf("Maximum Error: %.6e\n", maxError)

	// Assessment based on our discussion
	if relativeL2 < 0.01 {
		fmt.Printf("Assessment: Excellent accuracy (<1%%)\n")
	} else if relativeL2 < 0.03 {
		fmt.Printf("Assessment: Good accuracy (<3%%)\n")
	} else if relativeL2 < 0.05 {
		fmt.Printf("Assessment: Acceptable accuracy (<5%%)\n")
	} else {
		fmt.Printf("Assessment: Poor accuracy (>5%%) - consider mesh refinement\n")
	}
}

func (s *Burgers3DSolver) Free() {
	if s.runner != nil {
		s.runner.Free()
	}
}

// Helper functions

func computeMinimumEdgeLength(tn *tetnudg.TetNudgMesh) float64 {
	minLen := math.Inf(1)

	// Compute minimum edge length across all elements
	for k := 0; k < tn.K; k++ {
		// For each element, check all possible edges between nodes
		for i := 0; i < tn.Np; i++ {
			for j := i + 1; j < tn.Np; j++ {
				x1, y1, z1 := tn.X.At(i, k), tn.Y.At(i, k), tn.Z.At(i, k)
				x2, y2, z2 := tn.X.At(j, k), tn.Y.At(j, k), tn.Z.At(j, k)

				length := math.Sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1))
				if length > 1e-12 && length < minLen { // Avoid zero-length "edges"
					minLen = length
				}
			}
		}
	}

	return minLen
}

// FaceBuffer manages face data exchange for single partition DG solver
type FaceBuffer struct {
	// Dimensions
	K      int // Number of elements
	Np     int // Nodes per element
	Nfp    int // Face points per face
	Nfaces int // Faces per element (4 for tetrahedra)

	// Pick/place indices for local connections
	PickIndices  []int32 // Where to pick from solution array (size: K*Nfaces*Nfp)
	PlaceIndices []int32 // Where to place in M buffer (size: K*Nfaces*Nfp)

	// Host buffers
	MBuffer []float64 // Face values in natural order (size: K*Nfaces*Nfp)
	PBuffer []float64 // Neighbor face values (size: K*Nfaces*Nfp)
}

// BuildFaceBuffer creates the face connectivity for a single partition
func BuildFaceBuffer(tn *tetnudg.TetNudgMesh) (*FaceBuffer, error) {
	fb := &FaceBuffer{
		K:      tn.K,
		Np:     tn.Np,
		Nfp:    tn.Nfp,
		Nfaces: 4, // Tetrahedra have 4 faces
	}

	totalFacePoints := fb.K * fb.Nfaces * fb.Nfp
	fb.PickIndices = make([]int32, totalFacePoints)
	fb.PlaceIndices = make([]int32, totalFacePoints)
	fb.MBuffer = make([]float64, totalFacePoints)
	fb.PBuffer = make([]float64, totalFacePoints)

	// VmapM and VmapP should already be built by the mesh
	if len(tn.NUDGTet.VmapM) != totalFacePoints {
		return nil, fmt.Errorf("VmapM size mismatch: expected %d, got %d",
			totalFacePoints, len(tn.NUDGTet.VmapM))
	}
	if len(tn.NUDGTet.VmapP) != totalFacePoints {
		return nil, fmt.Errorf("VmapP size mismatch: expected %d, got %d",
			totalFacePoints, len(tn.NUDGTet.VmapP))
	}

	// Build pick and place indices
	for i := 0; i < totalFacePoints; i++ {
		// Pick index: VmapM gives us the global volume node to pick from
		globalVolNode := tn.NUDGTet.VmapM[i]
		fb.PickIndices[i] = int32(globalVolNode)

		// For place index, we need to find where VmapP[i] appears in VmapM
		// VmapP[i] gives the global volume node in the neighbor that matches this face point
		targetGlobalNode := tn.NUDGTet.VmapP[i]

		// Check if boundary (VmapP == VmapM for boundary nodes)
		if targetGlobalNode == globalVolNode {
			// Boundary face - place points to itself in M buffer
			fb.PlaceIndices[i] = int32(i)
		} else {
			// Interior face - find where targetGlobalNode appears in VmapM
			found := false
			for j := 0; j < totalFacePoints; j++ {
				if tn.NUDGTet.VmapM[j] == targetGlobalNode {
					fb.PlaceIndices[i] = int32(j)
					found = true
					break
				}
			}

			if !found {
				// This shouldn't happen if VmapP is correctly built
				return nil, fmt.Errorf("failed to find VmapP[%d]=%d in VmapM",
					i, targetGlobalNode)
			}
		}
	}

	return fb, nil
}

// UpdateSolver modifies the existing Burgers3DSolver to include face exchange
func (s *Burgers3DSolver) setupFaceExchange() error {
	// Build face buffer
	var err error
	s.faceBuffer, err = BuildFaceBuffer(s.tetMesh)
	if err != nil {
		return fmt.Errorf("failed to build face buffer: %w", err)
	}

	// Add face buffer arrays to runner bindings
	// This needs to be done during setupDGKernel, so we'll modify that function
	return nil
}

// ExtendSetupDGKernel adds face exchange parameters to the existing setup
func (s *Burgers3DSolver) extendSetupDGKernelForFaces(params []*builder.ParamBuilder) []*builder.ParamBuilder {
	// Add face-related arrays
	params = append(params,
		// M and P buffers for face values
		builder.Temp("MBuffer").Type(builder.Float64).Size(s.faceBuffer.K*s.faceBuffer.Nfaces*s.faceBuffer.Nfp),
		builder.Temp("PBuffer").Type(builder.Float64).Size(s.faceBuffer.K*s.faceBuffer.Nfaces*s.faceBuffer.Nfp),

		// Pick/place indices (static, embedded in kernel)
		builder.Input("PickIndices").Bind(s.faceBuffer.PickIndices),
		builder.Input("PlaceIndices").Bind(s.faceBuffer.PlaceIndices),

		// Surface geometry needed for flux computation
		builder.Input("nx").Bind(s.tetMesh.Nx),
		builder.Input("ny").Bind(s.tetMesh.Ny),
		builder.Input("nz").Bind(s.tetMesh.Nz),
		builder.Input("Fscale").Bind(s.tetMesh.Fscale),

		// LIFT operator for surface integration
		builder.Input("LIFT").Bind(s.tetMesh.LIFT).ToMatrix().Static(),

		// Temporary flux buffer
		builder.Temp("fluxBuffer").Type(builder.Float64).Size(s.faceBuffer.K*s.faceBuffer.Nfaces*s.faceBuffer.Nfp),
	)

	return params
}

// BuildFaceExchangeKernels creates the kernels for face operations
func (s *Burgers3DSolver) buildFaceExchangeKernels() error {
	// 1. Gather face values kernel
	gatherSignature, err := s.runner.GetKernelSignatureForConfig("gatherFaces")
	if err != nil {
		return fmt.Errorf("failed to get gather signature: %w", err)
	}

	gatherKernel := fmt.Sprintf(`
@kernel void gatherFaces(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* u = u_PART(part);
		double* MBuffer = MBuffer_PART(part);
		const int32* PickIndices = PickIndices_PART(part);
		
		const int totalFacePoints = K[part] * %d * %d;
		
		for (int i = 0; i < totalFacePoints; ++i; @inner) {
			MBuffer[i] = u[PickIndices[i]];
		}
	}
}`, gatherSignature, s.faceBuffer.Nfaces, s.faceBuffer.Nfp)

	_, err = s.runner.BuildKernel(gatherKernel, "gatherFaces")
	if err != nil {
		return fmt.Errorf("failed to build gather kernel: %w", err)
	}

	// 2. Exchange face values kernel (local copy within partition)
	exchangeSignature, err := s.runner.GetKernelSignatureForConfig("exchangeFaces")
	if err != nil {
		return fmt.Errorf("failed to get exchange signature: %w", err)
	}

	exchangeKernel := fmt.Sprintf(`
@kernel void exchangeFaces(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* MBuffer = MBuffer_PART(part);
		double* PBuffer = PBuffer_PART(part);
		const int32* PlaceIndices = PlaceIndices_PART(part);
		
		const int totalFacePoints = K[part] * %d * %d;
		
		for (int i = 0; i < totalFacePoints; ++i; @inner) {
			PBuffer[i] = MBuffer[PlaceIndices[i]];
		}
	}
}`, exchangeSignature, s.faceBuffer.Nfaces, s.faceBuffer.Nfp)

	_, err = s.runner.BuildKernel(exchangeKernel, "exchangeFaces")
	if err != nil {
		return fmt.Errorf("failed to build exchange kernel: %w", err)
	}

	// 3. Compute numerical flux kernel
	fluxSignature, err := s.runner.GetKernelSignatureForConfig("computeFlux")
	if err != nil {
		return fmt.Errorf("failed to get flux signature: %w", err)
	}

	props := s.tetMesh.GetProperties()
	_ = props
	fluxKernel := fmt.Sprintf(`
@kernel void computeFlux(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* MBuffer = MBuffer_PART(part);
		const double* PBuffer = PBuffer_PART(part);
		const double* nx = nx_PART(part);
		const double* ny = ny_PART(part);
		const double* nz = nz_PART(part);
		const double* Fscale = Fscale_PART(part);
		double* fluxBuffer = fluxBuffer_PART(part);
		
		for (int elem = 0; elem < K[part]; ++elem; @inner) {
			const int base = elem * %d * %d;
			
			for (int face = 0; face < %d; ++face) {
				for (int fp = 0; fp < %d; ++fp) {
					const int idx = base + face * %d + fp;
					
					// Get M and P values
					const double uM = MBuffer[idx];
					const double uP = PBuffer[idx];
					
					// Lax-Friedrichs flux for Burgers
					const double alpha = fmax(fabs(uM), fabs(uP));
					const double Fn = 0.5 * (uM*uM + uP*uP) * (nx[idx] + ny[idx] + nz[idx]);
					const double flux = 0.5 * Fn - 0.5 * alpha * (uP - uM);
					
					// Scale by face Jacobian
					fluxBuffer[idx] = flux * Fscale[idx];
				}
			}
		}
	}
}`, fluxSignature, s.faceBuffer.Nfaces, s.faceBuffer.Nfp,
		s.faceBuffer.Nfaces, s.faceBuffer.Nfp, s.faceBuffer.Nfp)

	_, err = s.runner.BuildKernel(fluxKernel, "computeFlux")
	if err != nil {
		return fmt.Errorf("failed to build flux kernel: %w", err)
	}

	// Configure kernels
	_, err = s.runner.ConfigureKernel("gatherFaces",
		s.runner.Param("u"),                    // Current solution (already on device)
		s.runner.Param("MBuffer"),              // Output M buffer
		s.runner.Param("PickIndices").CopyTo(), // Copy indices once
	)
	if err != nil {
		return fmt.Errorf("failed to configure gather kernel: %w", err)
	}

	_, err = s.runner.ConfigureKernel("exchangeFaces",
		s.runner.Param("MBuffer"),               // Input M buffer
		s.runner.Param("PBuffer"),               // Output P buffer
		s.runner.Param("PlaceIndices").CopyTo(), // Copy indices once
	)
	if err != nil {
		return fmt.Errorf("failed to configure exchange kernel: %w", err)
	}

	_, err = s.runner.ConfigureKernel("computeFlux",
		s.runner.Param("MBuffer"),     // M values
		s.runner.Param("PBuffer"),     // P values
		s.runner.Param("nx").CopyTo(), // Surface normals (copy once)
		s.runner.Param("ny").CopyTo(),
		s.runner.Param("nz").CopyTo(),
		s.runner.Param("Fscale").CopyTo(), // Face scaling (copy once)
		s.runner.Param("fluxBuffer"),      // Output flux
	)
	if err != nil {
		return fmt.Errorf("failed to configure flux kernel: %w", err)
	}

	return nil
}

// Modified computeRHSWithFaces executes the face exchange
func (s *Burgers3DSolver) computeRHSWithFaces() error {
	// Step 1: Compute volume derivatives (existing kernel)
	err := s.runner.ExecuteKernel("computeRHS")
	if err != nil {
		return fmt.Errorf("volume RHS failed: %w", err)
	}

	// Step 2: Gather face values to M buffer
	err = s.runner.ExecuteKernel("gatherFaces")
	if err != nil {
		return fmt.Errorf("gather faces failed: %w", err)
	}

	// Step 3: Exchange to create P buffer
	err = s.runner.ExecuteKernel("exchangeFaces")
	if err != nil {
		return fmt.Errorf("exchange faces failed: %w", err)
	}

	// Step 4: Compute numerical flux
	err = s.runner.ExecuteKernel("computeFlux")
	if err != nil {
		return fmt.Errorf("compute flux failed: %w", err)
	}

	// Note: LIFT operation will be added in the next step
	// For now, the flux is computed but not yet applied to the RHS

	return nil
}
