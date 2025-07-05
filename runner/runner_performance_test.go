package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
	"time"
)

// BenchmarkRunner_MatrixOperations benchmarks realistic DG operations
func BenchmarkRunner_MatrixOperations(b *testing.B) {
	sizes := []struct {
		name string
		k    []int
		np   int
	}{
		{"Small_10x20", []int{10}, 20},
		{"Medium_100x20", []int{100}, 20},
		{"Large_1000x20", []int{1000}, 20},
		{"MultiPart_256x256x20", []int{256, 256, 256, 256}, 20},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			benchmarkMatrixOps(b, size.k, size.np)
		})
	}
}

func benchmarkMatrixOps(b *testing.B, k []int, np int) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Calculate total nodes
	totalElements := 0
	for _, ki := range k {
		totalElements += ki
	}
	totalNodes := totalElements * np

	// Create differentiation matrix
	Dr := createTestMatrix(np, np)

	// Host arrays
	hostU := make([]float64, totalNodes)
	hostV := make([]float64, totalNodes)
	hostW := make([]float64, totalNodes)

	// Initialize with test data
	for i := range hostU {
		hostU[i] = 1.0 + float64(i%100)*0.01
	}

	// Define kernel with new API - much simpler!
	err := kp.DefineKernel("matmul",
		Input("Dr").Bind(Dr).ToMatrix().Static(),
		InOut("U").Bind(hostU).CopyTo(),  // Only copy once at start
		Output("V").Bind(hostV).NoCopy(), // Keep on device
		Output("W").Bind(hostW).NoCopy(), // Keep on device
	)
	if err != nil {
		b.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("matmul")

	// Kernel that chains matrix operations
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void matmul(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		real_t* W = W_PART(part);
		
		// Chain of operations: U→V→W→U (3 iterations)
		for (int iter = 0; iter < 3; ++iter) {
			MATMUL_Dr(U, V, K[part]);
			MATMUL_Dr(V, W, K[part]);
			MATMUL_Dr(W, U, K[part]);
		}
	}
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "matmul")
	if err != nil {
		b.Fatalf("Failed to build kernel: %v", err)
	}

	// Warm up
	for i := 0; i < 5; i++ {
		err = kp.RunKernel("matmul")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	// Time one execution to estimate iterations needed
	start := time.Now()
	kp.RunKernel("matmul")
	device.Finish()
	estimatedTime := time.Since(start)

	iterations := computeIterations(estimatedTime)

	// Run benchmark
	b.ResetTimer()
	start = time.Now()
	for i := 0; i < iterations; i++ {
		err = kp.RunKernel("matmul")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()
	b.StopTimer()

	totalTime := time.Since(start)
	avgTime := totalTime / time.Duration(iterations)

	// Calculate metrics
	// 9 matrix multiplies per kernel call (3 iterations × 3 ops)
	// Each matrix multiply is np × np × K operations (2 ops per multiply-add)
	matmulOps := int64(9)
	opsPerMatmul := int64(np * np * totalElements * 2)
	totalOps := matmulOps * opsPerMatmul * int64(iterations)

	gflops := float64(totalOps) / totalTime.Seconds() / 1e9

	b.ReportMetric(gflops, "GFLOPS")
	b.ReportMetric(float64(avgTime.Nanoseconds()), "ns/kernel")
	b.ReportMetric(float64(totalElements), "elements")
	b.ReportMetric(float64(kp.KpartMax), "KpartMax")
}

// BenchmarkRunner_TypeConversion benchmarks the cost of type conversion
func BenchmarkRunner_TypeConversion(b *testing.B) {
	device := utils.CreateTestDevice()
	defer device.Free()

	sizes := []int{10000, 100000, 1000000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			kp := NewRunner(device, builder.Config{
				K:         []int{size},
				FloatType: builder.Float32, // Device uses float32
			})
			defer kp.Free()

			// Host uses float64
			hostData := make([]float64, size)
			hostResult := make([]float64, size)

			for i := range hostData {
				hostData[i] = float64(i) * 1.1
			}

			// Without conversion
			b.Run("NoConversion_f32", func(b *testing.B) {
				hostData32 := make([]float32, size)
				for i := range hostData32 {
					hostData32[i] = float32(hostData[i])
				}

				kp.DefineKernel("process_f32",
					InOut("data").Bind(hostData32).Copy(),
				)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					kp.RunKernel("process_f32")
				}
			})

			// With conversion
			b.Run("WithConversion_f64_to_f32", func(b *testing.B) {
				kp.DefineKernel("process_convert",
					Input("data").Bind(hostData).CopyTo().Convert(builder.Float32),
					Output("result").Bind(hostResult).CopyBack().Convert(builder.Float64),
				)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					kp.RunKernel("process_convert")
				}
			})
		})
	}
}

// BenchmarkRunner_MemoryPatterns tests different memory access patterns
func BenchmarkRunner_MemoryPatterns(b *testing.B) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 20
	k := []int{100, 100, 100, 100} // 4 partitions

	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	totalNodes := 400 * np

	// Different binding patterns
	b.Run("Temp_Arrays", func(b *testing.B) {
		hostU := make([]float64, totalNodes)

		kp.DefineKernel("with_temp",
			Input("U").Bind(hostU).CopyTo(),
			Temp("scratch1").Type(builder.Float64).Size(totalNodes),
			Temp("scratch2").Type(builder.Float64).Size(totalNodes),
		)

		signature, _ := kp.GetKernelSignature("with_temp")
		kernelSource := fmt.Sprintf(`
@kernel void with_temp(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* scratch1 = scratch1_PART(part);
		real_t* scratch2 = scratch2_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				scratch1[i] = U[i] * 2.0;
				scratch2[i] = scratch1[i] + 1.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "with_temp")

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kp.RunKernel("with_temp")
		}
	})
}

func BenchmarkRunner_MemoryPatterns2(b *testing.B) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 20
	k := []int{100, 100, 100, 100} // 4 partitions

	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	totalNodes := 400 * np

	b.Run("InOut_Pattern", func(b *testing.B) {
		hostData := make([]float64, totalNodes)

		kp.DefineKernel("inout",
			InOut("data").Bind(hostData).Copy(),
		)

		signature, _ := kp.GetKernelSignature("inout")
		kernelSource := fmt.Sprintf(`
@kernel void inout(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * 2.0 + 1.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "inout")

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kp.RunKernel("inout")
		}
	})
}

// Helper functions

func createTestMatrix(rows, cols int) mat.Matrix {
	data := make([]float64, rows*cols)
	// Create a simple test pattern
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i == j {
				data[i*cols+j] = 2.0
			} else if abs(i-j) == 1 {
				data[i*cols+j] = -1.0
			}
		}
	}
	return mat.NewDense(rows, cols, data)
}

func computeIterations(estimatedTime time.Duration) int {
	targetTime := 100 * time.Millisecond
	iterations := int(targetTime / estimatedTime)
	if iterations < 10 {
		iterations = 10
	}
	if iterations > 1000 {
		iterations = 1000
	}
	return iterations
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
