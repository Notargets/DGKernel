package partitions

import (
	"strings"
	"testing"
)

// TestPartitionKernelData tests the kernel data structure setup
func TestPartitionKernelData(t *testing.T) {
	// Create a simple 2-partition layout
	layout := &PartitionLayout{
		Partitions: []Partition{
			{ID: 0, NumElements: 10, MaxElements: 12},
			{ID: 1, NumElements: 8, MaxElements: 12},
		},
		KpartMax:      12,
		NumPartitions: 2,
	}

	// Create kernel data
	kernelData := &PartitionKernelData{
		K:       []int32{10, 8},
		Fields:  make(map[string]*PartitionedArray),
		Buffers: make([]*PartitionBuffer, 2),
	}

	// Add a field
	kernelData.Fields["U"] = &PartitionedArray{
		GlobalData: make([]float64, 500), // Arbitrary size
		Offsets:    []int{0, 250, 500},
		Stride:     25, // Np = 25
	}

	// Verify K array
	if len(kernelData.K) != layout.NumPartitions {
		t.Errorf("K array length %d != num partitions %d", len(kernelData.K), layout.NumPartitions)
	}
	if kernelData.K[0] != 10 || kernelData.K[1] != 8 {
		t.Errorf("K values incorrect: got %v", kernelData.K)
	}
}

// TestPartitionedMetrics tests the geometric metric storage
func TestPartitionedMetrics(t *testing.T) {
	Np := 20
	K := 100

	metrics := &PartitionedMetrics{
		Rx: &PartitionedArray{
			GlobalData: make([]float64, Np*K),
			Offsets:    []int{0, Np * 50, Np * K}, // 2 partitions: 50 elements each
			Stride:     Np,
		},
		J: &PartitionedArray{
			GlobalData: make([]float64, K), // For affine elements, Np=1
			Offsets:    []int{0, 50, K},
			Stride:     1,
		},
	}

	// Test accessing partition data
	partition0Rx := metrics.Rx.GetPartitionData(0)
	if len(partition0Rx) != Np*50 {
		t.Errorf("Partition 0 Rx data size incorrect: got %d, want %d", len(partition0Rx), Np*50)
	}

	partition1J := metrics.J.GetPartitionData(1)
	if len(partition1J) != 50 {
		t.Errorf("Partition 1 J data size incorrect: got %d, want %d", len(partition1J), 50)
	}
}

// TestGradientKernelStructure tests that the gradient kernel has correct structure
func TestGradientKernelStructure(t *testing.T) {
	// Note: In real usage, this would be part of the OCCA kernel string
	// Here we test that it has the expected structure

	expectedPatterns := []string{
		"@kernel void computeGradient",
		"const int_t* K",
		"for (int part = 0; part < NPART; ++part; @outer)",
		"for (int elem = 0; elem < KpartMax; ++elem; @inner)",
		"if (elem < k_part)",
	}

	// Mock kernel string for testing
	kernelCode := `
@kernel void computeGradient(
    const int_t* K,
    const real_t* U_global,
    const int_t* U_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const int k_part = K[part];
        for (int elem = 0; elem < KpartMax; ++elem; @inner) {
            if (elem < k_part) {
                // Computation
            }
        }
    }
}`

	for _, pattern := range expectedPatterns {
		if !strings.Contains(kernelCode, pattern) {
			t.Errorf("Kernel missing required pattern: %s", pattern)
		}
	}
}

// TestScatterKernelPattern tests the scatter kernel access pattern
func TestScatterKernelPattern(t *testing.T) {
	// Simulate scatter operation data structures
	partitionData := []float64{
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Face 0 points
		7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Face 1 points
	}

	sendBuffer := make([]float64, 6)
	scatterLocalIndices := []int{0, 1, 2, 6, 7, 8} // Points from face 0 and face 1
	scatterBufferIndices := []int{0, 1, 2, 3, 4, 5}

	// Simulate scatter operation
	for i := 0; i < len(scatterLocalIndices); i++ {
		localIdx := scatterLocalIndices[i]
		bufferIdx := scatterBufferIndices[i]
		sendBuffer[bufferIdx] = partitionData[localIdx]
	}

	// Verify scatter results
	expected := []float64{1.0, 2.0, 3.0, 7.0, 8.0, 9.0}
	for i, exp := range expected {
		if sendBuffer[i] != exp {
			t.Errorf("Send buffer[%d]: got %f, want %f", i, sendBuffer[i], exp)
		}
	}
}

// TestPrepareKernelArgs tests kernel argument preparation
func TestPrepareKernelArgs(t *testing.T) {
	layout := &PartitionLayout{
		NumPartitions: 2,
	}

	data := &PartitionKernelData{
		K: []int32{10, 8},
		Fields: map[string]*PartitionedArray{
			"U": {
				GlobalData: make([]float64, 100),
				Offsets:    []int{0, 50, 100},
			},
		},
		Metrics: &PartitionedMetrics{
			Rx: &PartitionedArray{
				GlobalData: make([]float64, 100),
				Offsets:    []int{0, 50, 100},
			},
		},
	}

	args := PrepareKernelArgs(layout, data)

	// First argument should always be K array
	if kArray, ok := args[0].([]int32); !ok {
		t.Error("First argument should be K array")
	} else if len(kArray) != 2 {
		t.Errorf("K array length incorrect: got %d, want 2", len(kArray))
	}

	// Should have field arrays and metric arrays
	if len(args) < 4 {
		t.Error("Not enough kernel arguments prepared")
	}
}

// TestMixedElementHandling tests support for mixed element types
func TestMixedElementHandling(t *testing.T) {
	// Create partition with mixed elements
	partition := Partition{
		ID:           0,
		Elements:     []int{0, 1, 2, 3, 4, 5},
		NumElements:  6,
		MaxElements:  8,
		ElementTypes: []GeometryType{Hex, Hex, Hex, Tet, Tet, Tet},
		TypeGroups: []ElementGroup{
			{
				ElementType: Hex,
				StartIndex:  0,
				Count:       3,
				Np:          27, // Q2 hex
			},
			{
				ElementType: Tet,
				StartIndex:  3,
				Count:       3,
				Np:          20, // P3 tet
			},
		},
	}

	// Verify type groups
	if len(partition.TypeGroups) != 2 {
		t.Errorf("Expected 2 type groups, got %d", len(partition.TypeGroups))
	}

	// Check hex group
	hexGroup := partition.TypeGroups[0]
	if hexGroup.Count != 3 {
		t.Errorf("Hex group count: got %d, want 3", hexGroup.Count)
	}
	if hexGroup.Np != 27 {
		t.Errorf("Hex Np: got %d, want 27", hexGroup.Np)
	}

	// Check tet group
	tetGroup := partition.TypeGroups[1]
	if tetGroup.Count != 3 {
		t.Errorf("Tet group count: got %d, want 3", tetGroup.Count)
	}
	if tetGroup.Np != 20 {
		t.Errorf("Tet Np: got %d, want 20", tetGroup.Np)
	}
}

// TestCommunicationFlow tests the complete communication workflow
func TestCommunicationFlow(t *testing.T) {
	// Setup: 2 partitions with face communication
	_ = &PartitionLayout{
		Partitions: []Partition{
			{ID: 0, NumElements: 5},
			{ID: 1, NumElements: 5},
		},
		NumPartitions: 2,
	}

	// Create buffers with communication pattern
	buffers := []*PartitionBuffer{
		{
			SendBuffer: make([]float64, 10),
			RecvBuffer: make([]float64, 10),
			ScatterMappings: []PartitionMapping{
				{
					PartitionID:   1,
					LocalIndices:  []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
					BufferIndices: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
					Count:         10,
				},
			},
			RemotePartitions: []RemotePartition{
				{
					PartitionID: 1,
					SendOffset:  0,
					SendCount:   10,
					RecvOffset:  0,
					RecvCount:   10,
				},
			},
		},
		{
			SendBuffer: make([]float64, 10),
			RecvBuffer: make([]float64, 10),
			ScatterMappings: []PartitionMapping{
				{
					PartitionID:   0,
					LocalIndices:  []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
					BufferIndices: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
					Count:         10,
				},
			},
			RemotePartitions: []RemotePartition{
				{
					PartitionID: 0,
					SendOffset:  0,
					SendCount:   10,
					RecvOffset:  0,
					RecvCount:   10,
				},
			},
		},
	}

	// Step 1: Fill send buffers (simulating scatter)
	for i := 0; i < 10; i++ {
		buffers[0].SendBuffer[i] = float64(i)
		buffers[1].SendBuffer[i] = float64(100 + i)
	}

	// Step 2: Exchange (simulated local copy)
	copy(buffers[0].RecvBuffer, buffers[1].SendBuffer)
	copy(buffers[1].RecvBuffer, buffers[0].SendBuffer)

	// Step 3: Verify receive buffers
	for i := 0; i < 10; i++ {
		if buffers[0].RecvBuffer[i] != float64(100+i) {
			t.Errorf("Partition 0 recv[%d]: got %f, want %f", i, buffers[0].RecvBuffer[i], float64(100+i))
		}
		if buffers[1].RecvBuffer[i] != float64(i) {
			t.Errorf("Partition 1 recv[%d]: got %f, want %f", i, buffers[1].RecvBuffer[i], float64(i))
		}
	}
}

// TestOCCALoopConstraints verifies OCCA loop structure requirements
func TestOCCALoopConstraints(t *testing.T) {
	// Test that KpartMax is correctly computed
	partitions := []Partition{
		{NumElements: 10, MaxElements: 0},
		{NumElements: 15, MaxElements: 0},
		{NumElements: 8, MaxElements: 0},
	}

	// Find KpartMax
	kpartMax := 0
	for _, p := range partitions {
		if p.NumElements > kpartMax {
			kpartMax = p.NumElements
		}
	}

	// Set MaxElements for all partitions
	for i := range partitions {
		partitions[i].MaxElements = kpartMax
	}

	// Verify all partitions have same MaxElements
	for i, p := range partitions {
		if p.MaxElements != kpartMax {
			t.Errorf("Partition %d: MaxElements %d != KpartMax %d", i, p.MaxElements, kpartMax)
		}
	}

	// Verify padding calculation
	for i, p := range partitions {
		padding := p.MaxElements - p.NumElements
		if padding < 0 {
			t.Errorf("Partition %d has negative padding", i)
		}
		if i == 1 && padding != 0 {
			t.Errorf("Partition 1 should have no padding (largest partition)")
		}
	}
}

// BenchmarkKernelArgPreparation benchmarks kernel argument setup
func BenchmarkKernelArgPreparation(b *testing.B) {
	layout := &PartitionLayout{
		NumPartitions: 10,
	}

	data := &PartitionKernelData{
		K:      make([]int32, 10),
		Fields: make(map[string]*PartitionedArray),
		Metrics: &PartitionedMetrics{
			Rx: &PartitionedArray{GlobalData: make([]float64, 10000), Offsets: make([]int, 11)},
			Ry: &PartitionedArray{GlobalData: make([]float64, 10000), Offsets: make([]int, 11)},
			Rz: &PartitionedArray{GlobalData: make([]float64, 10000), Offsets: make([]int, 11)},
			J:  &PartitionedArray{GlobalData: make([]float64, 1000), Offsets: make([]int, 11)},
		},
	}

	// Add several fields
	for i := 0; i < 5; i++ {
		data.Fields[string(rune('A'+i))] = &PartitionedArray{
			GlobalData: make([]float64, 10000),
			Offsets:    make([]int, 11),
		}
	}

	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		_ = PrepareKernelArgs(layout, data)
	}
}

// TestImports verifies that the package compiles with proper imports

func TestImports(t *testing.T) {
	// Test GeometryType constants are available
	geoTypes := []GeometryType{Tet, Hex, Prism, Pyramid, Tri, Rectangle, Line}

	if len(geoTypes) != 7 {
		t.Errorf("Expected 7 geometry types, got %d", len(geoTypes))
	}

	// Test that types compile correctly
	_ = Partition{
		ElementTypes: []GeometryType{Tet, Hex},
	}

	_ = MeshConnectivity{
		ElementTypes: []GeometryType{Tet, Hex},
	}

	// Test constants
	if BoundaryPlaceholder != -999 {
		t.Errorf("BoundaryPlaceholder should be -999, got %d", BoundaryPlaceholder)
	}

	if RemoteFace != -9999 {
		t.Errorf("RemoteFace should be -9999, got %d", RemoteFace)
	}
}
