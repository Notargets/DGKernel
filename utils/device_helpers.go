package utils

import (
	"fmt"
	"github.com/notargets/gocca"
)

// CreateTestDevice creates a Device for testing, preferring parallel backends
func CreateTestDevice(useCudaO ...bool) *gocca.OCCADevice {
	var useCuda bool
	if len(useCudaO) > 0 {
		useCuda = useCudaO[0]
	}
	// Try OpenCL with different JSON formats, then OpenMP, then CUDA, then fall back to Serial
	var backends []string
	if useCuda {
		backends = []string{
			`{"mode": "CUDA", "device_id": 0}`,
			`{"mode": "OpenMP"}`,
			`{"mode": "Serial"}`,
		}
	} else {
		backends = []string{
			`{"mode": "OpenMP"}`,
			`{"mode": "Serial"}`,
		}
	}

	for _, props := range backends {
		device, err := gocca.NewDevice(props)
		if err == nil {
			fmt.Printf("Created %s Device\n", device.Mode())
			return device
		}
	}

	// Should not reach here
	panic("Failed to create any Device")
}
