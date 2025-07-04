package utils

import (
	"fmt"
	"github.com/notargets/gocca"
)

// CreateTestDevice creates a Device for testing, preferring parallel backends
func CreateTestDevice() *gocca.OCCADevice {
	// Try OpenCL with different JSON formats, then OpenMP, then CUDA, then fall back to Serial
	backends := []string{
		// Try without quotes around numbers
		// `{mode: 'OpenCL', platform_id: 0, device_id: 0}`,
		// Original OpenMP
		`{"mode": "OpenMP"}`,
		`{"mode": "CUDA", "device_id": 0}`,
		`{"mode": "Serial"}`,
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
