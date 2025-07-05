package builder

import (
	"fmt"
	"strings"
)

// GenerateKernelSignature generates the parameter list for kernel functions
// based on allocated arrays and device matrices
func (kb *Builder) GenerateKernelSignature() string {
	var params []string

	// K array is always first
	params = append(params, "const int_t* K")

	// Add device matrices in alphabetical order (same order as expandKernelArgs)
	deviceMatrixNames := make([]string, 0, len(kb.DeviceMatrices))
	for name := range kb.DeviceMatrices {
		deviceMatrixNames = append(deviceMatrixNames, name)
	}
	// Sort for consistent ordering
	for i := 0; i < len(deviceMatrixNames); i++ {
		for j := i + 1; j < len(deviceMatrixNames); j++ {
			if deviceMatrixNames[i] > deviceMatrixNames[j] {
				deviceMatrixNames[i], deviceMatrixNames[j] = deviceMatrixNames[j], deviceMatrixNames[i]
			}
		}
	}
	// Add device matrix parameters
	for _, name := range deviceMatrixNames {
		params = append(params, fmt.Sprintf("const real_t* %s", name))
	}

	// Add allocated arrays (global pointer and offsets for each)
	for _, arrayName := range kb.AllocatedArrays {
		// Determine const-ness based on common patterns
		// (this could be made configurable per array if needed)
		constQualifier := "const"
		if isOutputArray(arrayName) {
			constQualifier = ""
		}

		params = append(params,
			fmt.Sprintf("%s real_t* %s_global", constQualifier, arrayName),
			fmt.Sprintf("const int_t* %s_offsets", arrayName))
	}

	return strings.Join(params, ",\n\t")
}

// GenerateKernelDeclaration generates a complete kernel function declaration
func (kb *Builder) GenerateKernelDeclaration(kernelName string) string {
	return fmt.Sprintf("@kernel void %s(\n\t%s\n)",
		kernelName,
		kb.GenerateKernelSignature())
}

// GenerateKernelTemplate generates a basic kernel template with proper signature
func (kb *Builder) GenerateKernelTemplate(kernelName string, body string) string {
	var sb strings.Builder

	sb.WriteString(kb.GenerateKernelDeclaration(kernelName))
	sb.WriteString(" {\n")
	sb.WriteString("\tfor (int part = 0; part < NPART; ++part; @outer) {\n")

	// Add partition pointer setup for each array
	for _, arrayName := range kb.AllocatedArrays {
		constQualifier := "const"
		if isOutputArray(arrayName) {
			constQualifier = ""
		}
		sb.WriteString(fmt.Sprintf("\t\t%s real_t* %s = %s_PART(part);\n",
			constQualifier, arrayName, arrayName))
	}

	if body != "" {
		sb.WriteString("\n")
		// Indent the body
		lines := strings.Split(body, "\n")
		for _, line := range lines {
			if line != "" {
				sb.WriteString("\t\t")
				sb.WriteString(line)
				sb.WriteString("\n")
			}
		}
	} else {
		sb.WriteString("\t\t// TODO: Add kernel implementation\n")
	}

	sb.WriteString("\t}\n")
	sb.WriteString("}\n")

	return sb.String()
}

// Helper function to determine if an array is likely an output
// This is a simple heuristic - could be made more sophisticated
func isOutputArray(name string) bool {
	lowerName := strings.ToLower(name)

	// Common output array patterns
	outputPatterns := []string{
		"out", "result", "rhs", "residual", "flux",
		"du", "dx", "dy", "dz", // derivatives
		"ur", "us", "ut", // reference derivatives
		"ux", "uy", "uz", // physical derivatives
	}

	for _, pattern := range outputPatterns {
		if strings.Contains(lowerName, pattern) {
			return true
		}
	}

	// Arrays ending in numbers might be outputs (U1, U2, etc.)
	if len(name) > 1 {
		lastChar := name[len(name)-1]
		if lastChar >= '0' && lastChar <= '9' {
			penultimateChar := name[len(name)-2]
			if penultimateChar >= 'A' && penultimateChar <= 'Z' {
				return true
			}
		}
	}

	return false
}

// GetKernelSignatureInfo returns structured information about kernel parameters
// Useful for documentation or code generation
type KernelParameter struct {
	Type     string
	Name     string
	IsConst  bool
	Category string // "system", "matrix", "array_data", "array_offset"
}

func (kb *Builder) GetKernelSignatureInfo() []KernelParameter {
	var params []KernelParameter

	// K array
	params = append(params, KernelParameter{
		Type:     "int_t*",
		Name:     "K",
		IsConst:  true,
		Category: "system",
	})

	// Device matrices
	deviceMatrixNames := make([]string, 0, len(kb.DeviceMatrices))
	for name := range kb.DeviceMatrices {
		deviceMatrixNames = append(deviceMatrixNames, name)
	}
	// Sort for consistent ordering
	for i := 0; i < len(deviceMatrixNames); i++ {
		for j := i + 1; j < len(deviceMatrixNames); j++ {
			if deviceMatrixNames[i] > deviceMatrixNames[j] {
				deviceMatrixNames[i], deviceMatrixNames[j] = deviceMatrixNames[j], deviceMatrixNames[i]
			}
		}
	}
	for _, name := range deviceMatrixNames {
		params = append(params, KernelParameter{
			Type:     "real_t*",
			Name:     name,
			IsConst:  true,
			Category: "matrix",
		})
	}

	// Arrays
	for _, arrayName := range kb.AllocatedArrays {
		isOutput := isOutputArray(arrayName)
		params = append(params,
			KernelParameter{
				Type:     "real_t*",
				Name:     arrayName + "_global",
				IsConst:  !isOutput,
				Category: "array_data",
			},
			KernelParameter{
				Type:     "int_t*",
				Name:     arrayName + "_offsets",
				IsConst:  true,
				Category: "array_offset",
			})
	}

	return params
}
