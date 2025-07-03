package element

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"strings"
)

type ArgumentType uint8

const (
	INT ArgumentType = iota
	FLOAT32
	FLOAT64
	INTSLICE
	FLOAT32SLICE
	FLOAT64SLICE
	MATMATRIX
)

type OperandType uint8

const (
	INPUT OperandType = iota
	OUTPUT
	TEMPORARY
)

type Argument struct {
	Name    string
	IO      ArgumentType
	Operand OperandType
	Stride  int
}

type Operator interface {
	GetArguments() []Argument
	GetKernelSource() string
}

// generateMatrixMacros creates matrix multiplication macros with @inner loop
func GenerateMatrixMacros(el ReferenceElement) string {
	var (
		props = el.GetProperties()
		sb    strings.Builder
	)

	nm := el.GetNodalModal()
	ro := el.GetReferenceOperators()
	sn := props.ShortName
	StaticMatrices := map[string]mat.Matrix{
		"V_" + sn:    nm.V,
		"Vinv_" + sn: nm.Vinv,
		"M_" + sn:    nm.M,
		"Minv_" + sn: nm.Minv,
		"Dr_" + sn:   ro.Dr,
		"Ds_" + sn:   ro.Ds,
		"Dt_" + sn:   ro.Dt,
		"LIFT_" + sn: ro.LIFT,
	}

	sb.WriteString("// Matrix multiplication macros\n")
	sb.WriteString("// Automatically infer strides from matrix dimensions\n\n")

	for name, matrix := range StaticMatrices {
		rows, cols := matrix.Dims()
		sb.WriteString(fmt.Sprintf("%s\n", formatStaticMatrix(name, matrix, FLOAT64)))

		// Standard multiply: OUT = Matrix × IN
		sb.WriteString(fmt.Sprintf("#define MATMUL_%s(IN, OUT, K_VAL) \\\n", name))
		sb.WriteString("    do { \\\n")
		sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
		sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
		sb.WriteString("                if (elem < (K_VAL)) { \\\n")
		sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
		sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
		sb.WriteString(fmt.Sprintf("                        sum += %s[i][j] * (IN)[elem * %d + j]; \\\n", name, cols))
		sb.WriteString("                    } \\\n")
		sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] = sum; \\\n", rows))
		sb.WriteString("                } \\\n")
		sb.WriteString("            } \\\n")
		sb.WriteString("        } \\\n")
		sb.WriteString("    } while(0)\n\n")

		// Accumulating multiply: OUT += Matrix × IN
		sb.WriteString(fmt.Sprintf("#define MATMUL_ADD_%s(IN, OUT, K_VAL) \\\n", name))
		sb.WriteString("    do { \\\n")
		sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
		sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
		sb.WriteString("                if (elem < (K_VAL)) { \\\n")
		sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
		sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
		sb.WriteString(fmt.Sprintf("                        sum += %s[i][j] * (IN)[elem * %d + j]; \\\n", name, cols))
		sb.WriteString("                    } \\\n")
		sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] += sum; \\\n", rows))
		sb.WriteString("                } \\\n")
		sb.WriteString("            } \\\n")
		sb.WriteString("        } \\\n")
		sb.WriteString("    } while(0)\n\n")
	}

	return sb.String()
}

// formatStaticMatrix formats a single matrix as a static C array
func formatStaticMatrix(name string, m mat.Matrix, ft ArgumentType) string {
	rows, cols := m.Dims()
	var sb strings.Builder

	typeStr := "double"
	if ft == FLOAT32 {
		typeStr = "float"
	}

	sb.WriteString(fmt.Sprintf("const %s %s[%d][%d] = {\n", typeStr, name, rows, cols))

	for i := 0; i < rows; i++ {
		sb.WriteString("    {")
		for j := 0; j < cols; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			val := m.At(i, j)
			if ft == FLOAT32 {
				sb.WriteString(fmt.Sprintf("%.7ef", val))
			} else {
				sb.WriteString(fmt.Sprintf("%.15e", val))
			}
		}
		sb.WriteString("}")
		if i < rows-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("};\n\n")

	return sb.String()
}
