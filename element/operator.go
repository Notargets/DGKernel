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

func GetRefMatrices(el ReferenceElement) (refMats map[string]mat.Matrix) {
	var (
		props = el.GetProperties()
	)

	nm := el.GetNodalModal()
	ro := el.GetReferenceOperators()
	sn := props.ShortName
	refMats = map[string]mat.Matrix{
		"V_" + sn:    nm.V,
		"Vinv_" + sn: nm.Vinv,
		"M_" + sn:    nm.M,
		"Minv_" + sn: nm.Minv,
		"Dr_" + sn:   ro.Dr,
		"Ds_" + sn:   ro.Ds,
		"Dt_" + sn:   ro.Dt,
		"LIFT_" + sn: ro.LIFT,
	}

	return
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
