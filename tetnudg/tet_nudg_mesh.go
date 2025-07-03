package tetnudg

import (
	"fmt"
	"github.com/notargets/DGKernel/element"
	"github.com/notargets/DGKernel/element/library/gonudg"
	"github.com/notargets/DGKernel/mesh"
	"github.com/notargets/DGKernel/mesh/readers"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"strings"
)

type TetNudgMesh struct {
	*gonudg.NUDGTet
	*mesh.Mesh
}

// NewTetNudgMesh creates a new TetNudgMesh from a mesh file
func NewTetNudgMesh(order int, meshfile string) (tn *TetNudgMesh) {
	msh, err := readers.ReadMeshFile(meshfile)
	if err != nil {
		panic(err)
	}
	// Verify that this meshfile has only tetrahedra
	var foundTet bool
	var numTets int
	for _, el := range msh.ElementTypes {
		if el == utils.Tet {
			foundTet = true
			numTets++
		}
	}
	if !foundTet {
		err = fmt.Errorf("mesh file %s does not have any tets", meshfile)
		panic(err)
	}
	fmt.Printf("Meshfile: %s has %d tets...\n", meshfile, numTets)
	tn = &TetNudgMesh{
		Mesh: msh,
	}
	// Recase verts into separate X, Y, Z
	VX := make([]float64, len(msh.Vertices))
	VY := make([]float64, len(msh.Vertices))
	VZ := make([]float64, len(msh.Vertices))
	for i, v := range msh.Vertices {
		VX[i] = v[0]
		VY[i] = v[1]
		VZ[i] = v[2]
	}
	tn.NUDGTet, err = gonudg.NewDG3D(order, VX, VY, VZ, msh.EtoV)
	if err != nil {
		panic(err)
	}
	return
}

// String returns a comprehensive summary of the TetNudgMesh properties
func (t *TetNudgMesh) String() string {
	var sb strings.Builder

	// Header
	sb.WriteString("=== TetNudgMesh Summary ===\n")

	// Reference Element Properties
	refElem := t.GetReferenceElement()
	elemProps := refElem.GetProperties()
	sb.WriteString("\n--- Reference Element Properties ---\n")
	sb.WriteString(fmt.Sprintf("  Name: %s (%s)\n", elemProps.Name, elemProps.ShortName))
	sb.WriteString(fmt.Sprintf("  Type: %v\n", elemProps.Type))
	sb.WriteString(fmt.Sprintf("  Order: %d\n", elemProps.Order))
	sb.WriteString(fmt.Sprintf("  Nodes per element (Np): %d\n", elemProps.Np))
	sb.WriteString(fmt.Sprintf("  Nodes per edge (NEp): %d\n", elemProps.NEp))
	sb.WriteString(fmt.Sprintf("  Nodes per face (NFp): %d\n", elemProps.NFp))
	sb.WriteString(fmt.Sprintf("  Nodes strictly interior (not on vertex, edge or face: %d\n", elemProps.NIp))
	sb.WriteString(fmt.Sprintf("  Vertices per element (NVp): %d\n", elemProps.NVp))
	sb.WriteString(fmt.Sprintf("  Dimensions: %v\n", elemProps.Dimensions))

	// Reference Geometry Summary
	refGeom := refElem.GetReferenceGeometry()
	sb.WriteString("\n--- Reference Geometry ---\n")
	sb.WriteString(fmt.Sprintf("  Reference coordinates (r,s,t) have %d nodes\n", len(refGeom.R)))
	if len(refGeom.R) > 0 {
		sb.WriteString(fmt.Sprintf("  R range: [%.4f, %.4f]\n", minFloat64(refGeom.R), maxFloat64(refGeom.R)))
		sb.WriteString(fmt.Sprintf("  S range: [%.4f, %.4f]\n", minFloat64(refGeom.S), maxFloat64(refGeom.S)))
		sb.WriteString(fmt.Sprintf("  T range: [%.4f, %.4f]\n", minFloat64(refGeom.T), maxFloat64(refGeom.T)))
	}

	// Nodal-Modal Matrices
	nodalModal := refElem.GetNodalModal()
	sb.WriteString("\n--- Nodal-Modal Matrices ---\n")
	if nodalModal.V != nil {
		r, c := nodalModal.V.Dims()
		sb.WriteString(fmt.Sprintf("  Vandermonde matrix V: %d×%d\n", r, c))
	}
	if nodalModal.Vinv != nil {
		r, c := nodalModal.Vinv.Dims()
		sb.WriteString(fmt.Sprintf("  Inverse Vandermonde Vinv: %d×%d\n", r, c))
	}
	if nodalModal.M != nil {
		r, c := nodalModal.M.Dims()
		sb.WriteString(fmt.Sprintf("  Mass matrix M: %d×%d\n", r, c))
	}

	// Reference Operators
	refOps := refElem.GetReferenceOperators()
	sb.WriteString("\n--- Reference Operators ---\n")
	if refOps.Dr != nil {
		r, c := refOps.Dr.Dims()
		sb.WriteString(fmt.Sprintf("  Derivative Dr: %d×%d\n", r, c))
	}
	if refOps.Ds != nil {
		r, c := refOps.Ds.Dims()
		sb.WriteString(fmt.Sprintf("  Derivative Ds: %d×%d\n", r, c))
	}
	if refOps.Dt != nil {
		r, c := refOps.Dt.Dims()
		sb.WriteString(fmt.Sprintf("  Derivative Dt: %d×%d\n", r, c))
	}
	if refOps.LIFT != nil {
		r, c := refOps.LIFT.Dims()
		sb.WriteString(fmt.Sprintf("  LIFT operator: %d×%d\n", r, c))
	}

	// Physical Mesh Properties
	meshProps := t.GetMeshProperties()
	sb.WriteString("\n--- Physical Mesh Properties ---\n")
	sb.WriteString(fmt.Sprintf("  Number of elements: %d\n", meshProps.NumElements))
	sb.WriteString(fmt.Sprintf("  Number of vertices: %d\n", meshProps.NumVertices))
	sb.WriteString(fmt.Sprintf("  Number of faces: %d\n", meshProps.NumFaces))
	sb.WriteString(fmt.Sprintf("  Total degrees of freedom: %d\n", meshProps.NumElements*elemProps.Np))

	// Geometric Transform Properties
	geoTrans := t.GetGeometricTransform()
	sb.WriteString("\n--- Geometric Transform ---\n")
	if geoTrans.J != nil {
		r, c := geoTrans.J.Dims()
		sb.WriteString(fmt.Sprintf("  Jacobian determinant: %d×%d matrix\n", r, c))
		if jMin, jMax := matrixMinMax(geoTrans.J); jMin != 0 || jMax != 0 {
			sb.WriteString(fmt.Sprintf("  Jacobian range: [%.4e, %.4e]\n", jMin, jMax))
		}
	}
	if geoTrans.Rx != nil {
		sb.WriteString(fmt.Sprintf("  Transform metrics present: Rx, Ry, Rz, Sx, Sy, Sz, Tx, Ty, Tz\n"))
		// Sample metric ranges for brevity
		if rxMin, rxMax := matrixMinMax(geoTrans.Rx); rxMin != 0 || rxMax != 0 {
			sb.WriteString(fmt.Sprintf("  Rx range: [%.4f, %.4f]\n", rxMin, rxMax))
		}
	}

	// Surface Geometry Properties
	surfGeom := t.GetSurfaceGeometry()
	sb.WriteString("\n--- Surface Geometry ---\n")
	if surfGeom.Nx != nil {
		sb.WriteString(fmt.Sprintf("  Surface normals present: Nx, Ny, Nz\n"))
	}
	if surfGeom.SJ != nil {
		if sjMin, sjMax := matrixMinMax(surfGeom.SJ); sjMin != 0 || sjMax != 0 {
			sb.WriteString(fmt.Sprintf("  Surface Jacobian (SJ) range: [%.4e, %.4e]\n", sjMin, sjMax))
		}
	}
	if surfGeom.FScale != nil {
		if fsMin, fsMax := matrixMinMax(surfGeom.FScale); fsMin != 0 || fsMax != 0 {
			sb.WriteString(fmt.Sprintf("  Face scale (FScale) range: [%.4e, %.4e]\n", fsMin, fsMax))
		}
	}

	// Additional NUDGTet specific properties if available
	if t.NUDGTet != nil {
		sb.WriteString("\n--- Additional NUDGTet Properties ---\n")
		if t.NUDGTet.VmapM != nil && len(t.NUDGTet.VmapM) > 0 {
			sb.WriteString(fmt.Sprintf("  Face connectivity maps (VmapM/VmapP) size: %d\n", len(t.NUDGTet.VmapM)))
		}
		if t.NUDGTet.MapB != nil && len(t.NUDGTet.MapB) > 0 {
			sb.WriteString(fmt.Sprintf("  Boundary nodes: %d\n", len(t.NUDGTet.MapB)))
		}
		sb.WriteString(fmt.Sprintf("  Face count per element: %d\n", t.NUDGTet.Nfaces))
		sb.WriteString(fmt.Sprintf("  Total face nodes: %d\n", t.NUDGTet.Nfaces*t.NUDGTet.Nfp*meshProps.NumElements))
	}

	sb.WriteString("\n===========================\n")

	return sb.String()
}

// Helper functions for finding min/max of float64 slices
func minFloat64(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	min := s[0]
	for _, v := range s[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxFloat64(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	max := s[0]
	for _, v := range s[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// matrixMinMax extracts the minimum and maximum values from a matrix
func matrixMinMax(m mat.Matrix) (min, max float64) {
	if m == nil {
		return 0, 0
	}

	r, c := m.Dims()
	if r == 0 || c == 0 {
		return 0, 0
	}

	// Initialize with first element
	min = m.At(0, 0)
	max = min

	// Iterate through all elements
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := m.At(i, j)
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}
	}

	return min, max
}
func (t *TetNudgMesh) GetMeshProperties() element.MeshProperties {
	return element.MeshProperties{
		NumElements: t.Mesh.NumElements,
		NumVertices: t.Mesh.NumVertices,
		NumFaces:    t.Mesh.NumFaces,
	}
}

// GetReferenceElement returns reference element properties and operators
func (t TetNudgMesh) GetReferenceElement() element.ReferenceElement {
	return &TetReferenceElement{t.NUDGTet}
}

// GetGeometricTransform returns transformation from reference to physical space
func (t TetNudgMesh) GetGeometricTransform() element.GeometricTransform {
	dg := t.NUDGTet
	return element.GeometricTransform{
		Rx:       dg.Rx,
		Ry:       dg.Ry,
		Rz:       dg.Rz,
		Sx:       dg.Sx,
		Sy:       dg.Sy,
		Sz:       dg.Sz,
		Tx:       dg.Tx,
		Ty:       dg.Ty,
		Tz:       dg.Tz,
		J:        dg.J,
		IsAffine: nil, // Could be computed if needed
	}
}

// GetSurfaceGeometry returns face geometry for flux computations
func (t TetNudgMesh) GetSurfaceGeometry() element.SurfaceGeometry {
	dg := t.NUDGTet
	return element.SurfaceGeometry{
		Nx:     dg.Nx,
		Ny:     dg.Ny,
		Nz:     dg.Nz,
		SJ:     dg.SJ,
		FScale: dg.Fscale,
	}
}

// TetReferenceElement implements element.ReferenceElement interface
type TetReferenceElement struct {
	*gonudg.NUDGTet
}

func (dg *TetNudgMesh) GetRefMatrixMacros() string {
	return element.GenerateMatrixMacros(dg.GetReferenceElement())
}

func (t *TetReferenceElement) GetProperties() element.ElementProperties {
	dg := t.NUDGTet
	rg := t.GetReferenceGeometry()
	return element.ElementProperties{
		Name:       "NUDG Lagrange Tetrahedron Order " + string(rune('0'+dg.N)),
		ShortName:  "NUDGETet" + string(rune('0'+dg.N)),
		Type:       utils.Tet,
		Order:      dg.N,
		Np:         dg.Np,
		NFp:        len(rg.FacePoints[0]),
		NEp:        len(rg.EdgePoints[0]),
		NVp:        len(rg.VertexPoints),
		NIp:        len(rg.InteriorPoints),
		Dimensions: element.D3,
	}
}

func (t *TetReferenceElement) GetReferenceGeometry() element.ReferenceGeometry {
	dg := t.NUDGTet

	// Convert vectors to slices
	r := make([]float64, dg.Np)
	s := make([]float64, dg.Np)
	tt := make([]float64, dg.Np)

	// Create storage for vertex and edge indices
	vtx := make([]int, 4)
	edgePts := make([][]int, 6) // 6 edges in a tetrahedron
	for i := range edgePts {
		edgePts[i] = make([]int, 0) // Will grow as we find edge points
	}
	interiorOnly := make([]int, dg.Np)

	var ii int
	for i := 0; i < dg.Np; i++ {
		r[i] = dg.R[i]
		s[i] = dg.S[i]
		tt[i] = dg.T[i]
		R, S, T := r[i], s[i], tt[i]

		// Check for vertex points
		if (isEq(R, -1) && isEq(S, -1) && isEq(T, -1)) ||
			(isEq(R, 1) && isEq(S, -1) && isEq(T, -1)) ||
			(isEq(R, -1) && isEq(S, 1) && isEq(T, -1)) ||
			(isEq(R, -1) && isEq(S, -1) && isEq(T, 1)) {
			vtx[ii] = i
			ii++
			interiorOnly[i]--
		}
		// Check for edge points (including vertices at endpoints)
		// Edge 0: between vertices 0 (-1,-1,-1) and 1 (1,-1,-1)
		// Points on this edge have S=-1, T=-1
		if isEq(S, -1) && isEq(T, -1) {
			edgePts[0] = append(edgePts[0], i)
			interiorOnly[i]--
		} else if isEq(R, -1) && isEq(T, -1) {
			// Edge 1: between vertices 0 (-1,-1,-1) and 2 (-1,1,-1)
			// Points on this edge have R=-1, T=-1
			edgePts[1] = append(edgePts[1], i)
			interiorOnly[i]--
		} else if isEq(R, -1) && isEq(S, -1) {
			// Edge 2: between vertices 0 (-1,-1,-1) and 3 (-1,-1,1)
			// Points on this edge have R=-1, S=-1
			edgePts[2] = append(edgePts[2], i)
			interiorOnly[i]--
		} else if isEq(T, -1) && isEq(R+S, 0) {
			// Edge 3: between vertices 1 (1,-1,-1) and 2 (-1,1,-1)
			// Points on this edge have T=-1, and R+S=0
			edgePts[3] = append(edgePts[3], i)
			interiorOnly[i]--
		} else if isEq(S, -1) && isEq(R+T, 0) {
			// Edge 4: between vertices 1 (1,-1,-1) and 3 (-1,-1,1)
			// Points on this edge have S=-1, and R+T=0
			edgePts[4] = append(edgePts[4], i)
			interiorOnly[i]--
		} else if isEq(R, -1) && isEq(S+T, 0) {
			// Edge 5: between vertices 2 (-1,1,-1) and 3 (-1,-1,1)
			// Points on this edge have R=-1, and S+T=0
			edgePts[5] = append(edgePts[5], i)
			interiorOnly[i]--
		}
	}

	var intOnly []int
	for i := range interiorOnly {
		if interiorOnly[i] == 0 {
			intOnly = append(intOnly, i)
		}
	}

	return element.ReferenceGeometry{
		R:              r,
		S:              s,
		T:              tt,
		VertexPoints:   vtx,
		EdgePoints:     edgePts,
		FacePoints:     dg.Fmask,
		InteriorPoints: intOnly,
	}
}

func isEq(a, b float64) bool {
	tol := 1.e-9
	return math.Abs(a-b) < tol
}

func (t *TetReferenceElement) GetNodalModal() element.NodalModalMatrices {
	dg := t.NUDGTet

	// Compute mass matrix if not available
	var M mat.Matrix
	if !dg.V.IsEmpty() {
		VT := dg.V.T()
		M = mat.NewDense(dg.Np, dg.Np, nil)
		M.(*mat.Dense).Mul(dg.V, VT)
	}

	// Compute Minv if we have M
	var Minv mat.Matrix
	if M != nil {
		Minv = mat.NewDense(dg.Np, dg.Np, nil)
		err := Minv.(*mat.Dense).Inverse(M)
		if err != nil {
			Minv = nil
		}
	}

	return element.NodalModalMatrices{
		V:    dg.V,
		Vinv: dg.Vinv,
		M:    M,
		Minv: Minv,
	}
}

func (t *TetReferenceElement) GetReferenceOperators() element.ReferenceOperators {
	dg := t.NUDGTet
	return element.ReferenceOperators{
		Dr:   dg.Dr,
		Ds:   dg.Ds,
		Dt:   dg.Dt,
		LIFT: dg.LIFT,
	}
}
