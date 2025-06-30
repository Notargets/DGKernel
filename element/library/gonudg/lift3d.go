package gonudg

import (
	"gonum.org/v1/gonum/mat"
)

// Lift3D computes the 3D surface to volume lift operator used in DG formulation
// Purpose: Compute 3D surface to volume lift operator used in DG formulation
func (dg *DG3D) Lift3D() error {
	Np := dg.Np
	Nfp := dg.Nfp
	Nfaces := dg.Nfaces

	// Initialize Emat - this will hold face mass matrices
	Emat := mat.NewDense(Np, Nfaces*Nfp, nil)

	// Process each of the 4 faces
	for face := 0; face < Nfaces; face++ {
		// Extract the appropriate 2D coordinates for this face
		var faceR, faceS []float64

		switch face {
		case 0: // Face 1: t = -1, use (r, s)
			faceR = make([]float64, len(dg.Fmask[face]))
			faceS = make([]float64, len(dg.Fmask[face]))
			for i, idx := range dg.Fmask[face] {
				faceR[i] = dg.R[idx]
				faceS[i] = dg.S[idx]
			}

		case 1: // Face 2: s = -1, use (r, t)
			faceR = make([]float64, len(dg.Fmask[face]))
			faceS = make([]float64, len(dg.Fmask[face]))
			for i, idx := range dg.Fmask[face] {
				faceR[i] = dg.R[idx]
				faceS[i] = dg.T[idx]
			}

		case 2: // Face 3: r+s+t = -1, use (s, t)
			faceR = make([]float64, len(dg.Fmask[face]))
			faceS = make([]float64, len(dg.Fmask[face]))
			for i, idx := range dg.Fmask[face] {
				faceR[i] = dg.S[idx]
				faceS[i] = dg.T[idx]
			}

		case 3: // Face 4: r = -1, use (s, t)
			faceR = make([]float64, len(dg.Fmask[face]))
			faceS = make([]float64, len(dg.Fmask[face]))
			for i, idx := range dg.Fmask[face] {
				faceR[i] = dg.S[idx]
				faceS[i] = dg.T[idx]
			}
		}

		// Compute 2D Vandermonde matrix for the face
		VFace := Vandermonde2D(dg.N, faceR, faceS)

		// Compute face mass matrix: massFace = inv(VFace * VFace^T)
		var VFaceVFaceT mat.Dense
		VFaceVFaceT.Mul(VFace, VFace.T())

		var massFace mat.Dense
		err := massFace.Inverse(&VFaceVFaceT)
		if err != nil {
			return err
		}

		// Place face mass matrix into Emat at the appropriate location
		// The C++ code does: Emat(idr, JJ) = massFace
		// This sets a block of Emat where:
		// - rows are the face node indices (Fmask[face])
		// - columns are [face*Nfp : (face+1)*Nfp)
		for i, nodeIdx := range dg.Fmask[face] {
			for j := 0; j < Nfp; j++ {
				colIdx := face*Nfp + j
				Emat.Set(nodeIdx, colIdx, massFace.At(i, j))
			}
		}
	}

	// Compute LIFT = V * (V^T * Emat)
	var VtE mat.Dense
	VtE.Mul(dg.V.T(), Emat)

	dg.LIFT = mat.NewDense(Np, Nfaces*Nfp, nil)
	dg.LIFT.Mul(dg.V, &VtE)

	return nil
}
