from __future__ import annotations

import numpy as np


class OverridenFire:
    def __init__(
        self, dt=0.1, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99, a=0.1
    ):
        """Vectorised implementation of default parameter of FIRE class in ase."""

        self.dt = dt
        self.Nsteps = 0
        self.maxstep = 0.2
        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a

    def step_override(
        self,
        f: np.ndarray,
        v: np.ndarray,
        dt: np.ndarray,
        Nsteps: np.ndarray,
        a: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes step as equivalent of FIRE.step method."""
        if v is None:
            v = np.zeros((len(f), len(f[0]), 3))

        else:
            f_reshaped = f.reshape(len(f), -1)
            v_reshaped = v.reshape(len(v), -1)
            vf = np.diag(f_reshaped @ v_reshaped.T)

            vf_positive_mask = (vf > 0.0).reshape((-1, 1, 1))
            vf_negative_mask = ~vf_positive_mask

            vdot_ff = np.diag(f.reshape(len(f), -1) @ f.reshape(len(f), -1).T)
            vdot_vv = np.diag(v.reshape(len(f), -1) @ v.reshape(len(f), -1).T)
            v_positive = (1.0 - a).reshape((-1, 1, 1)) * (
                v * vf_positive_mask
            ) + a.reshape((-1, 1, 1)) * (f * vf_positive_mask) / np.sqrt(
                vdot_ff
            ).reshape(
                -1, 1, 1
            ) * np.sqrt(
                vdot_vv
            ).reshape(
                (-1, 1, 1)
            )
            v_negative = v * vf_negative_mask * 0
            v = v_positive + v_negative

            Nsteps_bigger_than_n_min = Nsteps > self.Nmin
            Nsteps_smaller_than_n_min = ~Nsteps_bigger_than_n_min
            dt_1 = np.min(
                np.vstack(
                    [
                        dt
                        * vf_positive_mask.reshape(-1)
                        * Nsteps_bigger_than_n_min
                        * self.finc,
                        [self.dtmax] * len(f),
                    ]
                ),
                axis=0,
            )
            dt_1b = dt * vf_positive_mask.reshape(-1) * Nsteps_smaller_than_n_min
            dt_2 = dt * vf_negative_mask.reshape(-1) * self.fdec
            dt = dt_1 + dt_1b + dt_2

            Nsteps[vf_positive_mask.reshape(-1)] = (
                Nsteps[vf_positive_mask.reshape(-1)] + 1
            )
            Nsteps_1 = Nsteps * vf_positive_mask.reshape(-1)
            Nsteps_2 = Nsteps * vf_negative_mask.reshape(-1) * 0
            Nsteps = Nsteps_1 + Nsteps_2

            # update a
            a_1 = a * vf_positive_mask.reshape(-1) * Nsteps_bigger_than_n_min * self.fa
            a_2 = a * vf_positive_mask.reshape(-1) * Nsteps_smaller_than_n_min
            a_3 = a * 0 + self.astart * vf_negative_mask.reshape(-1)
            a = a_1 + a_2 + a_3

        v += dt.reshape((-1, 1, 1)) * f
        dr = dt.reshape((-1, 1, 1)) * v

        normdr = np.sqrt(np.diag(dr.reshape(len(dr), -1) @ dr.reshape(len(dr), -1).T))

        update_dr_mask_positive = normdr > self.maxstep
        update_dr_mask_negative = ~update_dr_mask_positive
        dr_1 = dr * update_dr_mask_negative.reshape((-1, 1, 1))

        dr_2 = (
            dr
            * update_dr_mask_positive.reshape((-1, 1, 1))
            * self.maxstep
            / normdr.reshape((-1, 1, 1))
        )
        dr = dr_1 + dr_2

        return v, dt, Nsteps, a, dr

    def converged(self, forces: np.ndarrat, fmax: float) -> bool:
        """Did the optimization converge?"""
        return np.max((forces**2).sum(axis=2), axis=1) < fmax**2
