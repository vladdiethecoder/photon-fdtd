import YeeEnergy
#print axioms YeeEnergy.energy_conserved
#print axioms YeeEnergy.energy2_conserved
#print axioms YeeEnergy.density_step
#print axioms YeeEnergy.density_step_2d
#check @YeeEnergy.energy_conserved
#check @YeeEnergy.energy2_conserved

-- A concrete instance over Int: period 7 (1D) and 5 x 4 (2D), arbitrary
-- integer fields, coefficients 3 (and 2) far outside the CFL limit. The
-- energies before and after ten steps are printed; they must be equal.
open YeeEnergy in
def step1 (c : Int) (s : (Int → Int) × (Int → Int)) : (Int → Int) × (Int → Int) :=
  (eNext c s.1 s.2, hNext c s.1 s.2)

open YeeEnergy in
def step2 (cx cy : Int) (s : (Int → Int → Int) × (Int → Int → Int) × (Int → Int → Int)) :=
  (eNext2 cx cy s.1 s.2.1 s.2.2, hxNext cy s.1 s.2.1, hyNext cx s.1 s.2.2)

def iter {α : Type} (f : α → α) : Nat → α → α
  | 0, a => a
  | k + 1, a => iter f k (f a)

def e0 (i : Int) : Int := [3, -1, 4, 1, -5, 9, 2].getD (i % 7).toNat 0
def h0 (i : Int) : Int := [2, 7, -1, 8, 2, -8, 1].getD (i % 7).toNat 0
def E0 (i j : Int) : Int := (i % 5) * 3 - (j % 4) * 2 + 1
def X0 (i j : Int) : Int := ((i % 5) * (j % 4)) - 2
def Y0 (i j : Int) : Int := (j % 4) - 2 * (i % 5) + 3

-- Memoise each state on one period (the closures would otherwise re-evaluate
-- the whole history).
def tab1 (s : (Int → Int) × (Int → Int)) : (Int → Int) × (Int → Int) :=
  let a := (List.range 7).map (fun k : Nat => s.1 (k : Int))
  let b := (List.range 7).map (fun k : Nat => s.2 (k : Int))
  (fun i => a.getD (i % 7).toNat 0, fun i => b.getD (i % 7).toNat 0)

def tab2 (s : (Int → Int → Int) × (Int → Int → Int) × (Int → Int → Int)) :=
  let m (f : Int → Int → Int) := (List.range 20).map (fun q : Nat => f ((q : Int) % 5) ((q : Int) / 5))
  let g (l : List Int) : Int → Int → Int := fun i j => l.getD ((i % 5) + 5 * (j % 4)).toNat 0
  (g (m s.1), g (m s.2.1), g (m s.2.2))

#eval (List.range 11).map (fun k =>
  let s := iter (fun s => tab1 (step1 3 s)) k (e0, h0)
  YeeEnergy.energy 7 (3 : Int) s.1 s.2)
#eval (List.range 11).map (fun k =>
  let s := iter (fun s => tab2 (step2 3 2 s)) k (E0, X0, Y0)
  YeeEnergy.energy2 5 4 (3 : Int) 2 s.1 s.2.1 s.2.2)
