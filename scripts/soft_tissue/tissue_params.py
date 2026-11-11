"""Material and interface tables for ear-surgery soft tissue, with sources.

Every number has a nominal value, a low-high range, and a source: a citation,
"clinical judgement (user)", or "estimate: <reason>". Unsourced values are
estimates and must be replaced when data exists. This file is the single place
those numbers live; the bench fixtures read them through ``skin_params`` and
``wedge_interface``.

Units: mm, N, MPa (N/mm^2). Energies per area in N/mm (1 N/mm = 1000 J/m^2).

Membrane law (as in peel_model): St Venant-Kirchhoff with modulus E (the soft
"toe" of the curve), plus collagen recruitment k1 e exp(k2 e^2) on the Green
strain e along each fibre axis in tension. k1, k2 are fitted so the tangent
stiffness near failure matches the published linear-region modulus.

Interfaces: bilinear traction-separation laws, given by fracture energy G,
peak traction T and the separation d0 at the peak; the failure separation is
dc = 2 G / T. Attachments to bone or cartilage have separate opening (peel)
and sliding (shear) laws; layer-to-layer planes use one law (user decision).

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/tissue_params.py   # regime report
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402

DENSITY_G_PER_MM3 = 1.1e-3


@dataclass(frozen=True)
class Value:
    nominal: float
    low: float
    high: float
    source: str

    def at(self, corner="nominal"):
        return {"nominal": self.nominal, "low": self.low, "high": self.high}[corner]


def V(nominal, low=None, high=None, source="estimate"):
    return Value(nominal, nominal if low is None else low, nominal if high is None else high, source)


@dataclass(frozen=True)
class Material:
    name: str
    description: str
    E_mpa: Value  # toe (small-strain) modulus
    k1_mpa: Value  # collagen recruitment
    k2: Value  # nominal fit, for display; use k2_at(corner)
    thickness_mm: Value
    fail_green: Value  # Green strain along a fibre at rupture
    Jc_cut_n_per_mm: Value  # scalpel cutting toughness
    Jc_tear_n_per_mm: Value  # tearing toughness
    linear_modulus_mpa: Value  # published linear-region modulus the fit targets
    nu: float = 0.45  # tissue is near incompressible (0.49); 0.45 avoids membrane locking
    density: float = DENSITY_G_PER_MM3

    def k2_at(self, corner="nominal"):
        """Steepness fitted so the tangent reaches the linear modulus at this corner's failure strain."""
        k1 = self.k1_mpa.at(corner)
        if k1 <= 0:
            return 0.0
        return _fit_k2(k1, self.fail_green.at(corner), self.linear_modulus_mpa.at(corner))


@dataclass(frozen=True)
class Law:
    G_n_per_mm: Value
    T_mpa: Value
    delta0_mm: Value

    def bond_law(self, corner="nominal") -> pm.BondLaw:
        G, T, d0 = self.G_n_per_mm.at(corner), self.T_mpa.at(corner), self.delta0_mm.at(corner)
        dc = 2.0 * G / T
        return pm.BondLaw(G, min(d0, 0.5 * dc), dc)


@dataclass(frozen=True)
class Interface:
    name: str
    description: str
    opening: Law
    sliding: Law | None = None  # None: one law on the full separation (layer-to-layer)
    friction: Value | None = None  # after release; recorded, not simulated (user decision)


# -- sources ---------------------------------------------------------------------
ANNAIDH_2012 = ("Ni Annaidh et al. 2012, J Mech Behav Biomed Mater 5:139, doi:10.1016/j.jmbbm.2011.08.016 "
                "(human back skin; linear modulus 83+-35 MPa, UTS 21.6 MPa, failure strain 54%)")
ANNAIDH_GOH = "Ni Annaidh et al. 2012, Ann Biomed Eng 40:1666, doi:10.1007/s10439-012-0542-3 (GOH mu = 0.20 MPa)"
DANTER_1996 = "Danter et al. 1996, Laryngo-Rhino-Otologie 75:91 (auricular cutis 0.8 mm, ultrasound)"
ZWIRNER_2021 = ("Zwirner et al. 2021, Sci Rep 11:2127, doi:10.1038/s41598-020-80448-1 (deep temporal fascia: "
                "thickness 1.46 mm, E 36+-19 MPa, UTS 3.6 MPa, failure strain 17%)")
MCBRIDE_2011 = "McBride et al. 2011, J Biomech 44:1954, doi:10.1016/j.jbiomech.2011.04.036 (ovine periosteum toe 1.9 / linear 25.7 MPa)"
KADLUB_2020 = "Kadlub et al. 2020, Biomech Model Mechanobiol 19:461, doi:10.1007/s10237-019-01221-6 (calvarial periosteum 70+-71 MPa)"
HAMMA_2025 = ("Hamma, Boisson & Huynh, ESMC 2025 abstract, https://esmc2025.sciencesconf.org/593375/document "
              "(porcine mandible periosteum peel 324-695 J/m^2)")
COMLEY_2010 = "Comley & Fleck 2010, J Biomech 43:1823, doi:10.1016/j.jbiomech.2010.02.029 (dermis tear 17 kJ/m^2, adipose 4.1 kJ/m^2)"
USER = "clinical judgement (user)"


def _fit_k2(k1, e_fail, target_tangent):
    """k2 so the collagen tangent k1 exp(k2 e^2)(1 + 2 k2 e^2) reaches the target at e_fail."""
    lo, hi = 0.0, 500.0
    for _ in range(200):
        k2 = 0.5 * (lo + hi)
        x = k2 * e_fail**2
        lo, hi = (k2, hi) if k1 * math.exp(x) * (1 + 2 * x) < target_tangent else (lo, k2)
    return 0.5 * (lo + hi)


def _material(name, description, E, k1, linear, thickness, fail, Jc_cut, Jc_tear, scale=1.0, scale_source=None):
    """Build a material whose k2 is fitted to its linear-region modulus at failure."""
    k2 = _fit_k2(k1.nominal * scale, fail.nominal, linear.nominal * scale) if k1.nominal > 0 else 0.0
    note = f"fit: collagen tangent reaches the linear-region modulus ({linear.nominal * scale:g} MPa) at the failure strain"
    if scale != 1.0:
        E = V(E.nominal * scale, E.low * scale, E.high * scale, f"{scale_source}; scaled from {E.source}")
        k1 = V(k1.nominal * scale, k1.low * scale, k1.high * scale, f"{scale_source}; scaled from {k1.source}")
        linear = V(linear.nominal * scale, linear.low * scale, linear.high * scale, f"{scale_source}; scaled")
    return Material(name, description, E, k1, V(k2, k2, k2, note), thickness, fail, Jc_cut, Jc_tear, linear)


_periosteum_E = V(2.0, 1.0, 4.0, MCBRIDE_2011 + " (toe)")
_periosteum_k1 = V(1.0, 0.5, 2.0, "estimate: sets the start of recruitment")
_periosteum_linear = V(40.0, 18.0, 70.0, KADLUB_2020 + "; mandibular 18 MPa (Debelmas 2018); ovine 25.7 MPa")
_skin_E = V(0.6, 0.3, 1.0, "estimate: ~3 mu, " + ANNAIDH_GOH)
_skin_k1 = V(1.0, 0.5, 2.0, "estimate: sets the start of recruitment")
_skin_linear = V(83.0, 48.0, 118.0, ANNAIDH_2012)

MATERIALS = {
    m.name: m
    for m in (
        _material(
            "skin", "postauricular / auricular / scalp skin; also cartilaginous canal skin",
            _skin_E, _skin_k1, _skin_linear,
            V(1.5, 0.8, 2.5, "estimate between auricular 0.8 mm (" + DANTER_1996 + ") and scalp"),
            V(0.45, 0.28, 0.73, "Green strain; " + ANNAIDH_2012 + " (54% static, 25% dynamic)"),
            V(1.5, 0.5, 3.0, "estimate: scalpel cutting toughness 0.5-3 kJ/m^2"),
            V(17.0, 2.8, 30.0, COMLEY_2010),
        ),
        _material(
            "bony_canal_skin", "skin of the bony ear canal: very thin, fused to periosteum",
            _skin_E, _skin_k1, _skin_linear,
            V(0.1, 0.03, 0.2, "estimate: secondary sources disagree, 30-50 um vs 0.1-0.2 mm"),
            V(0.45, 0.28, 0.73, "as skin"),
            V(1.5, 0.5, 3.0, "as skin"),
            V(17.0, 2.8, 30.0, "as skin"),
        ),
        _material(
            "tpf", "subcutis + temporoparietal fascia: soft; bound to skin, the loose areolar plane lies beneath it",
            V(0.01, 0.001, 0.05, "estimate: 1-10 kPa"), V(0.0, 0.0, 0.0, "no recruitment modelled"),
            V(0.01, 0.001, 0.05, "estimate"),
            V(1.0, 0.2, 2.0, "estimate: areolar layer 0.2-1 mm; TPF ~1-2 mm"),
            V(1.0, 0.5, 2.0, "estimate: very extensible"),
            V(0.5, 0.1, 1.0, "estimate"),
            V(4.1, 2.9, 5.3, COMLEY_2010 + " (adipose)"),
        ),
        _material(
            "temporalis_fascia", "deep temporal fascia: very strong, about as strong as periosteum",
            V(2.0, 1.0, 4.0, "estimate: toe; " + USER + " strong as periosteum"), V(2.0, 1.0, 4.0, "estimate"),
            V(36.0, 17.0, 55.0, ZWIRNER_2021),
            V(1.46, 1.05, 1.87, ZWIRNER_2021),
            V(0.18, 0.11, 0.25, "Green strain; " + ZWIRNER_2021),
            V(2.0, 1.0, 5.0, "estimate"),
            V(5.0, 2.0, 10.0, "estimate: fascia tearing 2-10 kJ/m^2"),
        ),
        _material(
            "periosteum", "mastoid periosteum",
            _periosteum_E, _periosteum_k1, _periosteum_linear,
            V(0.25, 0.1, 0.4, "estimate: 0.1-0.4 mm"),
            V(0.3, 0.2, 0.48, "estimate: failure 20-40% engineering strain"),
            V(1.0, 0.5, 2.0, "estimate"),
            V(5.0, 2.0, 10.0, "estimate"),
        ),
        _material(
            "perichondrium", "auricular perichondrium: like periosteum, somewhat weaker",
            _periosteum_E, _periosteum_k1, _periosteum_linear,
            V(0.15, 0.05, 0.2, "estimate: 0.05-0.2 mm"),
            V(0.3, 0.2, 0.48, "as periosteum"),
            V(0.7, 0.35, 1.4, USER + ": periosteum x 0.7"),
            V(3.5, 1.4, 7.0, USER + ": periosteum x 0.7"),
            scale=0.7, scale_source=USER + ": periosteum x 0.7",
        ),
    )
}


# Shell materials: plates that bend with their Poisson ratio (``peel_model.Sheet(bending="shell")``,
# curved rest shape), not bonded sheets; tested by ``plate_test.py``, not the peel/wedge sweep.
NAYYER_2014 = "Nayyer et al. 2014, Nanomedicine 10:235, doi:10.1016/j.nano.2013.06.006 (5.02 MPa, UTS 2.02 MPa, 40.6+-28% at break)"
KHAN_2024 = ("Khan et al. 2024, J Craniofac Surg 35:768, doi:10.1097/SCS.0000000000010015 (review: human auricular cartilage "
             "tension 2.8-5.8 MPa; Zahnert 2000 concha 3.4 / tragus 2.8, Park 2004 5.11, Lem 2022 5.81)")
ROY_2004 = "Roy et al. 2004, J Biomed Mater Res A 68:597, doi:10.1002/jbm.a.10068 (porcine auricular bending modulus 4.6 MPa)"
BROCKHOFF_2014 = "Brockhoff et al. 2014, J Oral Maxillofac Surg 72:2248, doi:10.1016/j.joms.2014.04.024 (concha 1.15-1.25 mm)"
GRIFFIN_2016 = "Griffin et al. 2016, Ann Biomed Eng 44:3460, doi:10.1007/s10439-016-1688-1 (helix/antihelix thinner than concha)"
SHELL_MATERIALS = {
    "auricular_cartilage": Material(
        "auricular_cartilage", "auricular elastic cartilage: a curved plate (the auricle's framework)",
        V(5.0, 2.8, 5.8, NAYYER_2014 + "; " + KHAN_2024 + "; bending 4.6 MPa (" + ROY_2004 + ")"),
        V(0.0, 0.0, 0.0, "no recruitment modelled (near-linear in tension)"),
        V(0.0, 0.0, 0.0, "none"),
        V(1.0, 0.8, 1.25, "concha 1.2 mm (" + BROCKHOFF_2014 + "); helix/antihelix ~0.8 mm (estimate; " + GRIFFIN_2016 + ")"),
        V(0.48, 0.13, 0.93, "Green strain from 40% (12-69%) engineering at break; " + NAYYER_2014),
        V(1.0, 0.5, 2.0, "estimate: cartilage cutting"),
        V(1.0, 0.5, 2.0, "estimate: cartilage tearing ~1 kJ/m^2"),
        V(5.0, 2.8, 5.8, "as E (linear)"),
        nu=0.3,  # estimate: not measured for auricular cartilage
        density=1.08e-3,  # estimate from 73% water (Zopf 2015)
    ),
}


# Fan materials: the auricular muscles and ligaments as tension-only fibres (``fibres.py``), not sheets;
# tested by ``fibre_test.py``. Sources: AURICULAR_MUSCLES_LITERATURE.md.
MORROW_2010 = "Morrow et al. 2010, J Mech Behav Biomed Mater 3:124, PMID 19878911 (rabbit EDL along fibres: 447+-98 kPa, failure 0.51)"
ZWIRNER_2020_TEMPORALIS = "Zwirner et al. 2020, PMID 32517987 (human temporalis 1.58 MPa, UTS 0.26 MPa)"
HOJO_2024 = "Hojo et al. 2024 (human zygomaticus MR elastography 6.8 kPa at rest, 15.5 kPa stretched)"
ZWIRNER_2020_FASCIA = "Zwirner et al. 2020b (human deep temporal fascia, median 26.2 MPa, failure 14.1%)"
CHON_2021 = "Chon et al. 2021, Aesthetic Plast Surg 45:309, doi:10.1007/s00266-020-02045-x (auricularis superior 4.7 x 5.0 cm, 12 sides)"
DONOFRIO_2024 = "Donofrio et al. 2024, World Neurosurg 186:e721, PMID 38616028 (auricularis posterior 28.3 mm, 18 sides)"
CHI_2021 = "Chi et al. 2021, Arch Aesthetic Plast Surg 27:12, doi:10.14730/aaps.2020.02313 (posterior ligament: deep mastoid fascia to eminentia conchae)"
GRAY = "Gray's Anatomy (1918)"

FAN_MATERIALS = {
    m.name: m
    for m in (
        _material(
            "auricular_muscle", "passive auricular (mimetic) muscle along its fibres: soft, J-shaped",
            V(0.02, 0.01, 0.05, "estimate: toe ~3x shear stiffness; " + HOJO_2024),
            V(0.02, 0.01, 0.05, "estimate: sets the start of recruitment"),
            V(0.45, 0.35, 1.58, MORROW_2010 + "; up to " + ZWIRNER_2020_TEMPORALIS),
            V(1.0, 0.5, 2.0, "per fan, see FANS"),
            V(0.64, 0.33, 1.0, "Green strain from 0.51 (0.29-0.73) engineering; " + MORROW_2010),
            V(1.0, 0.5, 2.0, "estimate"), V(2.0, 1.0, 4.0, "estimate"),
        ),
        _material(
            "auricular_ligament", "extrinsic auricular ligament: a short fascial band, much stiffer than the muscles",
            V(2.0, 1.0, 4.0, "estimate: toe, like periosteum"), V(2.0, 1.0, 4.0, "estimate"),
            V(26.0, 22.5, 36.0, ZWIRNER_2020_FASCIA + " as a stand-in; scalp 22.5, deep temporal fascia 36 MPa"),
            V(1.0, 0.5, 2.0, "per band, see FANS"),
            V(0.15, 0.10, 0.20, "Green strain from 14% (deep temporal fascia); " + ZWIRNER_2020_FASCIA),
            V(2.0, 1.0, 5.0, "estimate"), V(5.0, 2.0, 10.0, "estimate"),
        ),
    )
}


@dataclass(frozen=True)
class FanSpec:
    """One fan holding the auricle: where it runs, its size (origin width -> insertion width) and material."""
    name: str
    material: str
    origin: str
    insertion: str
    length_mm: Value
    origin_width_mm: Value
    insertion_width_mm: Value
    thickness_mm: Value


FANS = {
    f.name: f
    for f in (
        FanSpec("superior", "auricular_muscle", "galea / temporoparietal fascia above the ear (" + CHON_2021 + ")",
                "upper cranial surface of the auricle: eminentia / fossa triangularis (" + GRAY + "; Chon: conchal cartilage)",
                V(45.0, 40.0, 47.0, "user's figure (central 4.5 cm); " + CHON_2021 + " mean 4.7"),
                V(49.0, 40.0, 50.0, "user's figure (4.9 cm); " + CHON_2021 + " mean 5.0"),
                V(10.0, 5.0, 15.0, "estimate: thin flat tendon (" + GRAY + ")"),
                V(0.75, 0.5, 1.0, "estimate: described as thin; no measurement")),
        FanSpec("anterior", "auricular_muscle", "lateral galea / temporoparietal fascia in front of the ear (" + GRAY + ")",
                "spina helicis / root of the helical crus (" + GRAY + ")",
                V(25.0, 20.0, 30.0, "estimate: the smallest of the three; no measurement"),
                V(25.0, 20.0, 30.0, "estimate"), V(5.0, 3.0, 8.0, "estimate"),
                V(0.4, 0.2, 0.5, "estimate: pale and indistinct (" + GRAY + ")")),
        FanSpec("posterior", "auricular_muscle", "mastoid surface, by short aponeurotic fibres (" + DONOFRIO_2024 + ")",
                "ponticulus of the eminentia conchae (" + GRAY + ")",
                V(28.0, 25.0, 31.0, DONOFRIO_2024),
                V(5.7, 4.3, 8.8, "5.70 +- 1.36 mm, 20 cadaver sides (World Neurosurgery 2024, as used for the atlas "
                  "posterior auricular muscle); a 99-specimen series ~8.8 mm; 2-3 fascicles"),
                V(8.0, 5.0, 10.0, "estimate"), V(1.5, 1.0, 2.0, "estimate: fleshy fascicles")),
        FanSpec("posterior_ligament", "auricular_ligament", "deep mastoid fascia (" + CHI_2021 + ")",
                "cymba concha at the eminentia conchae (" + CHI_2021 + ")",
                V(12.0, 10.0, 15.0, "estimate: the auricle-mastoid gap"), V(7.0, 5.0, 10.0, "estimate"),
                V(7.0, 5.0, 10.0, "estimate"), V(0.7, 0.5, 1.0, "estimate: ~5 mm^2 section")),
        FanSpec("anterior_ligament", "auricular_ligament", "root of the zygomatic process (" + GRAY + ")",
                "tragus and spina helicis (" + GRAY + ")",
                V(7.0, 5.0, 10.0, "estimate: short pre-auricular band"), V(5.0, 3.0, 7.0, "estimate"),
                V(5.0, 3.0, 7.0, "estimate"), V(0.8, 0.5, 1.0, "estimate: ~3-5 mm^2 section")),
    )
}

# Whole-ear check (for the real-ear pull): recoil of the auricle folded forward, 84 +- 10 N intact,
# 53 N without the posterior muscle, 40 N also without the posterior ligament (Chi 2021). End of range,
# no displacement reported: an upper bound, to be confirmed clinically.
WHOLE_EAR_FOLD_N = V(84.0, 60.0, 100.0, CHI_2021 + ": end-of-range fold, upper bound")


def fan_law(spec: str | FanSpec, corner="nominal") -> dict:
    """Fibre law (E_mpa, k1_mpa, k2) for ``fibres.fan``."""
    f = FANS[spec] if isinstance(spec, str) else spec
    m = FAN_MATERIALS[f.material]
    return dict(E_mpa=m.E_mpa.at(corner), k1_mpa=m.k1_mpa.at(corner), k2=m.k2_at(corner))


def shell_params(name="auricular_cartilage", corner="nominal", thickness_mm=None) -> pm.SkinParams:
    """Plate parameters for a shell material (no bonds, no floor)."""
    m = SHELL_MATERIALS[name]
    return pm.SkinParams(E_mpa=m.E_mpa.at(corner), nu=m.nu,
                         thickness_mm=thickness_mm if thickness_mm is not None else m.thickness_mm.at(corner),
                         contact_n_per_mm3=0.0)


def _law(G, T, d0):
    return Law(G, T, d0)


AREOLAR = _law(V(0.02, 0.005, 0.05, "estimate: 5-50 J/m^2; no peel data for loose areolar planes"),
               V(0.01, 0.002, 0.02, "estimate: 2-20 kPa"), V(0.3, 0.1, 0.5, "estimate: stringy, stretches before letting go"))

INTERFACES = {
    i.name: i
    for i in (
        Interface("areolar", "loose areolar plane beneath TPF (above periosteum / temporalis fascia): the dissection plane",
                  AREOLAR),
        Interface(
            "skin_subcutis", "skin bound to subcutis / TPF (dermis-hypodermis)",
            _law(V(0.2, 0.05, 0.5, "estimate: 50-500 J/m^2; below adipose tearing 4.1 kJ/m^2 (" + COMLEY_2010 + ")"),
                 V(0.05, 0.01, 0.1, "estimate: 10-100 kPa (porcine skin peel model, arXiv 2506.19855)"),
                 V(0.1, 0.05, 0.2, "estimate")),
        ),
        Interface(
            "periosteum_bone", "periosteum on bone (Sharpey's fibres)",
            _law(V(0.5, 0.324, 0.695, HAMMA_2025), V(0.3, 0.1, 0.5, "estimate: 0.1-0.5 MPa"),
                 V(0.05, 0.02, 0.1, "estimate")),
            _law(V(1.5, 0.65, 2.8, "estimate: G_II = 2-4 x G_I; no tissue data"), V(0.5, 0.2, 0.8, "estimate"),
                 V(0.05, 0.02, 0.1, "estimate")),
            V(0.1, 0.05, 0.3, "estimate"),
        ),
        Interface(
            "perichondrium_cartilage", "perichondrium on cartilage: weaker than periosteum, slippery in shear",
            _law(V(0.35, 0.2, 0.5, USER + ": weaker than periosteum on bone (x 0.7)"), V(0.21, 0.07, 0.35, USER + ": x 0.7"),
                 V(0.05, 0.02, 0.1, "estimate")),
            _law(V(0.2, 0.05, 0.4, "estimate: slippery; lower than its opening energy"), V(0.1, 0.03, 0.2, "estimate"),
                 V(0.05, 0.02, 0.1, "estimate")),
            V(0.03, 0.01, 0.05, "estimate: lubricated cartilage"),
        ),
        Interface(
            "skin_perichondrium", "auricular skin on perichondrium (anterior: tightly adherent)",
            _law(V(0.2, 0.05, 0.5, "estimate: tight; between areolar and periosteum"), V(0.1, 0.03, 0.2, "estimate"),
                 V(0.1, 0.05, 0.2, "estimate")),
        ),
        Interface(
            "temporalis_fascia_muscle", "temporalis fascia on the muscle (rigid stand-in until the muscle volume exists)",
            _law(V(0.1, 0.03, 0.3, "estimate"), V(0.05, 0.02, 0.1, "estimate"), V(0.1, 0.05, 0.2, "estimate")),
            _law(V(0.1, 0.03, 0.3, "estimate"), V(0.05, 0.02, 0.1, "estimate"), V(0.1, 0.05, 0.2, "estimate")),
        ),
        Interface(
            "canal_skin_periosteum", "bony canal skin fused to periosteum: they lift as one",
            _law(V(2.0, 1.0, 4.0, "estimate: fused; far stronger than periosteum on bone"), V(1.0, 0.5, 2.0, "estimate"),
                 V(0.02, 0.01, 0.05, "estimate")),
        ),
        Interface(
            "annulus", "tympanic annulus seated in its sulcus (ring)",
            _law(V(1.5, 0.65, 2.8, "estimate: 2-5 x canal skin on bone"), V(0.6, 0.3, 1.0, "estimate"),
                 V(0.05, 0.02, 0.1, "estimate")),
            _law(V(4.5, 2.0, 8.0, "estimate"), V(1.0, 0.5, 1.5, "estimate"), V(0.05, 0.02, 0.1, "estimate")),
        ),
    )
}

# Stacks, outside to inside, then the substrate and its interface.
STACKS = {
    "mastoid": {"layers": ["skin", "tpf", "periosteum"], "between": ["skin_subcutis", "areolar"],
                "substrate": "bone", "attachment": "periosteum_bone"},
    "temporal": {"layers": ["skin", "tpf", "temporalis_fascia"], "between": ["skin_subcutis", "areolar"],
                 "substrate": "muscle stand-in (rigid)", "attachment": "temporalis_fascia_muscle"},
    "bony_canal": {"layers": ["bony_canal_skin", "periosteum"], "between": ["canal_skin_periosteum"],
                   "substrate": "bone", "attachment": "periosteum_bone"},
    "auricle": {"layers": ["skin", "perichondrium"], "between": ["skin_perichondrium"],
                "substrate": "cartilage (rigid)", "attachment": "perichondrium_cartilage"},
}

# The attachment each material is peeled, wedged and cut with when tested alone.
SOLO_ATTACHMENT = {
    "skin": "skin_subcutis",
    "bony_canal_skin": "periosteum_bone",
    "tpf": "areolar",
    "temporalis_fascia": "temporalis_fascia_muscle",
    "periosteum": "periosteum_bone",
    "perichondrium": "perichondrium_cartilage",
}


# -- adapters for the existing fixtures ---------------------------------------------
def skin_params(material: str | Material, interface: str | Interface | None = None, corner="nominal",
                contact=None) -> pm.SkinParams:
    """``pm.SkinParams`` for a material, bonded with an interface's opening law."""
    m = MATERIALS[material] if isinstance(material, str) else material
    i = INTERFACES[interface or SOLO_ATTACHMENT[m.name]] if not isinstance(interface, Interface) else interface
    law = i.opening.bond_law(corner)
    E = m.E_mpa.at(corner)
    if contact is None:  # stiff enough that the floor never lets the sheet sink a tenth of its thickness
        contact = max(200.0, 50.0 * law.T_max / max(m.thickness_mm.at(corner), 1e-3))
    return pm.SkinParams(
        E_mpa=E, nu=m.nu, thickness_mm=m.thickness_mm.at(corner), G_n_per_mm=law.G_n_per_mm,
        delta0_mm=law.delta0_mm, delta_c_mm=law.delta_c_mm, contact_n_per_mm3=contact,
        k1_mpa=m.k1_mpa.at(corner), k2=m.k2_at(corner),
    )


def wedge_interface(interface: str | Interface, corner="nominal"):
    import wedge_test as wt

    i = INTERFACES[interface] if isinstance(interface, str) else interface
    sliding = i.sliding or i.opening
    return wt.Interface(i.opening.bond_law(corner), sliding.bond_law(corner))


# -- known answers for a material ------------------------------------------------------
def uniaxial_stress(p: pm.SkinParams, stretch):
    """Nominal (force per reference area) stress of a free-sided strip at this stretch."""
    import stretch_test as st

    return st.exact_force(p, stretch, 1.0) / p.thickness_mm


def kendall_general(p: pm.SkinParams, angle_deg=90.0):
    """Steady peel force per width for an arm with any hyperelastic uniaxial law.

    Energy balance per unit peeled length: G = f (1 - cos a + eps) - h W(eps),
    f the force per width, eps = s - 1 the arm's engineering strain, W its
    stored energy density. Reduces to Kendall's formula for a linear arm.
    """
    import stretch_test as st

    c = 1.0 - math.cos(math.radians(angle_deg))
    h, G = p.thickness_mm, p.G_n_per_mm

    def residual(s):
        f = st.exact_force(p, s, 1.0)
        return float(f * (c + s - 1.0) - h * st.exact_energy_density(p, s) - G)

    lo, hi = 1.0, 1.0 + 1e-6
    while residual(hi) < 0:
        hi = 1.0 + 2.0 * (hi - 1.0)
        if hi > 3.0:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    return float(st.exact_force(p, 0.5 * (lo + hi), 1.0))


def kendall_layers(layers, G, angle_deg=90.0):
    """General Kendall force per width for an arm of several bonded layers (sharing one surface).

    The arm's uniaxial force and stored energy are the sums over its layers.
    """
    import stretch_test as st

    c = 1.0 - math.cos(math.radians(angle_deg))

    def force(s):
        return sum(float(st.exact_force(p, s, 1.0)) for p in layers)

    def residual(s):
        stored = sum(p.thickness_mm * float(st.exact_energy_density(p, s)) for p in layers)
        return force(s) * (c + s - 1.0) - stored - G

    lo, hi = 1.0, 1.0 + 1e-6
    while residual(hi) < 0:
        hi = 1.0 + 2.0 * (hi - 1.0)
        if hi > 3.0:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    return force(0.5 * (lo + hi))


def zone_layers(layers, law: pm.BondLaw):
    """Bending-zone length for a bonded arm of several layers on a bond law."""
    D = sum(p.bending_D for p in layers)
    return (4.0 * D / (law.T_max / law.delta0_mm)) ** 0.25


def lap_shear_force(p: pm.SkinParams, G_ii):
    """Steady force per width to slide an extensible layer off along its plane (Kendall at 0 degrees)."""
    q = pm.SkinParams(**{**p.__dict__, "G_n_per_mm": G_ii})
    return kendall_general(q, 0.0)


def zone_length(p: pm.SkinParams):
    """Elastic-foundation length (4 D / k)^(1/4): how long the breaking zone at a peel front is."""
    k = p.T_max / p.delta0_mm
    return (4.0 * p.bending_D / k) ** 0.25


def strength_per_width(p: pm.SkinParams, fail_green):
    s = math.sqrt(1.0 + 2.0 * fail_green)
    return float(uniaxial_stress(p, s) * p.thickness_mm)


@dataclass
class Regime:
    material: str
    interface: str
    D_n_mm: float
    zone_mm: float
    delta_c_mm: float
    mesh_mm: float
    kendall_n_per_mm: float
    strength_n_per_mm: float
    fails_first: str
    boeing_mm: float
    notes: list = field(default_factory=list)


def regime(material: str, interface: str | None = None, corner="nominal", blade_mm=0.5) -> Regime:
    import wedge_test as wt

    m = MATERIALS[material]
    iname = interface or SOLO_ATTACHMENT[material]
    p = skin_params(m, iname, corner)
    zone = zone_length(p)
    f = kendall_general(p)
    strength = strength_per_width(p, m.fail_green.at(corner))
    out = Regime(material, iname, p.bending_D, zone, p.delta_c_mm, min(0.25, zone / 3.0), f, strength,
                 "interface" if f < strength else "tissue", wt.boeing_crack_length(p.bending_D, blade_mm, p.G_n_per_mm))
    if f >= 0.5 * strength:
        out.notes.append("peel force is over half the tissue strength: close to tearing")
    return out


def report(corners=("nominal",)):
    rows = []
    for corner in corners:
        for name in MATERIALS:
            r = regime(name, corner=corner)
            rows.append((corner, r))
    head = (f"{'corner':8s} {'material':18s} {'interface':24s} {'D N.mm':>9s} {'zone mm':>8s} {'dc mm':>6s} {'mesh mm':>8s} "
            f"{'peel N/mm':>10s} {'strength':>9s} {'fails':>9s} {'Boeing mm':>9s}")
    print(head)
    for corner, r in rows:
        print(f"{corner:8s} {r.material:18s} {r.interface:24s} {r.D_n_mm:9.2e} {r.zone_mm:8.3f} {r.delta_c_mm:6.2f} {r.mesh_mm:8.3f} "
              f"{r.kendall_n_per_mm:10.4f} {r.strength_n_per_mm:9.3f} {r.fails_first:>9s} {r.boeing_mm:9.3f}"
              + ("  " + "; ".join(r.notes) if r.notes else ""))
    print("\nStacks (each layer's own attachment above it):")
    for name, s in STACKS.items():
        print(f"  {name:10s} " + " / ".join(s["layers"]) + f" / {s['substrate']}   planes: "
              + ", ".join(s["between"]) + f", {s['attachment']}")
    return rows


def sources():
    """Every number and its source, for review."""
    out = []
    for m in MATERIALS.values():
        for k, v in m.__dict__.items():
            if isinstance(v, Value):
                out.append((f"material {m.name}", k, v))
    for i in INTERFACES.values():
        for mode in ("opening", "sliding"):
            law = getattr(i, mode)
            if law is None:
                continue
            for k, v in law.__dict__.items():
                out.append((f"interface {i.name} {mode}", k, v))
        if i.friction:
            out.append((f"interface {i.name}", "friction", i.friction))
    return out


if __name__ == "__main__":
    report(("nominal", "high"))
    n_est = sum(1 for _o, _k, v in sources() if v.source.startswith("estimate"))
    print(f"\n{len(sources())} values, {n_est} of them estimates")
