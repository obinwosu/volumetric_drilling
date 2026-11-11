# Auricle pinch-and-pull: literature values

Literature compiled 2026-11-09 to set the starting physics for a hand that pinches and pulls the human auricle. A clinician will tune the values later.

Rules used here:
- Every number has a source, listed in the references at the end. Where a value comes only from an abstract or a secondary table, the source column says so.
- **EST** marks an estimate I made where the literature is thin. The reasoning is given under the table.
- Values depend heavily on how they were measured. Indentation, confined compression, AFM and tension give moduli up to two orders of magnitude apart on the same tissue. Use the test that matches how the sim loads the tissue: tension and bending for a pulled ear, indentation for a pinch.

---

## 1. Auricular (elastic) cartilage

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Tensile Young's modulus | 5.0 | 2.8 | 5.8 | MPa | Human, region not stated, uniaxial tension | Nayyer 2014 (5.02 ± 0.04); Park 2004 (5.11); Lem 2022 (normal adult 5.81, abstract); Zahnert 2000 (concha 3.4, tragus 2.8, fresh cadaver). Nayyer, Park and Zahnert numbers as compiled by Khan 2024 |
| Ultimate tensile strength | 2.5 | 2.0 | 4.0 | MPa | Human, tension to failure | Nayyer 2014 (2.02 ± 0.25); Park 2004 (2.18); Lem 2022 (3.46 adult, 3.99 microtia) |
| Failure (elongation at break) strain | 0.40 | 0.12 | 0.69 | – | Human, tension | Nayyer 2014 (40.6 ± 28 %), via Khan 2024 |
| Compressive Young's modulus (indentation, bulk) | 1.66 | 1.41 (helix) | 2.08 (concha) | MPa | 15 cadavers, indentation 1 mm/s to 300 g | Griffin 2016. Regions: tragus 1.67 ± 0.61, antitragus 1.79 ± 0.56, concha 2.08 ± 0.70, antihelix 1.71 ± 0.63, helix 1.41 ± 0.67 |
| Compressive modulus (nanoindentation, concha) | 1.14 | 0.43 | 1.85 | MPa | 10 fresh-frozen cadavers (66.5 ± 6 y), 78 µm sphere, 1–17 µm deep | Bos 2018 (1.14 ± 0.71) |
| Equilibrium modulus (stress-relaxation indentation) | 3.3 | 2.0 | 4.6 | MPa | Human, 4 male donors 49–93 y, n=17 | Nimeskern 2013 (3.3 ± 1.3). Nimeskern 2015 maps E_in and E_eq by region (helix lowest, antitragus highest), but the numbers were not available to me |
| Equilibrium modulus (confined compression) | 0.11 | 0.06 | 0.16 | MPa | Human, 4 donors 26–71 y, 6 steps of 5 % | Vernice 2024 |
| Instantaneous modulus (confined compression) | 0.15 | 0.09 | 0.21 | MPa | same | Vernice 2024 |
| Aggregate modulus (creep indentation, biphasic) | 0.172 | – | – | MPa | Adult, −80 °C frozen, 0.5 mm tip | Lem 2022 (abstract) |
| Shear modulus (biphasic fit) | 0.086 | – | – | MPa | same | Lem 2022 (abstract) |
| Permeability | 36 × 10⁻¹⁵ | – | – | m⁴/(N·s) | same | Lem 2022; Vernice 2024 gives 0.9 ± 0.65 × 10⁻¹³ m⁴/(N·s) |
| AFM modulus (concha, surface) | 0.032 | 0.014 | 0.050 | MPa | Human concha, AFM | Pappa 2017 (31.8 ± 18 kPa), via Khan 2024. Local surface value, not for bulk use |
| Bending (flexural) modulus | 4.6 | – | – | MPa | **Porcine**, 3-point bending, 20 × 4 × ~1 mm strips, large-deflection plate fit | Roy 2004. The orientation of the perichondrium changed the result. No human bending data found |
| Hyperelastic fit (Yeoh C10, C20) | 0.27, −0.02 | 0.09 | 0.50 | (units not stated, likely MPa) | 24 human punch biopsies, 6 mm, unconfined compression to 60 % | Zopf 2015 (Mooney-Rivlin C10 0.30 ± 0.15, C01 −0.03; Ogden and Fung fits also given) |
| Water content | 73 | 71 | 75 | % wet mass | Human biopsies, 5 sites | Zopf 2015 |
| Elastin content | >15 | – | – | % wet mass | Human, fresh | Nimeskern 2015 (abstract) |
| Poisson's ratio | **EST 0.3** | 0.1 | 0.45 | – | none found | EST, see below |
| Density | **EST 1.08** | 1.05 | 1.10 | g/cm³ | none found | EST from 73 % water (Zopf 2015) |

Notes and estimates:
- **Poisson's ratio (EST):** I found no direct human measurement. Lem 2022's biphasic fit (aggregate modulus 172 kPa, shear modulus 86 kPa) implies a drained Poisson's ratio near 0. Fast (undrained) loading is nearly incompressible (about 0.45 or more). For a thin shell bent and pulled at hand speed, 0.3 is a reasonable middle value.
- **Density (EST):** 1 / (0.73 / 1.00 + 0.27 / 1.35) ≈ 1.07–1.08 g/cm³, taking about 1.35 g/cm³ for the collagen/elastin/GAG solids. This is a mixture-rule estimate only.
- **Which modulus to use:** a pulled ear loads the cartilage in tension and bending. Use about 5 MPa in tension, and 4–5 MPa in bending (Roy 2004, porcine). The 0.1–2 MPa compression and indentation values describe pinching across the thickness.
- Age: Nimeskern 2015 found significant age differences in thickness and composition. Ozturan 2015 found clinical stiffness rising to a peak at 35–39 y and falling after 40.

### Cartilage thickness by region

| region | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Concha | 1.2 | 0.77 | 1.79 | mm | 35 pairs of cadaver ears, 39–99 y, Boley gauge on a 4 × 5 grid | Brockhoff 2014 (female mean 1.15 ± 0.26, male 1.25 ± 0.23; thicker toward anterior) |
| Tragus | 1.15 | 0.89 | 1.43 | mm | 61 tragi harvested in surgery, Indian population | Khan & Parab 2015 (male 1.23 ± 0.20, female 1.09 ± 0.16; <15 y 1.02) |
| Helix, antihelix | **EST 0.8** | 0.5 | 1.0 | mm | – | Griffin 2016: helix and antihelix are significantly thinner than concha, tragus and antitragus (calipers, 15 cadavers), but the abstract gives no mm values. The number is my estimate |
| Antitragus | **EST 1.3** | 1.0 | 1.8 | mm | – | Nimeskern 2015 reports the antitragus as the thickest region (via Khan 2024). Number is EST |
| Whole framework (in vivo) | 0.6 | – | 3 | mm | 47 normal ears, 20 MHz ultrasound | Danter 1996 (mean 0.6, abstract only); Siegert & Magritz 2007 give 1–3 mm |

---

## 2. Auricular perichondrium and skin adhesion

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Perichondrium thickness | **EST 0.1** | 0.05 | 0.2 | mm | none found for the auricle | EST |
| Perichondrium tensile modulus | **EST 20** | 10 | 40 | MPa | none found for the auricle | EST. Stand-in: temporal muscle fascia 36 ± 19 MPa (Zwirner 2021). Human costal perichondrium has been tested in tension (Damron 2024 thesis), but I could not read its numbers |
| Anterior (lateral) skin-to-perichondrium bond | tight, no subcutis | – | – | qualitative | anatomy / surgical reviews | Siegert & Magritz 2007 ("almost immobile"); Pickrell 2017 |
| Posterior (medial) skin bond | loose, slides on a 1.2–3 mm subcutaneous / areolar fat layer | – | – | qualitative | same | Siegert & Magritz 2007; Pickrell 2017 |

Notes:
- **Perichondrium (EST):** perichondrium is a dense fibrous sheet and stiffer in tension than the cartilage. The 10–40 MPa range puts it near fascia (Zwirner 2021: 36 MPa). Roy 2004 found that its orientation changes auricular bending stiffness, so it matters in bending. Model it as a tension-only skin layer on both cartilage faces.
- **For the sim:** tie the anterior skin to the cartilage (shared nodes, or a very stiff tie). Let the posterior skin slide on a soft layer roughly 1–3 mm thick.

---

## 3. Auricular skin thickness

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Anterior skin (whole skin) | 1.0 | 0.8 | 1.2 | mm | review | Siegert & Magritz 2007 |
| Cutis, all sites (in vivo) | 0.8 | – | – | mm | 47 normal ears, 20 MHz B-mode ultrasound | Danter 1996 (abstract) |
| Subcutis (in vivo) | 0.6 | – | – | mm | same | Danter 1996 (abstract) |
| Posterior subcutaneous layer | 2.0 | 1.2 | 3.0 | mm | review | Siegert & Magritz 2007 |
| Postauricular (mastoid) skin, stand-in | 1.23 | 1.16 | 1.30 | mm | 10 fresh-frozen male heads, 77–94 y | Griffin 2017 |
| Postauricular skin tensile modulus, stand-in | 0.86 | 0.81 | 0.91 | MPa | same, tension 1 mm/s parallel to Langer's lines | Griffin 2017. Temporoparietal 0.65, forehead 0.33, submandibular 1.28 |
| Lobule | **EST 5–8** total thickness | – | – | mm | no cartilage, fibrofatty tissue | Siegert & Magritz 2007 (no cartilage). Thickness EST |

No study measured auricular skin on its own in tension. Use postauricular skin (Griffin 2017) as the stand-in.

---

## 4. Whole-auricle mechanics

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Whole-ear stiffness, helix-down compression | 0.19 | ~0.05 | ~0.4 | N/mm | 9 fresh cadaver ears, 7 donors, 72.6 ± 17 y, 10 mm/min, slope at 2–4 mm | Zopf 2015 (0.194 ± 0.202). Linear and compliant at first, then stiffens at large displacement |
| "Recoil force", auricle folded forward | 84 | 60 | 100 | N | 30 patients in surgery, tension gauge on a suture at the postaurale | Chi 2021. Method not standardised; displacement not given. Cutting the posterior auricular ligament removed 13.6 N (16 %); harvesting the cymba concha removed 11.3 N |
| Clinical antihelix stiffness | rises with age to a peak at 35–39 y | – | – | qualitative here | 190 volunteers, digital dynamometer at 4 antihelix points | Ozturan 2015. I could not get the numbers |
| Avulsion / tear-off force | **EST 100** | 50 | 500 | N | none found | EST, see below |
| Lobule stiffness | **EST E ≈ 10** | 5 | 30 | kPa | none found | EST, see below |
| Prosthesis flexibility | no force–displacement data found | – | – | – | – | gap |

Notes:
- **Chi 2021 vs Zopf 2015:** the two disagree by a large factor. Zopf pushed the free helix down by a few mm; Chi folded the whole living ear forward against its ligaments, skin and muscle. An 84 N recoil seems high and may reflect the gauge method. Treat it as an upper bound for the force needed to fold the ear flat, not as a stiffness.
- **Avulsion (EST, low confidence):** no measured value found. Rough check: the cartilage continues into the canal over about 2–3 cm × 1 mm, which is about 25 mm² × 3 MPa UTS ≈ 75 N. The skin ring around the root adds more. A tear-off force of about 100 N (range 50–500 N) is plausible. In practice a finger pinch slips well before this (see items 6–7: 5–20 N pinch × μ 0.5–1 gives a maximum pull of about 3–20 N). Detachment should therefore only matter if the sim allows a pull that does not slip.
- **Lobule (EST):** the lobule is fibrofatty tissue with no cartilage. A homogenised value between fat (a few kPa) and skin (about 1 MPa in tension, much lower in indentation; van Kuilenburg 2013 gives 15–150 kPa) gives about 10 kPa effective in indentation. Clinician to tune.

---

## 5. Extrinsic auricular ligaments and posterior auricular muscle

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Anterior ligament: attachments | tragus and spine of helix → root of zygomatic process | – | – | anatomy | classical anatomy | Gray's Anatomy (1918), via UNSW Embryology page |
| Superior ligament: attachments | upper margin of bony external meatus → spine of helix | – | – | anatomy | reference atlas | IMAIOS e-Anatomy. Not listed in the Gray's 1918 passage; some sources disagree |
| Posterior ligament: attachments | posterior (cranial) surface of concha (eminence of cymba concha) → mastoid surface / deep mastoid fascia | – | – | anatomy | 6 ears, 3 cadavers, 59–72 y | Gray's 1918; Chi 2021 (runs between superficial and deep mastoid fascia, together with the posterior auricular muscle) |
| Ligament count | 1 posterior + 2 anterior | – | – | anatomy | review | Siegert & Magritz 2007 |
| Ligament dimensions | **none found** | – | – | mm | – | gap. **EST** length 5–15 mm, width 3–8 mm, thickness 0.5–1 mm, from the size of the attachment regions |
| Contribution to support | posterior ligament ≈ 16 % of recoil force | – | – | % | in surgery | Chi 2021 |
| Posterior auricular muscle length | 28.3 | – | – | mm | 18 sides, 10 cadaver heads | Donofrio 2024 (mean; the first 20 mm lie over the mastoid). Width not found |
| Muscles generally | small, "relatively unimportant" | – | – | qualitative | review | Siegert & Magritz 2007 |
| **Stand-in ligament material:** temporal muscle fascia modulus | 36 | 17 | 55 | MPa | 74 fresh samples from 55 cadavers, mean 48 y, 0.014 /s | Zwirner 2021 |
| Stand-in UTS | 3.6 | 1.9 | 5.3 | MPa | same | Zwirner 2021 |
| Stand-in failure strain | 0.17 | 0.11 | 0.23 | – | same | Zwirner 2021 |
| Stand-in thickness | 1.46 | 1.05 | 1.87 | mm | same | Zwirner 2021 |

No mechanical data exist for the auricular ligaments. Temporal muscle fascia is the best stand-in: it is a nearby head-and-neck dense fibrous sheet, fresh human, with a full tensile dataset. Model each ligament as a short tension-only spring. Stiffness k = E·A/L. With E = 36 MPa, A ≈ 5 mm² (EST) and L ≈ 10 mm (EST), k ≈ 18 N/mm per ligament.

---

## 6. Fingertip pulp for a pinch grip

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Effective modulus of finger pad (Hertz, ≤0.4 N) | 70 | 50 | 90 | kPa | 26 adults, 34–56 y | Cornuault 2015 (women 82 ± 20, men 59 ± 13) |
| Effective modulus of finger pad (gross contact model) | 31 | 28 | 36 | kPa | in vivo index finger at 30° and 45° | Dzidek 2017 |
| Skin effective modulus, by contact size | 0.015 (10 mm radius) | – | 0.15 (10 µm radius) | MPa | review / analysis | van Kuilenburg 2013 |
| Finger pad radius of curvature | 13 | 10 | 16 | mm | 26 adults | Cornuault 2015 |
| Gross contact area at 1 N | 80 | 70 | 92 | mm² | in vivo, A = k·W^0.37 | Dzidek 2017 (k = 91.9 at 30°, 70.9 at 45°; mm² units inferred) |
| Contact area exponent with load | 0.37 | 0.36 | 0.37 | – | same | Dzidek 2017 |
| Force–displacement | compliant below 1 N, stiffens fast above | – | – | qualitative | 20 subjects, tapping 0.25–3 Hz, in vivo | Serina 1997. Serina 1998 validated a membrane model at 0.25–7 N |
| Viscoelastic relaxation time constants | 4 ms, 70 ms, 1.4 s | – | – | s | in vivo, 0–2 N, flat probe | Pawluk & Howe 1999 (quasi-linear viscoelastic; elastic part rises exponentially with indentation) |
| Static pinch to hold 200 g | 2.0 | 1.37 (sandpaper) | 2.68 (rayon) | N | thumb–index precision grip, young adults | Hiramatsu 2015 |
| Static pinch to hold 6 g | 0.15 | 0.13 | 0.18 | N | same | Hiramatsu 2015 |
| Pinch to hold and pull an ear | **EST 5** | 2 | 15 | N | none found | EST, see below |
| Maximum tip pinch, adult men | ~80 (18 lb) | – | – | N | normative, 20–24 y right hand | Mathiowetz 1985, via secondary norm tables (not checked against the original) |
| Maximum tip pinch, adult women | ~49 (11 lb) | – | – | N | same | same |

Notes:
- **Pinch on an ear (EST):** holding a light object takes 1–3 N (Hiramatsu 2015), and grip scales with the pulling (load) force divided by μ, plus a 20–50 % safety margin. Pulling the auricle with 2–5 N needs roughly 3–10 N of pinch at μ ≈ 0.7. That gives 5 N nominal and 2–15 N as the range. Maximum pinch (about 50–80 N) is the hard ceiling.
- **For the sim:** a Hertz sphere with R = 13 mm and E* ≈ 30–70 kPa matches the measured contact area at 0.5–2 N. Add stiffening above about 1 N (Serina 1997), or let the finger bone take over once the pulp is compressed by about 2–3 mm.

---

## 7. Friction: fingertip on skin

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Skin friction, dry | 0.5 | 0.2 | 0.6 | – | review of skin vs many materials | Derler & Gerhardt 2012 (dry: >0.2, typically about 0.5, independent of pressure) |
| Skin friction, moist / wet | 1.0 | 1.0 | 1.5+ | – | same | Derler & Gerhardt 2012 (moist or wet: typically >1) |
| Finger pad static μ on rayon / sandpaper | 0.71–1.49 | 0.71 | 1.49 | – | thumb–index precision grip | Hiramatsu 2015 |
| Finger pad μ_static on a sticky polymer (ABS) | 2.4 | 1.5 | 3.3 | – | 26 adults | Cornuault 2015. A high-end case |
| Fingertip-on-skin (skin–skin) | **EST 0.7 dry, 1.0 moist** | 0.4 | 1.5 | – | no direct measurement found | EST |
| Load dependence, dry | Coulomb (F ∝ W, n ≈ 1) | – | – | – | finger pad on glass | Dzidek 2017; Derler & Gerhardt 2012 |
| Load dependence, occluded / moist, smooth surface | μ ∝ W^(−1/3) (F ∝ W^(2/3)); μ rises about 5× as load falls over 0.02–2 N after >10 s contact | – | – | – | same | Dzidek 2017 |

Notes:
- No study I found measured fingertip on skin directly. Both surfaces are skin and keratinised, and hydration dominates (Derler & Gerhardt 2012). An estimate of 0.7 dry and 1.0 moist sits in the middle of their ranges. Under irrigation or blood, a gloved or wet finger could fall lower (EST 0.3–0.5). No source; clinician to tune.
- Use Coulomb friction with one μ for the dry case. For long, moist holds, μ = μ₀·(W/W₀)^(−1/3) is a defensible option.

---

## 8. Preauricular / parotid soft tissue (elastic foundation)

| quantity | nominal | low | high | units | specimen / test | source |
|---|---|---|---|---|---|---|
| Parotid stiffness reported by shear-wave elastography (SWE) | 9.0 | 5.5 | 12.5 | kPa (scanner output) | 128 healthy adults, 20–70+ y | Herman 2017 (9.0 ± 3.5) |
| Parotid stiffness reported by SWE | 10.4 | 6.9 | 13.9 | kPa (scanner output) | 127 healthy volunteers | Arda 2011 (10.38 ± 3.5) |
| Implied shear modulus μ | **≈3** | 2 | 4.5 | kPa | derived | EST: SWE scanners usually report E = 3μ. If a study reports μ directly, use it as is. Check each paper's convention |

The parotid is deep to the preauricular skin and SMAS. For an elastic foundation in front of the tragus, a layer about 10–20 mm thick with E ≈ 9–10 kPa (μ ≈ 3 kPa) is a reasonable start.

---

## Gaps and recommended estimates

Biggest gaps, roughly in order of how much they matter for pinch-and-pull:

1. **No whole-auricle pull or bend test** with force against displacement in a direction that matters for this sim (lateral or forward pull). The only data are Zopf 2015 (helix-down compression, 0.19 N/mm) and Chi 2021 (84 N folding recoil, crude method). **Recommend:** calibrate the shell with cartilage E = 5 MPa (tension) / 4.5 MPa (bending), thickness 0.8–1.2 mm by region, and check that a helix-down push gives about 0.2 N/mm at 2–4 mm.
2. **No ligament mechanics or dimensions.** **Recommend:** three tension-only springs at the anatomical attachments, each k ≈ 10–30 N/mm (fascia stand-in, EST geometry). Weight the posterior one by about 16 % of total support (Chi 2021).
3. **No avulsion force.** **Recommend:** a failure threshold around 100 N (EST). Expect the pinch to slip first at 3–20 N of pull.
4. **No perichondrium data.** **Recommend:** a tension-only membrane on both cartilage faces, 0.1 mm thick, E = 20 MPa (EST, fascia-like).
5. **No auricular skin tensile data or regional skin thickness map.** **Recommend:** anterior skin 1.0 mm bonded to the cartilage; posterior skin 1.0 mm on a 2 mm sliding fat layer; postauricular skin E = 0.86 MPa (Griffin 2017).
6. **No Poisson's ratio or density for auricular cartilage.** **Recommend:** ν = 0.3, ρ = 1.08 g/cm³.
7. **No lobule mechanics.** **Recommend:** a soft homogenised solid, E ≈ 10 kPa, about 5–8 mm thick.
8. **No fingertip-on-skin friction.** **Recommend:** μ = 0.7 dry, 1.0 moist, Coulomb.
9. **The Nimeskern 2015 regional E_in / E_eq table** was not reachable (paywalled). It is the best regional map of human auricular cartilage and is worth reading in full before tuning regional stiffness.

---

## References

- Arda K, Ciledag N, Aktas E, Arıbas BK, Köse K. Quantitative assessment of normal soft-tissue elasticity using shear-wave ultrasound elastography. *AJR Am J Roentgenol* 2011;197(3):532–536. https://doi.org/10.2214/AJR.10.5449
- Bos EJ, Pluemeekers M, Helder M, Kuzmin N, van der Laan K, Groot ML, van Osch G, van Zuijlen P. Structural and mechanical comparison of human ear, alar, and septal cartilage. *Plast Reconstr Surg Glob Open* 2018;6(1):e1610. https://doi.org/10.1097/GOX.0000000000001610
- Brockhoff HC, Morris CD, Throckmorton GS, Finn R. Anatomic analysis of the conchal bowl cartilage. *J Oral Maxillofac Surg* 2014;72(11):2248–2255. https://doi.org/10.1016/j.joms.2014.04.024
- Chi D, Lee S, Kim JH, Kim TK, Jeong JY, Kim S, Oh SH. Surgical importance of the posterior auricular ligament when harvesting ear cartilage in rhinoplasty. *Arch Aesthetic Plast Surg* 2021;27(1):12–17. https://doi.org/10.14730/aaps.2020.02313
- Cornuault PH, Carpentier L, Bueno MA, Cote JM, Monteil G. Influence of physico-chemical, mechanical and morphological fingerpad properties on the frictional distinction of sticky/slippery surfaces. *J R Soc Interface* 2015;12(110):20150495. https://doi.org/10.1098/rsif.2015.0495
- Damron JA. Tensile Material Properties of Human Costal Cartilage Perichondrium. MS thesis, Virginia Tech, 2024. https://hdl.handle.net/10919/119199
- Danter J, Siegert R, Weerda H. Ultrasonographische Haut- und Knorpeldickenmessungen an gesunden und rekonstruierten Ohren mit einem 20-MHz-Ultraschallgerät. *Laryngo-Rhino-Otologie* 1996;75(2):91–94. (Abstract only seen; DOI not verified.)
- Derler S, Gerhardt LC. Tribology of skin: review and analysis of experimental results for the friction coefficient of human skin. *Tribol Lett* 2012;45:1–27. https://doi.org/10.1007/s11249-011-9854-y
- Donofrio CA, Cardona JJ, Riccio L, Chaiyamoon A, Shekhawat D, Iwanaga J, et al. The posterior auricularis muscle as superficial landmark for the sigmoid sinus and transverse-sigmoid sinus junction: an anatomical study. *World Neurosurg* 2024;186:e721–e726. https://doi.org/10.1016/j.wneu.2024.04.040
- Dzidek BM, Adams MJ, Andrews JW, Zhang Z, Johnson SA. Contact mechanics of the human finger pad under compressive loads. *J R Soc Interface* 2017;14(127):20160935. https://doi.org/10.1098/rsif.2016.0935
- Gray H. *Anatomy of the Human Body*, 20th ed., 1918 (auricle, extrinsic ligaments). Reproduced at https://embryology.med.unsw.edu.au/embryology/index.php/File:Gray0906.jpg
- Griffin MF, Premakumar Y, Seifalian AM, Szarko M, Butler PE. Biomechanical characterisation of the human auricular cartilages; implications for tissue engineering. *Ann Biomed Eng* 2016;44(12):3460–3467. https://doi.org/10.1007/s10439-016-1688-1
- Griffin MF, Leung BC, Premakumar Y, Szarko M, Butler PE. Comparison of the mechanical properties of different skin sites for auricular and nasal reconstruction. *J Otolaryngol Head Neck Surg* 2017;46:33. https://doi.org/10.1186/s40463-017-0210-6
- Herman J, Sedlackova Z, Vachutka J, Furst T, Salzman R, Vomacka J. Shear wave elastography parameters of normal soft tissues of the neck. *Biomed Pap Med Fac Univ Palacky Olomouc* 2017;161(3):320–325. https://doi.org/10.5507/bp.2017.024
- Hiramatsu Y, Kimura D, Kadota K, Ito T, Kinoshita H. Control of precision grip force in lifting and holding of low-mass objects. *PLoS One* 2015;10(9):e0138506. https://doi.org/10.1371/journal.pone.0138506
- IMAIOS e-Anatomy. Superior ligament of auricle. https://www.imaios.cn/en/e-anatomy/anatomical-structures/superior-ligament-of-auricle-1159876
- Khan MM, Parab SR. Average thickness of tragal cartilage for slicing techniques in tympanoplasty. *J Laryngol Otol* 2015;129(5):435–439. https://doi.org/10.1017/S0022215115000055
- Khan N, Zaki D, Brown WE, Halaseh FF, Willette D, Ziegler M, Athanasiou KA, Widgerow AD. Tissue engineering auricular cartilage: a review of auricular cartilage characteristics and current techniques for auricular reconstruction. *J Craniofac Surg* 2024;35(3):768–778. https://doi.org/10.1097/SCS.0000000000010015 (Review used for the compiled tensile values of Zahnert, Park, Nayyer and Pappa.)
- Lem M, Pham J, Khan N, Zaki D, Brown W, Hu J, Ziegler M, Athanasiou K, Evans GRD, Widgerow A. Biomechanical characterization of human normal auricular and microtia cartilage. *Plast Reconstr Surg Glob Open* 2022;10(10 Suppl):150 (meeting abstract). https://escholarship.org/uc/item/7wn1c8w9
- Mathiowetz V, Kashman N, Volland G, Weber K, Dowe M, Rogers S. Grip and pinch strength: normative data for adults. *Arch Phys Med Rehabil* 1985;66(2):69–74. PMID 3970660. https://pubmed.ncbi.nlm.nih.gov/3970660/
- Nayyer L, Birchall M, Seifalian AM, Jell G. Design and development of nanocomposite scaffolds for auricular reconstruction. *Nanomedicine* 2014;10(1):235–246. https://doi.org/10.1016/j.nano.2013.06.006
- Nimeskern L, Martínez Ávila H, Sundberg J, Gatenholm P, Müller R, Stok KS. Mechanical evaluation of bacterial nanocellulose as an implant material for ear cartilage replacement. *J Mech Behav Biomed Mater* 2013;22:12–21. https://doi.org/10.1016/j.jmbbm.2013.03.005
- Nimeskern L, Pleumeekers MM, Pawson DJ, Koevoet WL, Lehtoviita I, Soyka MB, Röösli C, Holzmann D, van Osch GJ, Müller R, Stok KS. Mechanical and biochemical mapping of human auricular cartilage for reliable assessment of tissue-engineered constructs. *J Biomech* 2015;48(10):1721–1729. https://doi.org/10.1016/j.jbiomech.2015.05.019
- Öztüran O, Doğan R, Özücer B, Yıldırım YS, Meriç A. Dynamometric analysis of normative auricular stiffness and comparison with operated prominent ears. *J Craniofac Surg* 2015;26(7):2155–2159. https://doi.org/10.1097/SCS.0000000000002062
- Pappa AK, Soleimani S, Caballero M, Halevi AE, van Aalst JA. A pilot study comparing mechanical properties of tissue-engineered cartilages and various endogenous cartilages. *Clin Biomech* 2017;50:105–109. https://doi.org/10.1016/j.clinbiomech.2017.10.010
- Park SS, Jin HR, Chi DH, Taylor RS. Characteristics of tissue-engineered cartilage from human auricular chondrocytes. *Biomaterials* 2004;25(12):2363–2369. https://doi.org/10.1016/j.biomaterials.2003.09.019
- Pawluk DT, Howe RD. Dynamic lumped element response of the human fingerpad. *J Biomech Eng* 1999;121(2):178–183. https://doi.org/10.1115/1.2835100
- Pickrell BB, Hughes CD, Maricevich RS. Partial ear defects. *Semin Plast Surg* 2017;31(3):134–140. https://doi.org/10.1055/s-0037-1603968
- Roy R, Kohles SS, Zaporojan V, Peretti GM, Randolph MA, Xu J, Bonassar LJ. Analysis of bending behavior of native and engineered auricular and costal cartilage. *J Biomed Mater Res A* 2004;68(4):597–602. https://doi.org/10.1002/jbm.a.10068
- Serina ER, Mote CD Jr, Rempel D. Force response of the fingertip pulp to repeated compression—effects of loading rate, loading angle and anthropometry. *J Biomech* 1997;30(10):1035–1040. https://doi.org/10.1016/S0021-9290(97)00065-1
- Serina ER, Mockensturm E, Mote CD Jr, Rempel D. A structural model of the forced compression of the fingertip pulp. *J Biomech* 1998;31(7):639–646. https://doi.org/10.1016/S0021-9290(98)00067-0
- Siegert R, Magritz R. Reconstruction of the auricle. *GMS Curr Top Otorhinolaryngol Head Neck Surg* 2007;6:Doc02. https://pmc.ncbi.nlm.nih.gov/articles/PMC3199840/
- van Kuilenburg J, Masen MA, van der Heide E. Contact modelling of human skin: what value to use for the modulus of elasticity? *Proc IMechE Part J: J Eng Tribol* 2013;227(4):349–361. https://doi.org/10.1177/1350650112463307
- Vernice NA, Dong X, Matavosian AA, Corpuz GS, Shin J, Bonassar LJ, Spector JA. Bioengineering full-scale auricles using 3D-printed external scaffolds and decellularized cartilage xenograft. *Acta Biomater* 2024;179:121–129. https://doi.org/10.1016/j.actbio.2024.03.012
- Zahnert T, Hüttenbrink KB, Mürbe D, Bornitz M. Experimental investigations of the use of cartilage in tympanic membrane reconstruction. *Am J Otol* 2000;21(3):322–328. https://doi.org/10.1016/S0196-0709(00)80039-3
- Zopf DA, Flanagan CL, Nasser HB, Mitsak AG, Huq FS, Rajendran V, Green GE, Hollister SJ. Biomechanical evaluation of human and porcine auricular cartilage. *Laryngoscope* 2015;125(8):E262–E268. https://doi.org/10.1002/lary.25040
- Zwirner J, Ondruschka B, Scholze M, Schulze-Tanzil G, Hammer N. Biomechanical characterization of human temporal muscle fascia in uniaxial tensile tests for graft purposes in duraplasty. *Sci Rep* 2021;11:2127. https://doi.org/10.1038/s41598-020-80448-1
