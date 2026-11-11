# Extrinsic auricular muscles and ligaments: literature for calibration

Purpose: starting values for the invisible, passive, tension-only fibre fans
(auricularis anterior, superior, posterior; anterior / posterior / superior
auricular ligaments) that hold the auricle on the head in the soft-tissue sim.
The clinician tunes from here.

The working model is a **right** ear. Laterality of each source is noted where
the source states it.

Rules used here:
- Every citation below was checked against its PubMed record (authors, year,
  journal, DOI, PMID) or the publisher page, on 2026-11-10.
- Numbers come from the abstract or full text as stated. Where only the abstract
  was available, that is said.
- Anything not from a source is marked **estimate: reason**.

---

## 1. Geometry

### 1a. Classical description (all three muscles, ligaments)

Gray's Anatomy (Gray H, *Anatomy of the Human Body*, 20th ed., 1918), external
ear section, read from a verbatim transcription (hithera.com, Gray's text):

| Structure | Origin | Insertion | Gray's description |
|---|---|---|---|
| Auricularis anterior | Lateral edge of galea aponeurotica | Projection on the front of the helix (spina helicis) | "smallest of the three", thin, fan-shaped, pale, indistinct fibres |
| Auricularis superior | Galea aponeurotica | Upper part of the cranial surface of the auricle, by a thin flat tendon | "largest of the three", thin, fan-shaped |
| Auricularis posterior | Mastoid part of temporal bone, by short aponeurotic fibres | Lower part of the cranial surface of the concha; the **ponticulus** (vertical ridge on the eminentia conchae) "gives attachment to the Auricularis posterior" | 2 or 3 fleshy fascicles |
| Anterior ligament | Tragus and spina helicis | Root of the zygomatic process of the temporal bone | extrinsic |
| Posterior ligament | Posterior (cranial) surface of the concha | Outer surface of the mastoid process | extrinsic |
| (Intrinsic) | Tragus | Start of the helix | "strong fibrous band" |

Gray names only two extrinsic ligaments. A **superior ligament** (bony meatus /
spine of helix) appears in some modern atlases (e.g. IMAIOS e-Anatomy,
Terminologia-based) but I found no measured description of it.

### 1b. Measured geometry

| Structure | Quantity | Value | n / laterality | Source |
|---|---|---|---|---|
| Auricularis superior | Central height (origin to insertion) | **4.7 cm** (mean) | 12 hemifaces: 7 left, 5 right; no L/R difference | Chon 2021 |
| Auricularis superior | Width | **5.0 cm** (mean) | same | Chon 2021 |
| Auricularis superior | Origin / insertion | Fibres of galea **and temporal fascia** -> **conchal cartilage** (every specimen) | same | Chon 2021 |
| Auricularis superior | Function on traction | Pulling along its line slightly elevated the forehead and stopped galea sliding over the vertex: "occipitofrontalis stabilizer and weak brow elevator" | same | Chon 2021 |
| Auricularis posterior | Length | **28.28 mm** (mean) | 18 sides (9 L, 9 R) of 10 heads | Donofrio 2024 |
| Auricularis posterior | Position | First 20 mm lies over the mastoid process; superior border mean 2.22 mm below the transverse-sigmoid sinus junction | same | Donofrio 2024 |
| Auricularis posterior | Width | measured, but not in the abstract (full text paywalled) | | Donofrio 2024 |
| Auricularis posterior | Course | Mastoid -> concha; adducts (pulls back) the pinna; used as the vector for conchomastoid sutures | clinical | Stephen 2017 |
| Auricularis posterior / transversus | Fibre type | ~75 % slow (type I) fibres; look like ordinary skeletal muscle | 24 preserved + 6 fresh cadavers | Bendrihem 2015 (open-access repost; journal volume not confirmed) |
| Posterior auricular ligament (PAL) | Origin / insertion | Deep mastoid fascia -> caudal cymba concha at the eminentia conchae; runs between superficial and deep mastoid fascia, with the posterior auricular muscle | 6 ears, 3 cadavers (no dimensions given) | Chi 2021 |
| Anterior (preauricular) retaining ligament | Presence | Constant cutaneous ligament to the anterior helix in 8 pre-auricular TMJ approaches; separates sub- and supra-zygomatic compartments (no dimensions) | 8 patients | Kudva 2026 |
| Auricularis anterior | Dimensions | **None found** | | gap |
| Auricle to mastoid | Resting distance | 10-12 mm (top), 16-18 mm (middle), 20-22 mm (lobule) | secondary (reference values quoted by Russo 2026, refs 95-96 there) | Russo 2026 |

About the Cleveland Clinic numbers in the brief (central length 4.5 cm, origin
width 4.9 cm, anterior border 4.1 cm, posterior border 4.0 cm): the underlying
paper is almost certainly **Chon et al. 2021** (Cleveland Clinic Cole Eye
Institute; e-pub 30 Nov 2020, hence "2020"). The abstract gives means of 4.7 cm
height and 5.0 cm width. I could not open the full text, so I could **not
verify** the 4.5 / 4.9 / 4.1 / 4.0 values. They may be one specimen in a figure.
The abstract does not say which side its figures show.

Clinical evidence that these structures carry load:
- **Hoogbergen 1996**: a **right** ear that was hypermobile because the superior
  and posterior auricularis muscles were absent (agenesis) and the conchal
  cartilage was thin. Suturing the concha to the mastoid and temporal fascia
  fixed it. So without the muscles, skin and the remaining ligaments do not
  hold the ear well.
- **Yotsuyanagi 2015** (77 ears): an abnormal insertion of the superior
  auricular muscle is the main cause of cryptotia; prominent ear involves the
  transverse and sometimes the posterior auricular muscle.
- **Yamagata 2026**: in cryptotia, moving the superior and anterior auricular
  muscles back to their normal insertions ("behind the conchal bowl" and "the
  base of the helical crus") gave a stable correction.

### 1c. Working geometry for the right-ear fans (estimates where marked)

| Fan | Origin (head) | Insertion (auricle, cranial side) | Length | Width at origin -> insertion | Thickness | CSA |
|---|---|---|---|---|---|---|
| Superior | Galea / TPF above the ear | Cranial surface, upper third: eminentia triangularis / fossa triangularis region (Chon: "conchal cartilage") | 47 mm (Chon) | 50 mm (Chon) -> **~10 mm** (estimate: Gray's "thin flattened tendon"; no measurement found) | **0.5-1 mm** (estimate: described as thin; no measurement found) | **~25-50 mm² at origin** (estimate: 50 mm × 0.5-1 mm) |
| Anterior | Lateral galea / TPF in front of the ear | Spina helicis / root of helical crus | **~20-30 mm** (estimate: smallest muscle; no measurement found) | **~20-30 -> ~5 mm** (estimate) | **≤0.5 mm** (estimate: "pale and indistinct") | **~5-15 mm²** (estimate) |
| Posterior | Mastoid surface | Ponticulus / eminentia conchae | 28 mm (Donofrio) | **~10 -> ~8 mm** (estimate: 2-3 fascicles; width measured in Donofrio but not available) | **~1-2 mm** (estimate: fleshy fascicles) | **~10-20 mm²** (estimate) |
| Posterior ligament | Mastoid (deep mastoid fascia) | Cymba / eminentia conchae (Chi 2021) | **~10-15 mm** (estimate: matches the 10-22 mm auricle-mastoid gap) | **~5-10 mm** (estimate) | — | **~5 mm²** (estimate) |
| Anterior ligament | Root of zygomatic process | Tragus + spina helicis (Gray) | **~5-10 mm** (estimate: short pre-auricular band) | **~5 mm** (estimate) | — | **~3-5 mm²** (estimate) |
| Superior ligament | Bony meatus / squama | Spine of helix (atlases only) | **~5-10 mm** (estimate) | — | — | **small; could be merged into the anterior fan** (estimate) |

Fibre directions (Gray / Chon): superior fibres converge downward from the fan
onto the upper cranial surface. Anterior fibres run backward from the temple to
the helix. Posterior fibres run forward from the mastoid to the concha, roughly
horizontal. Ligaments are short and roughly radial from bone to cartilage.

---

## 2. Passive mechanics of muscle / facial soft tissue

No tensile data exist for the auricular muscles or any human mimetic muscle in
isolation (see Gaps). The closest data:

| Tissue | Test | Modulus | Failure | Notes | Source |
|---|---|---|---|---|---|
| Rabbit EDL muscle, **along fibres** | Uniaxial tension, aponeurosis removed | **447 ± 98 kPa** (linear region) | UTS **163 ± 76 kPa**, failure strain **0.51 ± 0.22** | Transverse: E 22.4 ± 14.7 kPa, UTS 27.5 kPa, strain 1.82; along-fibre shear 3.9 kPa | Morrow 2010 (Table 1, full text) |
| Porcine longissimus, along fibres | Uniaxial tension, optical strain | **~10 kPa stress at λ = 1.1** (nonlinear J-curve, toe region) | Fails at **λ ≈ 1.65** | Cross-fibre is stiffer (77 kPa at λ 1.1) but fails at λ ≈ 1.15; Poisson ν_LT 0.47 | Takaza 2013 |
| Human temporalis muscle | Quasi-static uniaxial, 88 samples | **1.58 ± 0.64 MPa** | UTS **0.26 ± 0.11 MPa**, strain at max force **26 ± 12 %** | No sex / side / post-mortem effect | Zwirner 2020a |
| Human zygomaticus major (a mimetic muscle), in vivo | MR elastography, 90 Hz, n = 3 | Stiffness **6.75 ± 3.36 kPa** (mouth closed), **15.5 ± 5.2 kPa** (mouth open, stretched) | — | Dynamic shear-wave stiffness, not tensile modulus; shows stiffening with stretch | Hojo 2024 |
| Human SMAS (preparotid), ex vivo | Bursting test | — | **Bursting force 55.7 N** (224.97 psi) | In-vivo SMAS fixation tension **7.21 ± 0.87 N** (pre-auricular) and **9.59 ± 1.8 N** (mastoid); creep only 14 % at that load | Trussler 2011 |
| Human SMAS | Instron loading / unloading, stress relaxation | not in abstract | — | Viscoelastic, similar to pre-auricular skin; stiffens and loses viscoelasticity with repeated loading; convoluted collagen + many elastic fibres | Har-Shai 1996; Har-Shai 1997 |
| Human SMAS, upper vs lower | 64 samples, 8 heads | not in abstract | — | Upper (near the ear) has higher modulus and UTS; lower stretches more before break | Hu 2014 |
| Human SMAS, suture tearing | Instron, 12 heads | not in abstract | — | Breaking strength and suture tearing force reported; mesh reinforcement raises both | Angelos 2014 |

Suggested fan law (**estimate: built from the table above, not measured on
auricular muscle**):
- Tension-only, exponential toe (J-curve), as in Takaza and Raposio.
- Toe region up to about 10 % strain. Linear modulus along fibres
  **0.3-0.5 MPa** (Morrow-like vestigial muscle) up to **1.6 MPa** (if
  perimysium and epimysium dominate, as in Zwirner's whole-muscle samples).
- Failure about 25-50 % strain, 0.15-0.3 MPa.
- Resulting linear stiffness k = E·A/L (estimate):
  - superior: 0.45 MPa × 37 mm² / 47 mm ≈ **0.35 N/mm**
  - posterior: 0.45 MPa × 15 mm² / 28 mm ≈ **0.24 N/mm**
  - anterior: ≈ **0.1 N/mm**
  - Multiply by about 3.5 for the 1.6 MPa upper bound.
- Ligaments: use the fascia values in section 4 (E about 25-36 MPa,
  failure about 14-17 % strain) on a small CSA.

---

## 3. Whole-auricle mechanics

| Measurement | Value | Method | Source |
|---|---|---|---|
| **Recoil force, auricle folded forward** (pulled anteriorly at the postaurale) | **84.0 ± 10.2 N** (range 60-100 N) intact | Tension gauge on a 3-0 nylon suture at the postaurale, 30 patients, supine, intra-op (rhinoplasty cartilage harvest); force read as the folded ear starts to move back | Chi 2021 |
| … after detaching the **posterior auricular muscle** | **53.4 ± 9.2 N** (−30.6 N, ≈36 %) | same | Chi 2021 |
| … after cutting the **posterior auricular ligament** | **39.8 ± 7.3 N** (−13.6 N, 16.2 % of total) | same | Chi 2021 |
| … after harvesting cymba concha cartilage | **28.6 ± 6.5 N** (−11.3 N, 13.4 %) | same | Chi 2021 |
| Auricle stiffness: force to push the antihelix medially 1, 2, 3, 4 mm, and peak force to within 1 mm of mastoid skin | Values in full text only. Stiffness rises with age (above 35 and 50 years) and with auricle projection; no sex effect | Digital Newton meter, 226 ears, ages 2-85 | Buyuklu 2025 |
| Dynamometric antihelix stiffness at 4 levels | Values in full text only. Peaks at 35-39 years, falls after 40; prominent ears are softer before surgery | Computer dynamometry, 190 volunteers + 28 operated ears | Ozturan 2015 |
| Otoplasty FE model | Mustardé sutures bend the upper third most; conchal setback lays the whole ear down with peak stress at the helix root; scaphal incision lowers Mustardé peak stress to 82 % | FE on 8 digitized prominent-ear cartilages (loads not given in abstract) | Miyamoto 2009 |
| Elastic recoil after otoplasty | Up to **~40 %** loss of intra-op correction | Technique chapter (secondary, not peer-reviewed numbers) | Vula / UCT otoplasty technique PDF; treat as estimate |

How to use this (**estimate: interpretation**):
- Chi 2021 is the only whole-ear force-vs-structure breakdown found. It splits
  the hold as: posterior muscle ≈ 36 %, posterior ligament ≈ 16 %, cymba
  cartilage ≈ 13 %, and the rest (≈ 34 %, ~29 N) is skin, the remaining
  cartilage, the anterior / superior structures and the sulcus.
- The displacement is not reported. The ear is fully folded forward, so this is
  a large-strain, near-end-of-range number. It is not stiffness at small pulls.
  Take 84 N as an upper bound for an extreme forward fold. It seems high for
  manual handling; check with the clinician.
- For everyday traction (otoscopy: pinna up and back, a few mm to ~10 mm) no
  measurement was found. **Estimate: 1-5 N for 5-10 mm**, from the SMAS fixation
  tensions (7-10 N, Trussler 2011) being "firm" surgical loads, and the fan
  stiffness above (~0.3-1 N/mm summed).

---

## 4. Temporoparietal fascia (TPF) and galea

Anatomy (Abul-Hassan 1986, 15 fresh cadavers / 30 sides): the superficial
temporal fascia (TPF) lies just deep to the hair follicles. It is part of the
SMAS layer and continuous with the galea above and the facial SMAS below. A
loose areolar plane separates it from the deep temporal fascia, which invests
temporalis and attaches to the zygomatic arch. Yi 2026 (60 embalmed hemisides):
a temporoparietalis muscle inside the TPF was present on 61.7 % of sides and
thick on 16.7 %.

No uniaxial test of isolated TPF or isolated galea was found. Nearest data:

| Tissue | Modulus | UTS | Strain | Source |
|---|---|---|---|---|
| Human **deep** temporal (temporalis muscle) fascia, 74 samples | **36 ± 19 MPa** | **3.6 ± 1.7 MPa**; max force 16 ± 8 N | max strain 13 ± 4 %; failure 17 ± 6 % | Zwirner 2021 |
| Human deep temporal fascia, 52 samples (median) | **26.2 MPa** | 2.9 MPa; max force 12.6 N | failure 14.1 % | Zwirner 2020b |
| Human native scalp (full thickness, temporal), Zwirner series | **22.5 ± 5.1 MPa** | **3.2 ± 1.0 MPa** | strain at max force 18.3 ± 3.4 % | Zwirner 2020c (native-scalp control group) |
| Human scalp flaps through the galea, in vivo intra-op | "E" = **49.2 g/mm** (≈0.48 N/mm, a flap stiffness, not a modulus); three phases: linear 0-500 g (0-4.9 N), stiffening 500-1500 g, exponential 1500-5000 g (to 49 N) | — | — | Raposio 1997 |
| Same, mean-curve fit | **117.1 g/mm** (≈1.15 N/mm) | — | — | Raposio 1998a |
| Effect of galeotomies | Each full-thickness galeotomy dropped stiffness by 16.6 g/mm (≈40 % less closing tension); +1.67 mm length per galeotomy at 500-1500 g | — | — | Raposio 1998b |
| Human scalp, regional | E, UTS, strains by region and sex (values not in abstract) | | | Falland-Cheung 2018 |

Notes:
- Deep temporal fascia is **not** the TPF. It is a dense investing fascia and an
  upper bound. **Estimate: TPF/galea ≈ 10-30 MPa along the collagen, ~1.5-3 MPa
  UTS, toe to ~5-10 % strain.** Reason: the galea is aponeurotic (like deep
  fascia); the TPF is a looser SMAS layer (like SMAS / skin).
- TPF thickness: **estimate 2-3 mm**. Reason: commonly quoted in flap
  literature, but I did not find a primary measurement. The scalp overall is
  ~8 mm in adults (Trotta 2019 intro, citing earlier work).
- For the sim, the fan origins can be fixed to the skull. The TPF/galea is
  10-100× stiffer than the muscle, so it is effectively rigid at these loads
  (**estimate: from moduli above**).

---

## Gaps

1. No dimensions for the **auricularis anterior** (length, width, thickness).
2. No thickness or insertion width for any auricular muscle. The superior and
   posterior values in 1c are estimates. The Donofrio 2024 posterior width and
   the Chon 2021 border lengths are in paywalled full texts.
3. The 4.5 / 4.9 / 4.1 / 4.0 cm Cleveland Clinic figures are not verified
   (abstract gives 4.7 cm × 5.0 cm).
4. No dimensions for the anterior, posterior or superior auricular ligaments.
   Only their presence and attachments are known (Gray, Chi 2021, Kudva 2026).
5. No tensile data for any human mimetic or auricular muscle, epimysium, isolated
   TPF or galea. The values used are from rabbit / porcine muscle, human
   temporalis, deep temporal fascia, SMAS and scalp.
6. No force-displacement curve for pulling the whole auricle. Chi 2021 gives
   forces at full forward fold with no displacement. Buyuklu 2025 and
   Ozturan 2015 have medial push forces at 1-4 mm, but only in the full text.
7. No otoscopy pinna-traction force, and no measured Mustardé / Furnas suture
   tension (the Miyamoto FE loads are not in the abstract).
8. Laterality: Chi 2021 and Chon 2021 figures do not state the side. Hoogbergen
   1996 is a right ear.

---

## References

- Abul-Hassan HS, von Drasek Ascher G, Acland RD. Surgical anatomy and blood supply of the fascial layers of the temporal region. *Plast Reconstr Surg.* 1986;77(1):17-28. PMID 3941846.
- Angelos PC, Brennan TE, Toriumi DM. Biomechanical properties of superficial musculoaponeurotic system tissue with vs without reinforcement with poly-4-hydroxybutyric acid absorbable mesh. *JAMA Facial Plast Surg.* 2014;16(3):199-205. doi:10.1001/jamafacial.2013.2738. PMID 24676588.
- Bendrihem R, Vacher C, Barbet JP. The posterior muscles of the auricle: anatomy and surgical applications. 2015 (JSciMed Central otolaryngology; open-access repost read, volume not confirmed).
- Buyuklu F, Cakmak MP, Cakmak BA, Durukan E, Hizal E. Age and gender related variability in the strength of auricular cartilage in living humans. *Aesthetic Plast Surg.* 2025;49(3):643-654. doi:10.1007/s00266-024-04481-5. PMID 39557640.
- Chi D, Lee S, Kim JH, Kim TK, Jeong JY, Kim S, Oh SH. Surgical importance of the posterior auricular ligament when harvesting ear cartilage in rhinoplasty. *Arch Aesthetic Plast Surg.* 2021;27(1):12-17. doi:10.14730/aaps.2020.02313.
- Chon BH, Blandford AD, Hwang CJ, Petkovsek D, Zheng A, Zhao C, Cao J, Grissom N, Perry JD. Dimensions, function and applications of the auricular muscle in facial plastic surgery. *Aesthetic Plast Surg.* 2021;45(1):309-314 (e-pub 30 Nov 2020). doi:10.1007/s00266-020-02045-x. PMID 33258010.
- Donofrio CA, Cardona JJ, Riccio L, et al. The posterior auricularis muscle as superficial landmark for the sigmoid sinus and transverse-sigmoid sinus junction: an anatomical study. *World Neurosurg.* 2024;186:e721-e726. doi:10.1016/j.wneu.2024.04.040. PMID 38616028.
- Falland-Cheung L, Scholze M, Lozano PF, et al. Mechanical properties of the human scalp in tension. *J Mech Behav Biomed Mater.* 2018;84:188-197. doi:10.1016/j.jmbbm.2018.05.024. PMID 29793156.
- Gray H. *Anatomy of the Human Body.* 20th ed. Philadelphia: Lea & Febiger; 1918. External ear (auricula: ligaments and muscles).
- Har-Shai Y, Bodner SR, Egozy-Golan D, et al. Mechanical properties and microstructure of the superficial musculoaponeurotic system. *Plast Reconstr Surg.* 1996;98(1):59-70. doi:10.1097/00006534-199607000-00009. PMID 8657788.
- Har-Shai Y, Bodner SR, Egozy-Golan D, et al. Viscoelastic properties of the superficial musculoaponeurotic system (SMAS): a microscopic and mechanical study. *Aesthetic Plast Surg.* 1997;21(4):219-224. doi:10.1007/s002669900113. PMID 9263540.
- Hojo E, Glaser KJ, Harada-Hayashi A, et al. Feasibility study of the application of magnetic resonance elastography (MRE) to measure oral posture related changes in stiffness of zygomaticus major muscle. *Magn Reson Imaging.* 2024;111:196-201. doi:10.1016/j.mri.2024.05.005. PMID 38723783.
- Hoogbergen MM, Schuurman AH, Rijnders W, Kon M. Auricular hypermobility due to agenesis of the extrinsic muscles. *Plast Reconstr Surg.* 1996;98(5):869-871. doi:10.1097/00006534-199610000-00021. PMID 8823029.
- Hu X, Wang Z, Wang Q, Zhang C, Hu G, Qin H. Are there differences between the upper and lower parts of the superficial musculoaponeurotic system? A preliminary biomechanical study. *Aesthet Surg J.* 2014;34(5):661-667. doi:10.1177/1090820X14528947. PMID 24744395.
- Kudva A, Suryanarayan RK, Roy S, Liu S, Shenoy P. Preauricular retaining ligament to anterior helix of the ear: is it a constant structure? *J Maxillofac Oral Surg.* 2026;25(1):285-287. doi:10.1007/s12663-025-02491-7. PMID 41648701.
- Miyamoto J, Nagasao T, Tamaki T, Nakajima T. Biomechanical evaluation of surgical correction of prominent ear. *Plast Reconstr Surg.* 2009;123(3):889-896. doi:10.1097/PRS.0b013e318199f098. PMID 19319052.
- Morrow DA, Haut Donahue TL, Odegard GM, Kaufman KR. Transversely isotropic tensile material properties of skeletal muscle tissue. *J Mech Behav Biomed Mater.* 2010;3(1):124-129. doi:10.1016/j.jmbbm.2009.03.004. PMID 19878911. PMC2772832.
- Ozturan O, Dogan R, Ozucer B, Yildirim YS, Meric A. Dynamometric analysis of normative auricular stiffness and comparison with operated prominent ears. *J Craniofac Surg.* 2015;26(7):2155-2159. doi:10.1097/SCS.0000000000002062. PMID 26468801.
- Raposio E, Nordström RE. Tension and flap advancement in the human scalp. *Ann Plast Surg.* 1997;39(1):20-23. doi:10.1097/00000637-199707000-00003. PMID 9229087.
- Raposio E, Nordström RE. Biomechanical properties of scalp flaps and their correlations to reconstructive and aesthetic surgery procedures. *Skin Res Technol.* 1998;4(2):94-98. doi:10.1111/j.1600-0846.1998.tb00093.x. PMID 27332106. (1998a)
- Raposio E, Santi P, Nordström RE. Effects of galeotomies on scalp flaps. *Ann Plast Surg.* 1998;41(1):17-21. doi:10.1097/00000637-199807000-00004. PMID 9678463. (1998b)
- Russo PR, Sbarbati A, Bartoletti E, et al. Non-surgical ear rejuvenation through selective neuromodulation with onabotulinum and abobotulinum toxin-A. *Toxins.* 2026;18:366. PMID 42784290. PMC13611376. (Secondary source for auricle-mastoid distances.)
- Stephen C, Lowrie AG. The posterior auricular muscle: a useful anatomical landmark for otoplasty. *J Laryngol Otol.* 2017;131(5):465-467. doi:10.1017/S0022215117000500. PMID 28264728.
- Takaza M, Moerman KM, Gindre J, Lyons G, Simms CK. The anisotropic mechanical behaviour of passive skeletal muscle tissue subjected to large tensile strain. *J Mech Behav Biomed Mater.* 2013;17:209-220. doi:10.1016/j.jmbbm.2012.09.001. PMID 23127635.
- Trotta A, Ní Annaidh A. Mechanical characterisation of human and porcine scalp tissue at dynamic strain rates. *J Mech Behav Biomed Mater.* 2019;100:103381. doi:10.1016/j.jmbbm.2019.103381. PMID 31430703.
- Trussler AP, Hatef D, Broussard GB, Brown S, Barton FE. The viscoelastic properties of the SMAS and its clinical translation: firm support for the high-SMAS rhytidectomy. *Plast Reconstr Surg.* 2011;128(3):757-764. doi:10.1097/PRS.0b013e3182221354. PMID 21572376.
- Yamagata K, Narushima M, Ishiura R, et al. Treatment of mild cryptotia by anatomical external auricular muscle transposition without flap transfer: a case report. *JPRAS Open.* 2026;51:697-702. doi:10.1016/j.jpra.2026.07.015. PMID 42633401.
- Yi KH, Wong IKJ, Park HJ, Rosellini I, Corrêa MC, Swift A. Temporoparietalis muscle: a cadaveric study of 60 hemisides. *Plast Reconstr Surg Glob Open.* 2026;14(6):e7887. doi:10.1097/GOX.0000000000007887. PMID 42325484.
- Yotsuyanagi T, Yamauchi M, Yamashita K, et al. Abnormality of auricular muscles in congenital auricular deformities. *Plast Reconstr Surg.* 2015;136(1):78e-88e. doi:10.1097/PRS.0000000000001383. PMID 26111335.
- Zwirner J, Ondruschka B, Scholze M, Hammer N. Passive load-deformation properties of human temporal muscle. *J Biomech.* 2020;106:109829. doi:10.1016/j.jbiomech.2020.109829. PMID 32517987. (2020a)
- Zwirner J, Ondruschka B, Scholze M, Schulze-Tanzil G, Hammer N. Mechanical properties of native and acellular temporal muscle fascia for surgical reconstruction and computational modelling purposes. *J Mech Behav Biomed Mater.* 2020;108:103833. doi:10.1016/j.jmbbm.2020.103833. PMID 32469726. (2020b)
- Zwirner J, Ondruschka B, Scholze M, Schulze-Tanzil G, Hammer N. Load-deformation characteristics of acellular human scalp: assessing tissue grafts from a material testing perspective. *Sci Rep.* 2020;10:19243. doi:10.1038/s41598-020-75875-z. PMID 33159106. (2020c; native-scalp values from full text)
- Zwirner J, Ondruschka B, Scholze M, Schulze-Tanzil G, Hammer N. Biomechanical characterization of human temporal muscle fascia in uniaxial tensile tests for graft purposes in duraplasty. *Sci Rep.* 2021;11:2127. doi:10.1038/s41598-020-80448-1. PMID 33483525.
