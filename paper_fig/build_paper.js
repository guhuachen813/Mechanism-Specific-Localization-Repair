// Build paper draft v0.2 .docx (IEEE-style two-column, Letter) — Simple-ViLMedSAM-style architecture
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
  SectionType, Table, TableCell, TableRow, WidthType, BorderStyle, ShadingType,
  LevelFormat,
} = require("docx");

const FIG = p => path.join(__dirname, p);
const TNR = "Times New Roman";

// ---------- helpers ----------
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: TNR, ...opts }));
    const seg = m[0];
    if (seg.startsWith("**")) out.push(new TextRun({ text: seg.slice(2, -2), bold: true, font: TNR, ...opts }));
    else out.push(new TextRun({ text: seg.slice(1, -1), italics: true, font: TNR, ...opts }));
    last = m.index + seg.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: TNR, ...opts }));
  return out;
}
const p = (t, o = {}) => new Paragraph({
  children: runs(t, { size: o.size ?? 20, ...o.run }),
  alignment: AlignmentType.JUSTIFIED,
  indent: o.noindent ? undefined : { firstLine: 200 },
  spacing: { after: 40, line: 240 },
  ...(o.para || {}),
});
const h1 = t => new Paragraph({
  children: runs(t.toUpperCase(), { bold: true, size: 21 }),
  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 },
});
const h2 = t => new Paragraph({
  children: runs(t, { bold: true, italics: true, size: 20 }),
  alignment: AlignmentType.LEFT, spacing: { before: 120, after: 60 },
});
const caption = (t, o = {}) => new Paragraph({
  children: runs(t, { size: 17 }),
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 120 },
  ...o,
});
const fig = (file, w, h, cap) => [
  new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(file)), transformation: { width: w, height: h } })],
    alignment: AlignmentType.CENTER, spacing: { before: 80 },
  }),
  caption(cap),
];
const ref = (t, n) => new Paragraph({
  children: runs(`[${n}] ` + t, { size: 17 }),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 },
});
const contrib = t => new Paragraph({
  children: runs(t, { size: 20 }),
  numbering: { reference: "contrib", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 },
});
const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "444444" };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };
function cell(text, { w, bold = false, fill = null, align = AlignmentType.LEFT, size = 15 } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    children: [new Paragraph({
      children: runs(text, { bold, size }),
      alignment: align, spacing: { after: 0 },
    })],
  });
}
function table(widths, rows, headerFill = "E8EEF7") {
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => cell(c, {
        w: widths[j], bold: i === 0, fill: i === 0 ? headerFill : null,
        align: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      })),
    })),
  });
}
const tcap = t => new Paragraph({
  children: runs(`**${t[0]}**` + t.slice(1), { size: 17 }),
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 100, after: 40 },
});

// ---------- content ----------
const C = [];

// Abstract (~150 words)
C.push(new Paragraph({
  children: runs(`*Abstract*—Weakly supervised localization (CAM) fails under real-world photographic capture shift in low-resource chest radiography settings. We reveal two critical gaps: (1) synthetic degradation overestimates localization robustness by 4–6×; (2) medical fine-tuning trades robustness for in-domain performance (MedSAM degrades 2.3× worse than general SAM under shift). Decomposing failure into readout-level and geometry-level mechanisms, we propose **PAIR-Loc** (paired distillation for readout failure) and **RectNet** (homography rectification for geometric failure), combined by disease-specific routing. On 229 phone-captured chest X-ray evaluations (123 patients), the routed system gains **+0.15–0.20 IoU@0.45** (p < 1e-4), with wrong-repair point estimates of 2.0–10.8% and exact upper bounds of 9.1–21%. To our knowledge, this is the first effective localization repair demonstrated in the real photography domain, established under a pre-registered audit that retracted six of our own initial claims.`, { size: 19 }),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
}));
C.push(new Paragraph({
  children: runs(`***Keywords*—*weakly supervised localization; chest radiographs; photographic domain shift; knowledge distillation; homography rectification; robustness**`, { size: 19 }),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
}));

// ============ 1. INTRODUCTION ============
C.push(h1("1. Introduction"));
C.push(p(`In point-of-care and low-resource settings, chest radiographs are increasingly consumed as *photographs*: a digital image displayed on a monitor or printed on film is re-captured with a smartphone and transmitted for interpretation. Weakly supervised pipelines built on class-activation maps (CAMs) [1] are attractive here because they localize disease from image-level labels alone, and large chest X-ray datasets [3] made such pipelines standard. However, while classification under photographic shift has been studied [10], [11], we showed in a systematic benchmark that *localization* fails disproportionately: under a pure contrast degradation, 94 of 146 test images lose accurate localization while the classification probability *rises* (0.795→0.996) — the model still answers correctly, but points wrongly. A deployed CAM that quietly points at the wrong lung region is worse than no CAM at all.`));
C.push(p(`Two findings from our audits shape this paper. **First, synthetic degradations drastically overestimate localization robustness.** On 202 paired CheXphotograph studies, the agreement between shifted and clean CAMs (agree-IoU at τ = 0.45) remains 0.640 under synthetic digital transforms and 0.474 under synthetic photographic transforms — but collapses to 0.106 under genuine re-photography, a 4–6× gap, with the CAM centroid drifting 61.5 px. Qualitatively, raw CAMs under real re-photography lock onto screen borders and bright windows: context leakage that no synthetic transform reproduces. **Second, medical fine-tuning carries a robustness cost.** With frozen encoders and identical probes, the general-purpose SAM loses 0.078 AUROC from clean to real re-photographs, while the medically fine-tuned MedSAM loses 0.176 and falls to 0.553 — near chance. The default assumption that adapting to medical data always helps is reversed at the capture-physics boundary.`));
C.push(p(`These findings imply that repair must be mechanism-specific. We decompose real-photograph localization failure into two families: *readout failure*, where photographic texture distracts the classifier's readout while the capture geometry is intact — repairable by distilling the model's own clean-domain CAM, propagated into the photographic frame through a registration homography; and *geometric failure*, where genuine perspective and scale distortion of the photographed film must be corrected in image space. We instantiate the first as **PAIR-Loc**, a paired localization distillation into a zero-initialized low-rank adapter on a frozen backbone, and the second as **RectNet**, a photo-only corner-regression and homography rectification. A *fixed per-finding routing* — derived from development experiments, not learned at inference — assigns each finding to the action its mechanism calls for.`));
C.push(p(`Our contributions:`));
C.push(contrib(`We quantify the gap between synthetic and real capture shift at the localization layer (4–6× overestimate of stability, with a context-leakage mechanism), showing that synthetic-robustness results do not transfer to real re-photography.`));
C.push(contrib(`We reveal a robustness cost of medical fine-tuning (SAM −0.078 vs. MedSAM −0.176 AUROC under real re-photography), challenging the default "always fine-tune on medical data."`));
C.push(contrib(`We propose mechanism-specific repair — PAIR-Loc for readout failure, RectNet for geometric failure — with fixed per-finding routing achieving +0.15–0.20 IoU@0.45 (p < 1e-4) in the real re-photography domain, with wrong-repair point estimates of 2.0–10.8% (pooled 6.6%).`));
C.push(contrib(`We demonstrate a pre-registered audit protocol that retracted six of our own initial claims, establishing statistical discipline for shift-robustness studies.`));

// ============ 2. RELATED WORK ============
C.push(h1("2. Related Work"));
C.push(h2("2.1. Weakly supervised localization under distribution shift"));
C.push(p(`CAM [1] and Grad-CAM [2] established that classification supervision alone yields localizing explanations, and ChestX-ray14 [3], CheXpert [9] and the expert-mask benchmark CheXlocalize [8] made thoracic localization measurable at scale. Beyond plain CAM, methods exploit limited box annotations [4], [5], anatomy-guided attention [6], or reframe WSOL as domain adaptation [7]. Evaluation under shift, however, is almost exclusively synthetic: CheXphoto [10] and CheXphotogenic [11] established that photographic shift degrades *classification*, and robustness studies routinely rely on synthetic photographic transforms. **What is missing is the real re-photography axis at the localization layer**: our paired-agreement protocol shows synthetic conditions overestimate localization stability by 4–6×, and no prior work repairs localization against genuinely re-photographed radiographs with paired statistics.`));
C.push(h2("2.2. Medical foundation models and domain adaptation"));
C.push(p(`Vision foundation models (SAM [15]) and their medical adaptations (MedSAM [16], BioViL [17]) dominate current medical imaging pipelines, and test-time adaptation (TENT [18]) is the standard answer to shift. Hospital-shift confounds [12], hidden stratification [13] and shortcut learning [14] are documented for classification. **These works assume adaptation to medical data is beneficial**; our frozen-probe comparison finds the opposite at the capture boundary — medical fine-tuning spent robustness that the general model possessed (−0.176 vs. −0.078 AUROC) — and TTA-style feature adaptation, in our audits, recovered gains an order of magnitude below the mechanism-targeted repairs proposed here.`));
C.push(h2("2.3. CAM refinement and geometric rectification"));
C.push(p(`On the readout side, distillation [19] and low-rank adapters [20] are standard adaptation machinery; we combine them with a *registration-aware CAM target* so that the adapter is supervised to localize in the photographed geometry. On the geometric side, classical keypoint registration [21] and deep homography regression [22] are mature, and PTRN [23] — the closest prior work — predicts the projective transform of photographed chest X-rays to improve *classification*, trained on synthetic captures. **PTRN does not address localization, uses no paired clean–photo supervision, and reports no mechanism analysis or failure bound.** We show that geometric rectification helps exactly one mechanism family (and one finding type), harms the synthetic domain, and must be routed per finding — an analysis geometric-rectification pipelines alone cannot make.`));

C.push(h2("2.4. Motivating evidence: two systematic gaps"));
C.push(p(`Before presenting the method, we establish the two systematic gaps that motivate mechanism-specific repair, using a paired protocol on 202 CheXphoto studies (same image, clean digital vs. shifted; patient-level statistics). **First, synthetic degradations drastically overestimate localization robustness** (Figure 1). CAM agree-IoU (τ = 0.45) holds at 0.640 under synthetic digital transforms and 0.474 under synthetic photographic transforms, but collapses to 0.106 under genuine re-photography — a 4–6× gap — with CAM centroids drifting 61.5 px instead of 24.8 px. Raw CAMs under real re-photography lock onto screen borders and bright windows: context leakage that no synthetic transform reproduces.`));
C.push(...fig("fig4_gap.png", 240, 167,
  `**Figure 1.** Synthetic vs. real capture shift at the localization layer (202 paired studies, agree-IoU at τ = 0.45 with the clean CAM). Synthetic conditions overestimate localization stability 4–6×; centroid drift more than doubles under real re-photography.`));
C.push(p(`**Second, medical fine-tuning carries a robustness cost** (Figure 2). With frozen encoders and identical linear probes, the general-purpose SAM loses 0.078 AUROC from clean to real re-photographs, while the medically fine-tuned MedSAM loses 0.176 — 2.3× more — and falls to 0.553, near chance. The default assumption that adapting to medical data always helps reverses at the capture boundary. The two gaps jointly imply that repair must target specific failure mechanisms rather than apply universal domain adaptation — and that the evidence guiding repair must come from real re-photography.`));
C.push(...fig("fig5_finetune.png", 240, 167,
  `**Figure 2.** Robustness cost of medical fine-tuning: Cardiomegaly AUROC, clean digital vs. real re-photograph (202 paired studies, patient-level bootstrap). * = paired 95% CI excludes zero. The medically fine-tuned model degrades most and lands near chance.`));

// ============ 3. METHODOLOGY ============
C.push(h1("3. Methodology"));
C.push(h2("3.1. Problem formulation and mechanism decomposition"));
C.push(p(`Let a classifier f with a CAM readout produce, for finding c, a heat map M = CAM(f, c) over a 224 px image. Localization quality is the IoU between the thresholded CAM (τ = 0.45) and a reference mask; under photographic shift we additionally measure *stability* as agree-IoU between the shifted and clean CAMs. An image is a *wrong repair* if a repair changes its IoU@0.45 by more than −0.05.`));
C.push(p(`Given a clean–photo pair with registration homography H (SIFT + RANSAC; Sec. 4.1), we isolate each mechanism with an oracle probe. The *readout oracle* warps the clean model's own CAM into the photographic frame (M̃ = W_H(M_clean)), assuming the geometry is fine and only the readout drifted. The *geometry oracle* rectifies the photograph with H and computes the CAM on the rectified image. The two probes dissociate sharply: on Cardiomegaly the readout oracle recovers +0.139 IoU@0.45 while rectification adds little; on Atelectasis the pattern inverts (+0.005 vs. +0.140). Real capture shift therefore causes *mechanistically heterogeneous* failure. Table 1 extends the probes to four findings together with the two deployable actions and the per-image deployment oracle (the ceiling of any fixed routing); GT-rectification combined ceilings, available for Cardiomegaly and Atelectasis, reach +0.184 and +0.172 respectively. Figure 3 shows the resulting pipeline: one frozen backbone, two mechanism-matched repair branches, and a fixed routing table.`));
C.push(tcap(`Table 1. Mechanism evidence across four findings (ΔIoU@0.45 vs. raw on real re-photographs). "Readout oracle" propagates the clean-teacher CAM through the GT registration homography; PAIR-Loc and RectNet are the deployable actions (predicted homography; the GT-rectification oracle, available for Cardiomegaly/Atelectasis, gives the ceilings in the text); "deployment oracle" takes the per-image best of the two actions — the ceiling of any fixed per-finding routing without gates.`));
C.push(table(
  [880, 480, 880, 760, 760, 1000],
  [
    ["Finding", "n", "Readout oracle", "PAIR-Loc", "RectNet", "Deployment oracle"],
    ["Cardiomegaly", "65", "+0.139", "+0.121", "+0.005", "+0.162"],
    ["Atelectasis", "73", "+0.005", "+0.014", "+0.202", "+0.208"],
    ["Effusion", "50", "+0.218", "+0.147", "+0.021", "+0.165"],
    ["Edema", "41", "+0.137", "+0.038", "+0.076", "+0.145"],
  ]));
C.push(caption(`Readout-dominant findings (Cardiomegaly, Effusion) have high readout-oracle ceilings that PAIR-Loc largely converts into deployed gains; Atelectasis is geometry-dominant (readout oracle ≈ 0, RectNet +0.202); Edema is mixed — both actions positive on average but neither dominant, and the per-image best action scatters (the deployment oracle exceeds each fixed action by a wide margin). This disease-specific pattern motivates the two-branch design (Figure 3) and the fixed per-finding routing (Sec. 3.4).`));
C.push(...fig("fig1_pipeline.png", 300, 150,
  `**Figure 3.** Pipeline overview and mechanism decomposition. A frozen DenseNet-121 backbone feeds two mechanism-matched repair branches: PAIR-Loc (Sec. 3.2) for readout-level failures (Cardiomegaly, Effusion) and RectNet (Sec. 3.3) for geometric failure (Atelectasis). Fixed per-finding routing (Sec. 3.4), derived from the Table 1 oracle evidence, assigns each finding to its matched branch or to the raw fallback. No learned failure-mode classifier is used; routing is experimentally identified and frozen at deployment.`));
C.push(h2("3.2. PAIR-Loc: paired localization distillation"));
C.push(p(`*Design motivation.* Contrast- and texture-driven degradations corrupt the CAM readout while leaving the anatomical geometry intact; the frozen features still support localization, so a small readout-side adaptation suffices. PAIR-Loc inserts a zero-initialized low-rank adapter (1024→16→1024, LoRA-style [20]) on the frozen DenseNet-121 [24] feature maps — at initialization the system is exactly the raw model — and trains it *only on paired clean–photo data*. The supervision is the clean model's own CAM warped into the photographic frame by the pair's registration homography: a *registration-aware CAM target* that requires no extra annotation.`));
C.push(p(`The adapter minimizes a five-term objective:`));
C.push(new Paragraph({
  children: runs(`L = L(logit) + λ(cam) L(cam; M̃) + λ(geo) L(equiv) + λ(anat) L(anat) + λ(ident) L(ident),`,
    { italics: true, size: 20 }),
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
}));
C.push(p(`where L(logit) distills the teacher's clean-image class probability, L(cam) regresses the photographic CAM toward the propagated target M̃, L(equiv) enforces cross-pair geometric consistency of feature maps, L(anat) penalizes activations outside a coarse anatomical field, and L(ident) is a clean-image consistency term. At deployment the adapter sees only the photograph. Our deployment configuration is *zero-contact*: a fixed 40-epoch schedule with no early stopping and no model selection on evaluation data; variants that used early-stop selection are reported only as sensitivity rows (Sec. 4.4).`));
C.push(h2("3.3. RectNet: photo-only geometric rectification"));
C.push(p(`*Design motivation.* Re-photographing a film from an off-axis position introduces real perspective distortion: the CAM's spatial coordinates are simply wrong, and no readout adaptation can recover what the geometry changed. RectNet regresses the four image corners (8 offsets) that map the photographed frame back to the frontal X-ray plane, from frozen GAP-pooled backbone features through a lightweight MLP head. The head is trained on the SIFT ground-truth homographies of the paired data (median corner error 3.5 px on Atelectasis pairs); at inference it needs no clean–photo pairing and no GT registration. The predicted homography rectifies the photograph, the CAM is computed on the rectified image, and warped back to the original frame for evaluation. As Sec. 4.4 shows, this physically motivated module is *not* universal: on Cardiomegaly its corner error is 19 px and in the synthetic domain it is actively harmful.`));
C.push(h2("3.4. Fixed per-finding routing"));
C.push(p(`The mechanism probes extend to four findings (Figure 4): Effusion reproduces the readout-dominant pattern (clean-teacher ceiling +0.218, the highest of the four), Atelectasis is geometry-dominant, and Edema is genuinely mixed — both probes positive, neither dominant, and the per-image best action scattered. We therefore deploy a *fixed routing table* derived from these development experiments: Cardiomegaly and Effusion → PAIR-Loc; Atelectasis → RectNet; all other findings → raw fallback. Edema is analyzed as a mixed-mechanism stress test rather than routed. We emphasize that this is an experimentally identified assignment; no component diagnoses the failure mechanism at inference time, and the framework figure's mechanism decomposition is an analysis lens, not a trained router.`));
C.push(...fig("fig3_mechanism_evidence.png", 300, 135,
  `**Figure 4.** Mechanism evidence per finding: ΔIoU@0.45 of the readout oracle (clean-teacher propagation), PAIR-Loc (readout repair) and RectNet (geometry repair) against the unmodified backbone on real re-photographs (development-set frozen evaluation). The cross-finding pattern motivates fixed per-finding routing.`));

// ============ 4. EXPERIMENTS ============
C.push(h1("4. Experiments"));
C.push(h2("4.1. Datasets and evaluation protocol"));
C.push(p(`Table 2 summarizes the data. Silver masks are derived from CheXlocalize expert segmentations [8] on CheXpert validation images: each clean–photo pair is registered (SIFT + BF matching + RANSAC homography; ratio test 0.7, threshold 5 px) and a pair passes ("reg-ok") with ≥25 inliers, inlier ratio ≥0.20, median reprojection error ≤4 px, and warped mask occupancy in [0.002, 0.6] (98.2% of validation pairs pass; median reprojection error 0.5 px). Passing masks are warped into the photographic frame at 224 px. **These are registration-based silver references, not radiologist annotations on photographs** — localization is measured relative to the photographed anatomy as aligned by SIFT. All comparisons are paired on the same image and patient (1 image = 1 patient in the evaluation set): ΔIoU@0.45 with patient-level bootstrap 95% CIs (400 resamples, seed 42), sign-flip permutation tests, and Holm correction within pre-specified primary families; wrong-repair counts carry exact Clopper–Pearson 95% upper bounds [25]. All artifacts are checksummed in a released manifest; every number traces to one canonical freeze table. Figure 5 summarizes the protocol.`));
C.push(tcap(`Table 2. Datasets. Silver masks from CheXlocalize [8] are registration-propagated into photographs (SIFT + RANSAC; 98.2% pass rate, median reprojection error 0.5 px). The evaluation set doubles as the development set; the locked external test (668 images) is pending and contributes no number to this paper.`));
C.push(table(
  [2000, 900, 1300, 760],
  [
    ["Split", "n", "Purpose", "Domain"],
    ["NIH ChestX-ray14 test, Cardiomegaly", "146 img", "mechanism audit (synthetic conditions)", "digital"],
    ["CheXphoto train/natural iPhone", "847 pairs", "PAIR-Loc training (Cardiomegaly)", "photo"],
    ["CheXphoto train/natural OnePlus", "129 pairs", "PAIR-Loc / RectNet training (Atelectasis)", "photo"],
    ["CheXphoto valid/natural OnePlus", "229 pairs / 123 patients", "frozen development-set evaluation (4 findings)", "photo"],
    ["CheXphoto test/natural", "668 img", "locked external test — pending", "photo"],
  ]));
C.push(...fig("fig2_protocol.png", 300, 142,
  `**Figure 5.** Evaluation protocol: registration-propagated silver labels, reg-ok criteria, paired patient-level statistics, and their interpretation boundary.`));

C.push(h2("4.2. Main results: repair effectiveness"));
C.push(tcap(`Table 3. Main results on the frozen development-set evaluation (123 patients, 229 finding–image pairs): paired ΔIoU@0.45 vs. the unmodified backbone, patient-level bootstrap 95% CI (400, seed 42), wrong-repair count and exact Clopper–Pearson 95% upper bound (CP). The sensitivity row uses early-stop model selection and is not the deployment configuration. * = CI > 0 after Holm correction (p < 1e-4).`));
C.push(table(
  [700, 1230, 330, 560, 900, 960],
  [
    ["Finding", "Action", "n", "Δ IoU45", "95% CI", "wrong (CP)"],
    ["Cardiomegaly", "PAIR-Loc (zero-contact)", "65", "+0.159*", "[+0.130, +0.192]", "7/65 (0.193)"],
    ["Cardiomegaly", "sensitivity: early-stop", "65", "+0.185", "[+0.157, +0.217]", "2/65 (0.094)"],
    ["Atelectasis", "RectNet", "73", "+0.202*", "[+0.174, +0.230]", "3/73 (0.103)"],
    ["Effusion", "PAIR-Loc (full)", "50", "+0.147*", "[+0.105, +0.186]", "1/50 (0.091)"],
    ["Edema", "RectNet + gate + fallback", "41", "+0.057*", "[+0.024, +0.090]", "4/41 (0.210)"],
    ["Pooled", "fixed routing", "229", "+0.152*", "[+0.135, +0.168]", "15/229 (0.099)"],
  ]));
C.push(caption(`Row status — *deployment configuration*: Cardiomegaly (zero-contact), Atelectasis, Effusion, and the pooled row; *sensitivity analysis* (early-stop model selection, not deployment): the second Cardiomegaly row; *mixed-mechanism stress test* (not a deployment claim): Edema. The three deployment findings gain +0.147 to +0.202; their wrong-repair point estimates are 2.0% (Effusion), 4.1% (Atelectasis) and 10.8% (Cardiomegaly). At this sample size the exact upper bounds reach 9.1–21%; we therefore report the bounds and do not claim a strict 5% risk guarantee (pooled bound 9.9%).`));
C.push(p(`On the clean-domain external check (146 NIH ChestX-ray14 images), the adapters change localization by Δ −0.007 to −0.009 with CIs containing zero: the readout repair is photographic-domain-specific and does not disturb clean behavior.`));

C.push(h2("4.3. Comparison with unified baselines"));
C.push(tcap(`Table 4. Baseline comparison (ΔIoU@0.45 vs. raw, development set, same protocol and statistics as Table 3). Only the no-learning center-field prior and the routed main action are shown per finding; the earlier gate-based system (gated_v2) and zero-shot repairer baselines are negative results and are discussed in Sec. 5.1.`));
C.push(table(
  [950, 1900, 330, 560, 900, 610],
  [
    ["Finding", "Comparator", "n", "Δ IoU45", "95% CI", "wrong (CP)"],
    ["Cardiomegaly", "center-field prior", "65", "−0.113", "[−0.152, −0.076]", "44/65 (0.772)"],
    ["Cardiomegaly", "PAIR-Loc (main)", "65", "+0.159*", "[+0.130, +0.192]", "7/65 (0.193)"],
    ["Atelectasis", "center-field prior", "73", "−0.012", "[−0.029, +0.006]", "9/73 (0.184)"],
    ["Atelectasis", "PAIR-Loc", "73", "+0.014", "[−0.002, +0.029]", "5/73 (0.130)"],
    ["Atelectasis", "RectNet (main)", "73", "+0.202*", "[+0.174, +0.230]", "3/73 (0.103)"],
    ["Effusion", "center-field prior", "50", "+0.001", "[−0.005, +0.007]", "3/50 (0.140)"],
    ["Effusion", "PAIR-Loc (main)", "50", "+0.147*", "[+0.105, +0.186]", "1/50 (0.091)"],
    ["Edema", "center-field prior", "41", "+0.001", "[−0.025, +0.029]", "12/41 (0.341)"],
    ["Edema", "RectNet + gate + fallback", "41", "+0.057*", "[+0.024, +0.090]", "4/41 (0.210)"],
  ]));
C.push(caption(`Two observations. (i) The no-learning center-field prior is Cardiomegaly-specific (Δ −0.113); for the peripheral/diffuse findings its Δ ≈ 0 (CI contains zero) — prior-based "repair" does not generalize across mechanisms. (ii) PAIR-Loc alone does not help Atelectasis (+0.014, CI contains zero) — the geometry-dominant mechanism requires RectNet, and this mechanism-specific asymmetry is exactly what the fixed routing encodes.`));

C.push(h2("4.4. Ablations and routing necessity"));
C.push(tcap(`Table 5. PAIR-Loc loss ablation (Cardiomegaly, 847 pairs, early-stop regime for comparability). The paired registration-aware CAM term carries the gain.`));
C.push(table(
  [2300, 1100, 1280],
  [
    ["Variant", "Δ IoU45", "wrong"],
    ["Full", "+0.129", "—"],
    ["− L(ident)  (deployment variant)", "+0.185", "2/65 (3.1%)"],
    ["− L(cam)  (CAM distillation)", "+0.024", "—"],
    ["− all paired losses (L(cam)+L(logit))", "+0.005", "collapse"],
    ["− L(geo)  (equivariance)", "+0.123", "—"],
    ["− L(anat)  (anatomical field)", "+0.148", "—"],
    ["− L(reg)  (registration-aware target)", "+0.089", "—"],
  ]));
C.push(caption(`Removing the paired CAM-distillation losses collapses Cardiomegaly to the raw level (+0.005), while removing the clean-identity term *improves* it (+0.185 with early-stop selection): with 847 pairs, L(ident) is a pure conservatism tax. On Atelectasis (129 pairs) the conclusion reverses — full +0.032 with 1.4% wrong-repair vs. no-identity +0.016 with 6.8% — so we describe L(ident) as a regularizer for small paired sets, not a core component. The deployment row is the zero-contact retrain of the no-identity variant (+0.159, Table 3), which avoids early-stop selection entirely.`));
C.push(tcap(`Table 6. Routing necessity: both actions evaluated on every finding (ΔIoU@0.45, wrong-repair; * = CI > 0). No single action covers all findings, and RectNet on Cardiomegaly is mean-neutral but harmful per-image (35% wrong-repair, 19 px corner error).`));
C.push(table(
  [950, 1250, 1250, 1230],
  [
    ["Finding", "PAIR-Loc Δ (wrong)", "RectNet Δ (wrong)", "Routed action"],
    ["Cardiomegaly", "+0.159* (7/65)", "+0.005 (35%)", "PAIR-Loc"],
    ["Atelectasis", "+0.014 (5/73)", "+0.202* (3/73)", "RectNet"],
    ["Effusion", "+0.147* (1/50)", "+0.021 (—)", "PAIR-Loc"],
    ["Edema", "+0.038* (8/41)", "+0.076 (11/41)", "mixed — stress test"],
  ]));

C.push(p(`**Geometric correction is domain-bounded.** RectNet transfers the mechanism evidence into a working repair on Atelectasis (+0.202), but the same module on Cardiomegaly is mean-neutral (+0.005) with 35% wrong-repair and 19 px corner error — the frozen backbone's features do not encode the global geometric cue for that finding. On CheXphoto valid/synthetic (digital), where there is no real geometry to undo, applying the predicted homography is actively harmful (−0.050 to −0.070). Physical corrections carry domain assumptions that must be switched on deliberately; this is precisely what the fixed routing encodes.`));

C.push(h2("4.5. Failed gating explorations and pending external validation"));
C.push(p(`**Failed gating explorations.** We attempted three generations of label-free gating to adaptively select repairs. A pre-registered patient-isolated audit retracted six claims: apparent gains originated in patient leakage or evaluation-label-informed thresholds, and signals with genuine discriminative power failed operationally. Label-free gating for these repairs remains an open problem; the deployed system uses fixed routing with a raw fallback. **External validation.** A pre-registered new-capture replication (frozen seed-42 protocol; a second smartphone and display, the same 123 source images — a capture-condition replication, not a patient-independent test) has been completed: registration passed 120/123 images (97.6%), within-shoot consistency was high (pass1–pass2 raw-IoU Pearson r = 0.93–0.98), and the frozen system was evaluated unchanged. The readout branch replicated — Cardiomegaly Δ +0.144 [+0.118, +0.170] with 4.8% wrong-repair, Effusion Δ +0.054 [+0.035, +0.070] — while the geometry branch did not transfer to the new capture geometry (Atelectasis Δ −0.005 with 41% wrong-repair), and the Edema stress test remained negative. The pooled primary endpoint met its pre-registered directional-replication criterion (Δ +0.044 [+0.030, +0.056], CI > 0); the branch divergence is the replication's main finding: frozen readout adapters survive a capture-domain change that breaks homography rectification. The locked external test on 668 CheXphoto test/natural images (independent patients and devices) remains pre-registered and pending.`));

// ============ 5. DISCUSSION ============
C.push(h1("5. Discussion"));
C.push(h2("5.1. Clinical implications"));
C.push(p(`The deployed configuration requires no new annotation, no clean–photo pairing at inference, and a single forward pass per image — compatible with the LMIC deployment path that motivates it: existing X-ray machines plus a smartphone. The gains are large where the mechanism matches (+0.15–0.20 IoU@0.45), and the wrong-repair accounting (2.0–10.8% point estimates across routed findings, exact bounds up to 19–21% at n = 41–73) says plainly that a clinical deployment must still verify repairs before acting on them. The Edema stress test shows where the approach stops: when both mechanisms are active and neither dominates, fixed routing degenerates to a modest average gain with real per-image risk.`));
C.push(p(`Our mechanism-specific approach also supersedes the earlier gate-based system (gated_v2), which was significantly negative on Cardiomegaly photographs (−0.044, CI [−0.078, −0.015]) despite having passed a synthetic-domain external test. Rebuilding repair from mechanism evidence rather than gate design produced the +0.203 [CI +0.168, +0.241] gain of the zero-contact configuration over gated_v2 on the same images — the clearest single demonstration in this work that matching the action to the failure mechanism, rather than learning when to apply a single action, is what the capture boundary rewards.`));
C.push(h2("5.2. Robustness–performance trade-off"));
C.push(p(`Our two negative-adjacent findings — medical fine-tuning reducing capture-shift robustness, and gating signals failing under patient isolation — share a lesson: performance measured in-domain says little about behavior at the capture-physics boundary, and the measurements that do transfer (paired, patient-level, pre-registered) can overturn apparently settled design choices. We suggest task-specific robustness evaluation as a standard component of medical-model development: report the clean-to-real paired delta alongside the in-domain metric, and treat synthetic chains as a screening tool with known 4–6× optimism at the localization layer. The completed new-capture replication adds the transfer-side lesson: the frozen readout adapter carried its repair across a capture-domain change that invalidated the geometric branch — mechanism specificity, not domain robustness, is what these repairs are certified for.`));
C.push(h2("5.3. Limitations"));
C.push(p(`(1) Evaluation uses registration-propagated silver masks, not radiologist annotations on photographs; exact wrong-repair upper bounds (9.1–21%) preclude strict risk claims at 123 patients. (2) The completed new-capture replication (second smartphone and display, same source cohort) confirmed the readout branch but showed that RectNet's homography is bound to the training capture geometry; the patient-independent external test (668 images) is pending. (3) Routing is fixed per finding and derived from development experiments — a validated protocol, not an adaptive system — and Edema's mixed mechanism is characterized but unsolved.`));

// ============ 6. CONCLUSION ============
C.push(h1("6. Conclusion"));
C.push(p(`We showed that weakly supervised localization in chest radiographs fails under real photographic capture shift through two distinct mechanisms — readout drift and capture geometry — that synthetic evaluations overestimate by 4–6× and that medical fine-tuning does not mitigate (and can worsen). Matching repair to mechanism — registration-aware paired distillation (PAIR-Loc) for readout failure, photo-only homography rectification (RectNet) for geometric failure, combined by a fixed per-finding routing — yields +0.15–0.20 IoU@0.45 in the real re-photography domain (p < 1e-4) with explicitly bounded wrong-repair rates, while the same audit protocol retracted six earlier gating claims. The completed pre-registered new-capture replication confirmed the readout branch, bounded the geometry branch's transfer, and the locked external test on independent patients and devices will subject the system to exactly the numbers reported here.`));

// ============ References ============
C.push(h1("References"));
[
  `B. Zhou, A. Khosla, A. Lapedriza, A. Oliva and A. Torralba, "Learning deep features for discriminative localization," in *Proc. IEEE CVPR*, 2016, pp. 2921–2929.`,
  `R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.`,
  `X. Wang *et al.*, "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization," in *Proc. IEEE CVPR*, 2017, pp. 2097–2106.`,
  `Z. Li, C. Wang, M. Han, Y. Xue, W. Wei, L.-J. Li and L. Fei-Fei, "Thoracic disease identification and localization with limited supervision," in *Proc. IEEE CVPR*, 2018, pp. 4655–4664.`,
  `E. Rozenberg, D. Freedman and A. Bronstein, "Localization with limited annotation for chest X-rays," in *Proc. ML4H (PMLR)*, vol. 116, 2020, pp. 52–65.`,
  `K. Yu, S. Ghosh, Z. Liu, C. Deible and K. Batmanghelich, "Anatomy-guided weakly-supervised abnormality localization in chest X-rays," in *Proc. MICCAI*, 2022.`,
  `L. Zhu, Q. She, Q. Chen, Y. You, B. Wang and Y. Lu, "Weakly supervised object localization as domain adaption," in *Proc. IEEE/CVF CVPR*, 2022, pp. 14637–14646.`,
  `A. Saporta *et al.*, "Benchmarking saliency methods for chest X-ray interpretation," *Nature Machine Intelligence*, vol. 4, pp. 867–878, 2022.`,
  `J. Irvin *et al.*, "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in *Proc. AAAI*, 2019, pp. 590–597.`,
  `Stanford ML Group, "CheXphoto: 10,000+ smartphone photos and synthetic photographic transformations of chest X-rays for benchmarking deep learning robustness," arXiv:2007.06199, 2020.`,
  `P. Rajpurkar, A. Joshi, A. Pareek, J. Irvin, A. Y. Ng and M. P. Lungren, "CheXphotogenic: Generalization of deep learning models for chest X-ray interpretation to photos of chest X-rays," arXiv:2011.06129, 2020.`,
  `J. M. Zech *et al.*, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Medicine*, vol. 15, no. 11, e1002683, 2018.`,
  `L. Oakden-Rayner, J. Dunnmon, G. Carneiro and C. Ré, "Hidden stratification: Clinically meaningful failure of machine learning in medical imaging," in *Proc. CHIL*, 2020, pp. 151–159.`,
  `A. J. DeGrave, J. D. Janizek and S. I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nature Machine Intelligence*, vol. 3, pp. 610–619, 2021.`,
  `A. Kirillov *et al.*, "Segment anything," in *Proc. IEEE ICCV*, 2023, pp. 4015–4026.`,
  `J. Ma *et al.*, "Segment anything in medical images," *Nature Communications*, vol. 15, art. 654, 2024.`,
  `B. Boecking *et al.*, "Making the most of text semantics to improve biomedical vision–language processing," in *Proc. ECCV*, 2022, pp. 1–21.`,
  `D. Wang, E. Shelhamer, S. Liu, B. Olshausen and T. Darrell, "Tent: Fully test-time adaptation by entropy minimization," in *Proc. ICLR*, 2021.`,
  `G. Hinton, O. Vinyals and J. Dean, "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015.`,
  `E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, 2022.`,
  `D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *Int. J. Comput. Vis.*, vol. 60, no. 2, pp. 91–110, 2004.`,
  `D. DeTone, T. Malisiewicz and A. Rabinovich, "Deep homography estimation," in *Proc. IEEE CVPR Workshops*, 2016, pp. 1142–1150.`,
  `C. F. Chong, Y. Wang, B. K. Ng, W. Luo and X. Yang, "Image projective transformation rectification with synthetic data for smartphone-captured chest X-ray photos classification," *Computers in Biology and Medicine*, vol. 164, art. 107277, 2023.`,
  `G. Huang, Z. Liu, L. van der Maaten and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700–4708.`,
  `C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial," *Biometrika*, vol. 26, no. 4, pp. 404–413, 1934.`,
].forEach((t, i) => C.push(ref(t, i + 1)));
// ---------- document ----------
const doc = new Document({
  numbering: {
    config: [{
      reference: "contrib",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: "left",
        style: { paragraph: { indent: { left: 280, hanging: 280 } } },
      }],
    }],
  },
  styles: { default: { document: { run: { font: TNR, size: 20 } } } },
  sections: [
    {
      properties: {
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1080, bottom: 1440, left: 890, right: 890 } },
      },
      children: [
        new Paragraph({
          children: runs(`Mechanism-Specific Localization Repair under Real Acquisition Shift in Chest Radiographs`, { bold: true, size: 34 }),
          alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
        }),
        new Paragraph({
          children: runs(`Huachen Gu`, { size: 22 }), alignment: AlignmentType.CENTER, spacing: { after: 40 },
        }),
        new Paragraph({
          children: runs(`Affiliation, Department — City, Country`, { size: 18, italics: true }),
          alignment: AlignmentType.CENTER, spacing: { after: 20 },
        }),
        new Paragraph({
          children: runs(`e-mail: author@example.com　　(Draft v0.3 — pre-registered replication and locked external test pending)`, { size: 16, color: "777777" }),
          alignment: AlignmentType.CENTER, spacing: { after: 60 },
        }),
      ],
    },
    {
      properties: {
        type: SectionType.CONTINUOUS,
        page: { size: { width: 12240, height: 15840 },
                margin: { top: 1080, bottom: 1440, left: 890, right: 890 } },
        column: { count: 2, space: 420 },
      },
      children: C,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "..", "论文初稿-Mechanism-Specific Localization Repair under Real Acquisition Shift.docx"), buf);
  console.log("docx written");
});
