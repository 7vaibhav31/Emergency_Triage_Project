# ============================================================
# rag_data/knowledge_base.py
#
# WHAT IS THIS FILE?
# This is our "medical knowledge base" — a list of text chunks
# that represent emergency protocols, drug info, and guidelines.
#
# IN A REAL SYSTEM: these chunks would come from actual PDF
# medical documents loaded into a vector database like ChromaDB
# or Pinecone. For this project, we hardcode them so you don't
# need any external database — everything runs locally.
#
# HOW RAG WORKS (simple explanation):
# 1. We have many knowledge chunks stored here
# 2. When a patient case comes in, we find the chunks most
#    SIMILAR to the symptoms (using sentence embeddings)
# 3. We inject only those relevant chunks into Claude's prompt
# 4. Claude uses that context to give a better, grounded answer
#
# This is better than just asking Claude directly because:
# - Claude gets specific protocol references
# - Answers are grounded in real guidelines
# - Less hallucination risk in clinical context
# ============================================================

KNOWLEDGE_CHUNKS = [

    # ── CARDIAC EMERGENCIES ──────────────────────────────────
    {
        "id": "cardiac_001",
        "category": "cardiac",
        "text": """STEMI Protocol: ST-Elevation Myocardial Infarction.
Presentation: crushing chest pain, radiation to arm/jaw, diaphoresis, nausea.
Vitals: hypotension, tachycardia, low SpO2.
Immediate actions: activate cath lab, dual antiplatelet therapy (aspirin + P2Y12),
anticoagulation, supplemental O2 if SpO2 < 94%.
Target door-to-balloon time: under 90 minutes.
High-risk features: cardiogenic shock, new LBBB, anterior ST elevation."""
    },
    {
        "id": "cardiac_002",
        "category": "cardiac",
        "text": """Cardiogenic Shock Protocol.
Definition: cardiac output failure causing end-organ hypoperfusion.
Signs: BP < 90 systolic, HR > 100, cold clammy skin, altered mentation, oliguria.
Management: avoid aggressive fluids, consider vasopressors (norepinephrine first line),
urgent revascularization if MI-related, mechanical circulatory support if refractory."""
    },
    {
        "id": "cardiac_003",
        "category": "cardiac",
        "text": """Unstable Angina / NSTEMI Protocol.
Presentation: chest pain at rest or minimal exertion, no ST elevation.
Troponin may be elevated. Risk stratify with TIMI or GRACE score.
Actions: aspirin, heparin, beta blocker if no contraindication,
nitrates for ongoing pain, cardiology consult within 24 hours."""
    },

    # ── NEUROLOGICAL EMERGENCIES ─────────────────────────────
    {
        "id": "neuro_001",
        "category": "neurology",
        "text": """Acute Ischemic Stroke Protocol — FAST Assessment.
FAST: Face drooping, Arm weakness, Speech difficulty, Time to call.
Golden window: IV tPA within 4.5 hours of symptom onset (if eligible).
Contraindications to tPA: active bleeding, recent surgery, BP > 185/110, blood glucose < 50.
Imaging: non-contrast CT head immediately to rule out hemorrhage.
Do not lower BP aggressively unless > 220/120."""
    },
    {
        "id": "neuro_002",
        "category": "neurology",
        "text": """Hemorrhagic Stroke / Intracranial Bleed Protocol.
Signs: sudden worst headache of life, vomiting, altered consciousness, focal deficits.
CT head without contrast is diagnostic. Reverse anticoagulation if on warfarin/DOAC.
Neurosurgery consult immediately. Control BP to target < 140 systolic.
Avoid antiplatelet agents and anticoagulants."""
    },

    # ── RESPIRATORY EMERGENCIES ──────────────────────────────
    {
        "id": "resp_001",
        "category": "respiratory",
        "text": """Anaphylaxis Protocol.
Trigger: allergen exposure (food, drug, insect, latex).
Signs: urticaria, angioedema, stridor, wheeze, hypotension, tachycardia.
First line: epinephrine IM 0.3–0.5mg into lateral thigh immediately.
Repeat every 5–15 min if no improvement. IV access, fluid bolus, supine position.
Second line: diphenhydramine, corticosteroids, bronchodilators.
Observe minimum 4–6 hours post-reaction for biphasic response."""
    },
    {
        "id": "resp_002",
        "category": "respiratory",
        "text": """Severe Asthma Exacerbation Protocol.
Signs of severity: SpO2 < 92%, inability to speak full sentences, accessory muscle use,
silent chest (ominous — air movement absent).
Treatment: salbutamol nebulized continuously, ipratropium, systemic corticosteroids,
magnesium sulfate IV for severe cases. Consider heliox and NIV.
Intubation: last resort — high risk in acute asthma."""
    },
    {
        "id": "resp_003",
        "category": "respiratory",
        "text": """Tension Pneumothorax Protocol.
Signs: tracheal deviation away from affected side, absent breath sounds,
hypotension, distended neck veins, hypoxia.
This is a clinical diagnosis — do not delay for imaging.
Immediate needle decompression: 2nd intercostal space, midclavicular line.
Follow with chest tube insertion."""
    },

    # ── TRAUMA ───────────────────────────────────────────────
    {
        "id": "trauma_001",
        "category": "trauma",
        "text": """Major Trauma — Primary Survey (ABCDE).
A: Airway with cervical spine control.
B: Breathing and ventilation.
C: Circulation with hemorrhage control — apply direct pressure, tourniquet for limb bleeding.
D: Disability — GCS, pupils, blood glucose.
E: Exposure — full body exam, prevent hypothermia.
Activate massive transfusion protocol (MTP) for hemodynamic instability."""
    },
    {
        "id": "trauma_002",
        "category": "trauma",
        "text": """Hemorrhagic Shock Classification.
Class I: < 750mL loss, HR < 100, normal BP. Minimal intervention.
Class II: 750–1500mL, HR 100–120, narrow pulse pressure. IV fluids.
Class III: 1500–2000mL, HR 120–140, hypotension. Blood products + surgery.
Class IV: > 2000mL, HR > 140, life-threatening. Immediate MTP + OR."""
    },

    # ── SEPSIS ───────────────────────────────────────────────
    {
        "id": "sepsis_001",
        "category": "sepsis",
        "text": """Sepsis and Septic Shock — Hour-1 Bundle.
Sepsis: suspected infection + 2 SIRS criteria or SOFA score ≥ 2.
Septic shock: sepsis + vasopressors needed to maintain MAP ≥ 65 + lactate > 2.
Hour-1 bundle: measure lactate, blood cultures x2 before antibiotics,
broad-spectrum antibiotics within 1 hour, 30mL/kg IV crystalloid if hypotensive."""
    },
    {
        "id": "sepsis_002",
        "category": "sepsis",
        "text": """qSOFA Score for Sepsis Screening (outside ICU).
Criteria: altered mentation (GCS < 15), RR ≥ 22/min, systolic BP ≤ 100mmHg.
Score ≥ 2 suggests high risk of poor outcome — escalate care immediately.
Common sources: pneumonia, UTI, abdominal, skin/soft tissue, line infection."""
    },

    # ── METABOLIC / DIABETIC ─────────────────────────────────
    {
        "id": "meta_001",
        "category": "metabolic",
        "text": """Diabetic Ketoacidosis (DKA) Protocol.
Diagnosis: glucose > 250, pH < 7.3, bicarbonate < 18, ketones present.
Signs: polyuria, polydipsia, nausea, fruity breath, Kussmaul breathing, abdominal pain.
Treatment: IV fluid resuscitation (0.9% NaCl), insulin infusion,
potassium replacement (add K+ when level < 5.5), monitor glucose hourly."""
    },
    {
        "id": "meta_002",
        "category": "metabolic",
        "text": """Hypoglycemia Protocol.
Definition: blood glucose < 70 mg/dL (3.9 mmol/L).
Severe: altered consciousness, seizure, inability to self-treat.
Conscious patient: 15g fast-acting carbohydrate PO, recheck in 15 min.
Unconscious: dextrose 50% IV push OR glucagon IM/intranasal.
Identify and treat underlying cause."""
    },

    # ── OVERDOSE / TOXICOLOGY ────────────────────────────────
    {
        "id": "tox_001",
        "category": "toxicology",
        "text": """Opioid Overdose Protocol.
Classic triad: miosis (pinpoint pupils), respiratory depression, unconsciousness.
Treatment: naloxone IV/IM/intranasal — titrate to adequate respirations (not full reversal).
Repeat dosing every 2–3 min as needed. Monitor for re-narcotization.
Observe minimum 4–6 hours. Supportive airway management."""
    },
]
