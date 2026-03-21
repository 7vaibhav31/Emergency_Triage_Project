# ============================================================
# app.py
#
# WHAT IS THIS FILE?
# The Flask web server. It has two jobs:
#
#   1. Serve the frontend HTML page (GET /)
#   2. Handle triage API requests     (POST /triage)
#
# HOW IT WORKS:
#   Browser → POST /triage with JSON body
#           → TriageService.run_triage()
#           → Returns JSON response
#           → Browser renders triage card
#
# TO RUN:
#   python app.py
#   Then open http://localhost:5000 in your browser
# ============================================================

import os
from flask import Flask, request, jsonify, render_template_string
from triage_service import TriageService

app = Flask(__name__)

# Initialize triage service ONCE at startup
# (RAG model loads here — takes ~10 sec first time)
print("[App] Initializing Triage Service...")
service = TriageService()
print("[App] Ready. Visit http://localhost:5000")


# ── ROUTE 1: Serve the frontend ────────────────────────────────
@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template_string(HTML_PAGE)


# ── ROUTE 2: Triage API endpoint ───────────────────────────────
@app.route("/triage", methods=["POST"])
def triage():
    """
    Accepts patient data, runs the full RAG + Claude pipeline,
    returns a structured triage assessment as JSON.

    Expected request body:
        { "symptoms": "...", "history": "..." }

    Returns:
        { "priority": ..., "diagnosis": ..., "actions": [...],
          "confidence": ..., "rationale": ..., "latency_ms": ...,
          "retrieved_protocols": [...] }
    """
    data = request.get_json()

    # Validate input
    symptoms = data.get("symptoms", "").strip()
    if not symptoms:
        return jsonify({"error": "Symptoms field is required."}), 400

    history = data.get("history", "")

    try:
        result = service.run_triage(symptoms=symptoms, history=history)
        return jsonify(result)
    except Exception as e:
        # Return error details so you can debug from the browser
        return jsonify({"error": str(e)}), 500


# ============================================================
# FRONTEND HTML
# All the UI lives here as a Python string so the project
# stays in a single folder with no separate HTML files.
# ============================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VB PROJECT</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── Design tokens (Light Pink/Skin Theme) ── */
  :root {
    --bg: #fff5f8;       /* Very light pink background */
    --surface: #ffffff;  /* Pure white cards/sidebar */
    --border: #fdd8e5;   /* Soft pink border */
    --text: #2d3748;     /* Dark text for readability */
    --muted: #718096;    /* Gray text */
    --red: #e53e3e;      /* Red for critical priorities */
    --yellow: #d69e2e;
    --green: #38a169;
    --accent: #ed64a6;   /* Vibrant pink accent */
    --accent-hover: #d53f8c;
    --mono: 'Courier New', monospace;
  }

  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, -apple-system, sans-serif;
         display: flex; min-height: 100vh; font-weight: 600; }

  /* ── Sidebar ── */
  aside { width: 250px; background: var(--surface); border-right: 1px solid var(--border);
          display: flex; flex-direction: column; }
  .logo-area { padding: 24px; border-bottom: 1px solid var(--border); }
  .logo { font-weight: 800; font-size: 1.4rem; color: var(--accent); letter-spacing: -0.5px; }
  
  nav { flex: 1; padding: 20px 0; display: flex; flex-direction: column; gap: 4px; }
  nav a { padding: 12px 24px; color: var(--text); text-decoration: none; font-weight: 600;
          font-size: 0.95rem; border-left: 3px solid transparent; transition: all 0.2s; }
  nav a:hover, nav a.active { background: #fff0f5; color: var(--accent); border-left-color: var(--accent); }

  /* ── Main Layout ── */
  .layout-wrapper { flex: 1; display: flex; flex-direction: column; }
  
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 16px 36px; display: flex; align-items: center; justify-content: space-between; }
  .header-title { font-weight: 600; font-size: 1.1rem; color: var(--muted); }
  
  .badge { font-family: var(--mono); font-size: 0.70rem; padding: 4px 10px;
           border: 1px solid var(--border); border-radius: 20px; color: var(--muted); background: var(--bg); font-weight: 700;}
  .badge.live { border-color: var(--green); color: var(--green); background: rgba(56,161,105,0.05); }

  main { flex: 1; display: grid; grid-template-columns: 1fr 1fr;
         gap: 24px; padding: 36px; width: 100%; max-width: 1400px; margin: 0 auto; }

  /* ── Cards ── */
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 26px; display: flex;
          flex-direction: column; gap: 18px; box-shadow: 0 4px 20px rgba(237,100,166,0.05); }
  .card-title { font-family: var(--mono); font-size: 0.8rem; color: var(--accent);
                letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; }

  /* ── Form elements ── */
  label { font-family: var(--mono); font-size: 0.8rem; color: var(--muted);
          text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
  textarea {
    background: #fafafa; border: 1px solid #e2e8f0; border-radius: 8px;
    color: var(--text); font-family: var(--mono); font-size: 0.9rem;
    line-height: 1.75; padding: 14px; resize: vertical; outline: none;
    transition: all 0.2s; width: 100%; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
  }
  textarea:focus { border-color: var(--accent); background: #ffffff; box-shadow: 0 0 0 3px rgba(237,100,166,0.15); }

  /* ── Scenario buttons ── */
  .scenarios { display: flex; flex-wrap: wrap; gap: 8px; }
  .sc-btn { font-family: var(--mono); font-size: 0.8rem; padding: 8px 16px; font-weight: 600;
            border: 1px solid var(--border); border-radius: 20px;
            background: var(--surface); color: var(--text); cursor: pointer;
            transition: all 0.2s; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
  .sc-btn:hover { border-color: var(--accent); color: var(--accent); background: #fff0f5; transform: translateY(-1px); }

  /* ── Run button ── */
  .run-btn { background: var(--accent); color: white; border: none;
             border-radius: 8px; padding: 15px 28px; font-weight: 700;
             font-size: 1.05rem; cursor: pointer; transition: all 0.2s;
             display: flex; align-items: center; gap: 8px; justify-content: center; width: 100%;
             box-shadow: 0 4px 15px rgba(237,100,166,0.3); }
  .run-btn:hover   { background: var(--accent-hover); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(237,100,166,0.4); }
  .run-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; box-shadow: none; }

  /* ── Result panel ── */
  .result-area { min-height: 350px; display: flex; flex-direction: column; gap: 16px; }
  .empty-state { display: flex; flex: 1; align-items: center; justify-content: center;
                 color: var(--muted); font-size: 0.95rem; font-style: italic; }

  /* ── Priority badge ── */
  .priority { display: inline-flex; align-items: center; gap: 6px;
              font-family: var(--mono); font-size: 0.75rem; font-weight: 800;
              padding: 6px 14px; border-radius: 6px; letter-spacing: 0.1em; }
  .priority.critical { background: rgba(229,62,62,0.1); color: var(--red); border: 1px solid rgba(229,62,62,0.2); }
  .priority.moderate { background: rgba(214,158,46,0.1); color: var(--yellow); border: 1px solid rgba(214,158,46,0.2); }
  .priority.low      { background: rgba(56,161,105,0.1); color: var(--green); border: 1px solid rgba(56,161,105,0.2); }

  /* ── Field groups ── */
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field-label { font-family: var(--mono); font-size: 0.75rem; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
  .field-value { font-size: 1rem; line-height: 1.6; color: var(--text); font-weight: 500; }

  /* ── Action list ── */
  .action-list { list-style: none; display: flex; flex-direction: column; gap: 8px; padding-left: 4px; }
  .action-list li { display: flex; gap: 12px; font-family: var(--mono);
                    font-size: 0.95rem; line-height: 1.6; color: var(--text); align-items: flex-start; }
  .action-list li::before { content: '✓'; color: var(--accent); font-weight: 900; }

  /* ── RAG debug panel ── */
  .rag-panel { background: #fafafa; border: 1px solid #e2e8f0;
               border-radius: 8px; padding: 16px; margin-top: 10px; }
  .rag-title { font-family: var(--mono); font-size: 0.75rem; color: var(--muted);
               letter-spacing: 0.08em; margin-bottom: 12px; font-weight: 700; }
  .rag-item  { display: flex; justify-content: space-between; padding: 8px 0;
               border-bottom: 1px solid #e2e8f0; font-family: var(--mono);
               font-size: 0.85rem; color: #4a5568; }
  .rag-item:last-child { border-bottom: none; padding-bottom: 0; }
  .rag-score { color: var(--accent); font-weight: 600; }

  /* ── Loading ── */
  .loading { display: flex; align-items: center; gap: 12px; padding: 30px 0;
             font-family: var(--mono); font-size: 0.95rem; color: var(--accent); font-weight: 600; justify-content: center; }
  .dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%;
         animation: bounce 0.9s infinite; }
  .dot:nth-child(2) { animation-delay: 0.15s; }
  .dot:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce { 0%,100%{transform:translateY(0)} 45%{transform:translateY(-8px)} }

  /* ── Footer ── */
  footer { background: var(--surface); border-top: 1px solid var(--border);
           padding: 24px; text-align: center; color: var(--muted); font-size: 0.95rem;
           margin-top: auto; display: flex; flex-direction: column; gap: 6px; }
  .footer-name { font-weight: 800; color: var(--text); font-size: 1.05rem; }
  .footer-email { color: var(--accent); font-family: var(--mono); font-size: 0.9rem; }

  @media (max-width: 900px) { main { grid-template-columns: 1fr; } body { flex-direction: column; } aside { width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border); } nav { flex-direction: row; flex-wrap: wrap; padding: 10px; gap: 10px; } nav a { padding: 8px 16px; border-left: none; border-bottom: 3px solid transparent; } nav a:hover, nav a.active { border-left: none; border-bottom-color: var(--accent); } }
</style>
</head>
<body>

<!-- SIDEBAR -->
<aside>
  <div class="logo-area">
    <div class="logo">VB PROJECT</div>
  </div>
  <nav>
    <a href="#" id="tab-dashboard" class="active" onclick="switchTab('dashboard')">🏥 Triage Dashboard</a>
    <a href="#" id="tab-upload" onclick="switchTab('upload')">📁 Patient Upload</a>
    <a href="#" id="tab-records" onclick="switchTab('records')">📋 Patient Records</a>
    <a href="#" id="tab-analytics" onclick="switchTab('analytics')">📈 Analytics</a>
    <a href="#" id="tab-resources" onclick="switchTab('resources')">⚕️ Resources</a>
    <a href="#" id="tab-settings" onclick="switchTab('settings')">⚙️ Settings</a>
    <a href="#" id="tab-developer" onclick="switchTab('developer')" style="margin-top: auto; border-top: 1px solid var(--border);">👨‍💻 About Developer</a>
  </nav>
</aside>

<!-- MAIN APP AREA -->
<div class="layout-wrapper">
  
  <header>
    <div class="header-title">Aegis: Advanced Medical Intelligence</div>
    <div style="display:flex;gap:10px;">
      <span class="badge live">● LIVE SYSTEM</span>
      <span class="badge">RAG + ICP RULES</span>
    </div>
  </header>

  <main>
    <!-- DASHBOARD SECTION -->
    <div id="dashboard-section" style="display: contents;">
      <!-- LEFT: Input panel -->
      <div class="card">
        <div class="card-title">PATIENT INTAKE</div>

      <div class="field">
        <label>Quick Scenarios</label>
        <div class="scenarios">
          <button class="sc-btn" onclick="load(0)">🫀 STEMI</button>
          <button class="sc-btn" onclick="load(1)">🧠 Stroke</button>
          <button class="sc-btn" onclick="load(2)">🩸 Trauma</button>
          <button class="sc-btn" onclick="load(3)">😮 Anaphylaxis</button>
          <button class="sc-btn" onclick="load(4)">🤒 Sepsis</button>
          <button class="sc-btn" onclick="load(5)">💉 DKA</button>
        </div>
      </div>

      <div class="field" style="margin-top: 4px;">
        <label>Symptoms / Presentation *</label>
        <textarea id="symptoms" rows="5"
          placeholder="Enter patient's chief complaint, vitals, onset, findings..."></textarea>
      </div>

      <div class="field">
        <label>Patient History (optional)</label>
        <textarea id="history" rows="3"
          placeholder="Known conditions, medications, allergies, recent visits..."></textarea>
      </div>

      <button class="run-btn" id="runBtn" onclick="runTriage()">
        ⚡ Run Triage Analysis
      </button>
    </div>

    <!-- RIGHT: Result panel -->
    <div class="card">
      <div class="card-title">TRIAGE ASSESSMENT</div>
      <div class="result-area" id="resultArea">
        <div class="empty-state">Select a scenario or enter patient details to begin triage...</div>
      </div>
      </div>
    </div>

    <!-- UPLOAD SECTION -->
    <div id="upload-section" style="display: none; grid-column: 1 / -1; width: 100%; max-width: 800px; margin: 0 auto; padding-top: 40px;">
      <div class="card" style="text-align: center; padding: 60px; border: 2px dashed var(--accent); cursor: pointer;" onclick="document.getElementById('patientFile').click()">
        <div class="card-title" style="font-size: 1.2rem; margin-bottom: 20px;">UPLOAD PATIENT DOCUMENT</div>
        <p style="color: var(--muted); margin-bottom: 30px; font-weight: 500; font-size: 1.05rem;">
          Upload a patient's medical history, chief complaints, or text-based prescription files (.txt format). <br/>
          The system will automatically parse the document and prepare the triage analysis.
        </p>
        <input type="file" id="patientFile" accept=".txt" style="display: none;" onchange="handleFileUpload(event)">
        <button class="run-btn" style="width: auto; margin: 0 auto; padding: 15px 40px;" onclick="event.stopPropagation(); document.getElementById('patientFile').click()">
          📄 Select .TXT File
        </button>
      </div>
    </div>

    <!-- NEW TABS SECTIONS -->
    <div id="records-section" style="display: none; grid-column: 1 / -1; width: 100%; max-width: 800px; margin: 0 auto; padding-top: 40px;">
      <div class="card" style="text-align: center; padding: 60px;">
        <h2 style="color: var(--accent); margin-bottom: 15px;">📋 Patient Records</h2>
        <p style="color: var(--muted); font-size: 1.1rem;">This module is currently offline. Connect to an EHR system to view historical records.</p>
      </div>
    </div>

    <div id="analytics-section" style="display: none; grid-column: 1 / -1; width: 100%; max-width: 1000px; margin: 0 auto; padding-top: 20px;">
      <h2 style="color: var(--accent); margin-bottom: 25px; font-size: 1.8rem; text-align: center;">📈 Triage Analytics Overview</h2>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 24px;">
        <div class="card" style="text-align: center; padding: 35px;">
          <h3 style="color: var(--muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px;">Total Analyses (30 Days)</h3>
          <p style="font-size: 2.8rem; font-weight: 800; color: var(--text);">1,248</p>
          <p style="color: var(--green); font-size: 0.9rem; font-weight: 700; margin-top: 8px;">+12.4% vs Last Month</p>
        </div>
        <div class="card" style="text-align: center; padding: 35px;">
          <h3 style="color: var(--muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px;">System Avg Latency</h3>
          <p style="font-size: 2.8rem; font-weight: 800; color: var(--text);">1.8s</p>
          <p style="color: var(--yellow); font-size: 0.9rem; font-weight: 700; margin-top: 8px;">Optimal Response Range</p>
        </div>
        <div class="card" style="text-align: center; padding: 35px;">
          <h3 style="color: var(--muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px;">Critical P1 Priority Rate</h3>
          <p style="font-size: 2.8rem; font-weight: 800; color: var(--red);">18%</p>
          <p style="color: var(--muted); font-size: 0.9rem; font-weight: 700; margin-top: 8px;">Standard ED Flow</p>
        </div>
      </div>
      
      <div class="card" style="margin-top: 24px; padding: 40px;">
        <h3 style="color: var(--text); font-size: 1.2rem; margin-bottom: 20px; text-align: center;">AI Confidence Score Distribution</h3>
        <div style="width: 100%; background: #f1f5f9; border-radius: 12px; height: 35px; display: flex; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
          <div style="width: 78%; background: var(--green); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.85rem; letter-spacing: 0.05em;">HIGH CONFIDENCE (>80%)</div>
          <div style="width: 18%; background: var(--yellow); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.85rem;">MODERATE</div>
          <div style="width: 4%; background: var(--red); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.85rem;">LOW</div>
        </div>
      </div>
    </div>

    <div id="resources-section" style="display: none; grid-column: 1 / -1; width: 100%; max-width: 900px; margin: 0 auto; padding-top: 20px;">
      <h2 style="color: var(--accent); margin-bottom: 25px; font-size: 1.8rem; text-align: center;">⚕️ Core RAG Clinical Protocols</h2>
      <p style="text-align: center; color: var(--muted); margin-bottom: 30px; font-size: 1.1rem;">These are the foundational medical guidelines injected directly into the active Inference Engine.</p>
      
      <div style="display: flex; flex-direction: column; gap: 18px;">
        
        <div class="card" style="padding: 26px 36px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s;" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">
          <div>
            <h3 style="color: var(--text); font-size: 1.2rem; margin-bottom: 6px; font-weight: 800;">AHA STEMI Guidelines</h3>
            <p style="color: var(--muted); font-size: 1rem; font-weight: 500;">Protocol for suspected Acute ST-Elevation Myocardial Infarction.</p>
          </div>
          <button class="sc-btn" style="color: var(--accent); border-color: var(--accent); padding: 10px 20px;" onclick="alert('Viewing Core Protocol: AHA STEMI')">View Details</button>
        </div>

        <div class="card" style="padding: 26px 36px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s;" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">
          <div>
            <h3 style="color: var(--text); font-size: 1.2rem; margin-bottom: 6px; font-weight: 800;">AHA/ASA Acute Ischemic Stroke</h3>
            <p style="color: var(--muted); font-size: 1rem; font-weight: 500;">Time-sensitive workflow and tPA screening criteria for stroke presentation.</p>
          </div>
          <button class="sc-btn" style="color: var(--accent); border-color: var(--accent); padding: 10px 20px;" onclick="alert('Viewing Core Protocol: AHA Ischemic Stroke')">View Details</button>
        </div>

        <div class="card" style="padding: 26px 36px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s;" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">
          <div>
            <h3 style="color: var(--text); font-size: 1.2rem; margin-bottom: 6px; font-weight: 800;">Surviving Sepsis Campaign</h3>
            <p style="color: var(--muted); font-size: 1rem; font-weight: 500;">Adult screening, sepsis, and septic shock early goal-directed therapy.</p>
          </div>
          <button class="sc-btn" style="color: var(--accent); border-color: var(--accent); padding: 10px 20px;" onclick="alert('Viewing Core Protocol: Surviving Sepsis')">View Details</button>
        </div>

        <div class="card" style="padding: 26px 36px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s;" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">
          <div>
            <h3 style="color: var(--text); font-size: 1.2rem; margin-bottom: 6px; font-weight: 800;">ATLS Major Trauma Outline</h3>
            <p style="color: var(--muted); font-size: 1rem; font-weight: 500;">Primary triage mapping prioritizing ABCDE interventions.</p>
          </div>
          <button class="sc-btn" style="color: var(--accent); border-color: var(--accent); padding: 10px 20px;" onclick="alert('Viewing Core Protocol: ATLS Major Trauma')">View Details</button>
        </div>

      </div>
    </div>

    <div id="settings-section" style="display: none; grid-column: 1 / -1; width: 100%; max-width: 800px; margin: 0 auto; padding-top: 40px;">
      <div class="card" style="text-align: center; padding: 60px;">
        <h2 style="color: var(--accent); margin-bottom: 15px;">⚙️ Application Settings</h2>
        <p style="color: var(--muted); font-size: 1.1rem;">No global settings defined right now. System pulls from local environment variables.</p>
      </div>
    </div>

    <div id="developer-section" style="display: none; grid-column: 1 / -1; width: 100%; max-width: 1000px; margin: 0 auto; padding-top: 40px;">
      <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center;">
        
        <div class="card" style="text-align: center; padding: 40px; flex: 1; min-width: 300px; max-width: 450px;">
          <img src="https://ui-avatars.com/api/?name=Vaibhav+Sharma&background=ed64a6&color=fff&size=100&bold=true" style="border-radius: 50%; margin: 0 auto 20px auto; width: 100px; height: 100px;">
          <h2 style="font-size: 2rem; color: var(--accent); margin-bottom: 10px;">Vaibhav Sharma</h2>
          <p style="font-size: 1.1rem; color: var(--text); font-weight: 700; margin-bottom: 5px;">B.Tech Student - CSE AI/ML</p>
          <p style="font-size: 1rem; color: var(--muted); margin-bottom: 30px; line-height: 1.6; font-weight: 500;">
            Passionate about building intelligent AI-powered solutions like this robust clinical triage system.
          </p>
          <div style="display: flex; justify-content: center; gap: 15px;">
            <a href="https://github.com/7vaibhav31" target="_blank" style="display: flex; align-items: center; gap: 8px; font-weight: 800; color: var(--text); text-decoration: none; padding: 8px 16px; border: 2px solid #e2e8f0; border-radius: 12px; transition: all 0.2s;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg> GitHub
            </a>
            <a href="https://www.linkedin.com/in/vaibhav731/" target="_blank" style="display: flex; align-items: center; gap: 8px; font-weight: 800; color: #0a66c2; text-decoration: none; padding: 8px 16px; border: 2px solid #0a66c2; border-radius: 12px; transition: all 0.2s;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg> LinkedIn
            </a>
          </div>
        </div>

        <div class="card" style="text-align: center; padding: 40px; flex: 1; min-width: 300px; max-width: 450px;">
          <img src="https://ui-avatars.com/api/?name=Bhaskar+Mishra&background=ed64a6&color=fff&size=100&bold=true" style="border-radius: 50%; margin: 0 auto 20px auto; width: 100px; height: 100px;">
          <h2 style="font-size: 2rem; color: var(--accent); margin-bottom: 10px;">Bhaskar Mishra</h2>
          <p style="font-size: 1.1rem; color: var(--text); font-weight: 700; margin-bottom: 5px;">3rd-year B.Tech Explorer</p>
          <p style="font-size: 1rem; color: var(--muted); margin-bottom: 30px; line-height: 1.6; font-weight: 500;">
            A curious explorer turning code, data, and ideas into intelligent solutions.
          </p>
          <div style="display: flex; justify-content: center; gap: 15px;">
            <a href="https://github.com/Bhaskar7462" target="_blank" style="display: flex; align-items: center; gap: 8px; font-weight: 800; color: var(--text); text-decoration: none; padding: 8px 16px; border: 2px solid #e2e8f0; border-radius: 12px; transition: all 0.2s;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg> GitHub
            </a>
            <a href="https://www.linkedin.com/in/kumar-bhaskar-3727162b3" target="_blank" style="display: flex; align-items: center; gap: 8px; font-weight: 800; color: #0a66c2; text-decoration: none; padding: 8px 16px; border: 2px solid #0a66c2; border-radius: 12px; transition: all 0.2s;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg> LinkedIn
            </a>
          </div>
        </div>

      </div>
    </div>
  </main>

  <!-- FOOTER -->
  <footer>
    <div>Built with ♥ by <span class="footer-name">Vaibhav Sharma</span></div>
    <div class="footer-email">Contact: m.7vansh31@gmail.com</div>
  </footer>
</div>

<script>
// ── Tab Switching & Upload Logic ───────────────────────────────
function switchTab(tab) {
  // Update sidebar active state
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');

  // Toggle all sections
  const tabs = ['dashboard', 'upload', 'records', 'analytics', 'resources', 'settings', 'developer'];
  tabs.forEach(t => {
    let el = document.getElementById(t + '-section');
    if(el) {
      if(t === tab) {
        el.style.display = (t === 'dashboard') ? 'contents' : 'block';
      } else {
        el.style.display = 'none';
      }
    }
  });
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Reset value so same file can be uploaded again if needed
  event.target.value = '';

  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    
    // Switch to dashboard
    switchTab('dashboard');
    
    // Populate symptoms textarea
    const symptomsField = document.getElementById("symptoms");
    symptomsField.value = "[[ DOCUMENT UPLOAD: " + file.name + " ]]\\n\\n" + text;
    
    // Visual feedback
    symptomsField.style.transition = "all 0.3s";
    symptomsField.style.borderColor = "var(--accent)";
    symptomsField.style.boxShadow = "0 0 0 3px rgba(237,100,166,0.15)";
    setTimeout(() => {
      symptomsField.style.borderColor = "#e2e8f0";
      symptomsField.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)";
      
      // Auto-run the triage
      runTriage();
    }, 500);
  };
  reader.readAsText(file);
}

// ── Preset scenarios ──────────────────────────────────────────
const SCENARIOS = [
  {
    symptoms: "65-year-old male. Sudden crushing chest pain radiating to left arm and jaw. Diaphoretic, pale. BP 82/50, HR 125, SpO2 91%. Onset 25 minutes ago.",
    history: "Known hypertension, type 2 diabetes. On metformin, amlodipine. Smoker 30 pack-years. No prior cardiac history."
  },
  {
    symptoms: "72-year-old female. Sudden right-sided facial droop, slurred speech, right arm weakness. FAST positive. Symptom onset approximately 40 minutes ago.",
    history: "Atrial fibrillation on warfarin. INR 2.4 two weeks ago. Hypertension, on lisinopril."
  },
  {
    symptoms: "28-year-old male. Motorcycle accident. Deep thigh laceration with arterial bleeding. GCS 13. BP 78/48, HR 136. Pale and confused.",
    history: "No known medical history. No medical alert bracelet."
  },
  {
    symptoms: "19-year-old female. Peanut exposure 5 minutes ago. Diffuse urticaria, throat tightness, audible stridor. BP 72/38, HR 145. Has known peanut allergy.",
    history: "Peanut allergy diagnosed age 8. Carries EpiPen — not yet administered."
  },
  {
    symptoms: "60-year-old male. Fever 39.9°C, HR 128, RR 26, BP 82/52, SpO2 90% on room air. Confused, mottled extremities. Indwelling urinary catheter.",
    history: "Type 2 diabetes, CKD stage 3. On insulin and lisinopril. Post-prostate surgery 3 weeks ago."
  },
  {
    symptoms: "22-year-old type 1 diabetic. Nausea, vomiting, abdominal pain, fruity breath. RR 28 (Kussmaul breathing). Blood glucose 490 mg/dL. Altered mentation.",
    history: "Type 1 diabetes since age 9. On insulin pump. Pump site infection noted. No other conditions."
  }
];

function load(i) {
  document.getElementById("symptoms").value = SCENARIOS[i].symptoms;
  document.getElementById("history").value  = SCENARIOS[i].history;
}

async function runTriage() {
  const symptoms = document.getElementById("symptoms").value.trim();
  const history  = document.getElementById("history").value.trim();

  if (!symptoms) { alert("Please enter patient symptoms."); return; }

  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing Patient Data...";

  document.getElementById("resultArea").innerHTML = `
    <div class="loading">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      Processing Triage via AI & Medical Protocols...
    </div>`;

  try {
    const res  = await fetch("/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms, history })
    });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    renderResult(data);
  } catch (err) {
    document.getElementById("resultArea").innerHTML =
      `<div style="color:var(--red);font-family:var(--mono);font-size:0.9rem;font-weight:600;padding:20px;">
        ⚠️ Error: ${err.message}
       </div>`;
  }

  btn.disabled = false;
  btn.textContent = "⚡ Run Triage Analysis";
}

function renderResult(d) {
  const pClass = d.priority.toLowerCase();
  const dot    = { critical: "🚨", moderate: "⚠️", low: "ℹ️" }[pClass] || "";

  const actionsHTML = (d.actions || [])
    .map(a => `<li>${a}</li>`).join("");

  const ragHTML = (d.retrieved_protocols || []).map(p => `
    <div class="rag-item">
      <span>${p.category.toUpperCase()}</span>
      <span class="rag-score">relevance: ${(p.score * 100).toFixed(0)}%</span>
    </div>`).join("");

  const cColor = d.confidence >= 75 ? "var(--green)"
               : d.confidence >= 50 ? "var(--yellow)"
               : "var(--red)";

  document.getElementById("resultArea").innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:8px;">
      <span class="priority ${pClass}">${dot} ${d.priority}</span>
      <div style="display:flex;gap:16px;align-items:center;background:#fafafa;padding:6px 12px;border:1px solid #e2e8f0;border-radius:6px;">
        <span style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);font-weight:700;">
          CONFIDENCE: <span style="color:${cColor};font-size:0.9rem;">${d.confidence}%</span>
        </span>
        <span style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);font-weight:700;">
          LATENCY: <span style="color:var(--accent);">${d.latency_ms}ms</span>
        </span>
      </div>
    </div>

    <div class="field">
      <div class="field-label">Probable Diagnosis</div>
      <div class="field-value" style="font-size:1.1rem;color:var(--accent);font-weight:800;">${d.diagnosis}</div>
    </div>

    <div class="field">
      <div class="field-label">Immediate Recommended Actions</div>
      <ul class="action-list">${actionsHTML}</ul>
    </div>

    <div class="field" style="margin-top:6px;">
      <div class="field-label">Clinical Rationale</div>
      <div class="field-value" style="font-family:var(--mono);font-size:0.9rem;color:#4a5568;background:#fff5f8;padding:14px;border-radius:8px;border-left:4px solid var(--accent);">
        ${d.rationale}
      </div>
    </div>

    <div class="rag-panel">
      <div class="rag-title">KNOWLEDGE BASE MATCHES (TOP 3)</div>
      ${ragHTML}
    </div>`;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    # Check for API key before starting
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  ERROR: OPENROUTER_API_KEY not set.")
        print("   Run: $env:OPENROUTER_API_KEY='your_key_here'  (Windows PowerShell)")
        print("   Or:  export OPENROUTER_API_KEY=your_key_here  (Mac/Linux)\n")
        exit(1)

    app.run(debug=True, port=5000)
