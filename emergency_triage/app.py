import os
from flask import Flask, request, jsonify, render_template_string
from triage_service import DocumentQAService

app = Flask(__name__)

print("[App] Initializing Document QA Engine...")
qa_service = DocumentQAService()
print("[App] Service Ready. Visit http://localhost:5000")

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/upload", methods=["POST"])
def upload_doc():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text provided."}), 400
    try:
        chunks = qa_service.ingest_document(text)
        return jsonify({"message": f"Successfully indexed into {chunks} context fragments.", "chunks": chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/qa", methods=["POST"])
def qa_document():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query required."}), 400
    try:
        if not qa_service.rag.chunks:
            return jsonify({"error": "Please upload a prescription or document first."}), 400
        result = qa_service.answer_medical_question(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MedRAG — AI Doctor Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --pink:         #ff6b9d;
    --pink-light:   #ffb3d1;
    --pink-soft:    #fff0f6;
    --yellow:       #ffcc44;
    --yellow-light: #fff5cc;
    --yellow-warm:  #ffe58a;
    --sky:          #87ceeb;
    --sky-light:    #d6f0ff;
    --sky-mid:      #5ba3c9;
    --warm-bg:      #fffaf3;
    --text-dark:    #2d1b4e;
    --text-mid:     #6b4f7c;
    --text-light:   #a08fb0;
    --border:       #f0e0f5;
    --shadow:       0 8px 32px rgba(255,107,157,0.10);
    --shadow-lg:    0 20px 60px rgba(255,107,157,0.15);
  }

  body { font-family: 'Inter', sans-serif; background: var(--warm-bg); color: var(--text-dark); overflow-x: hidden; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--pink-soft); }
  ::-webkit-scrollbar-thumb { background: var(--pink-light); border-radius: 10px; }

  /* PAGE SYSTEM */
  .page { display: none; min-height: 100vh; }
  .page.active { display: flex; animation: pageIn 0.55s cubic-bezier(0.16,1,0.3,1) forwards; }
  @keyframes pageIn { from { opacity:0; transform:translateY(28px); } to { opacity:1; transform:translateY(0); } }

  /* BUTTONS */
  .btn-primary {
    background: linear-gradient(135deg, var(--pink) 0%, #ff9a44 100%);
    color: white; border: none; padding: 16px 44px; border-radius: 50px;
    font-size: 1.05rem; font-weight: 700; cursor: pointer;
    box-shadow: 0 6px 24px rgba(255,107,157,0.40); transition: all 0.3s ease; letter-spacing:0.5px;
  }
  .btn-primary:hover { transform: translateY(-3px); box-shadow: 0 12px 36px rgba(255,107,157,0.50); }

  /* SPINNER */
  .spinner { width:20px; height:20px; border:3px solid rgba(255,255,255,0.3); border-top:3px solid white; border-radius:50%; animation:spin 0.8s linear infinite; display:inline-block; }
  @keyframes spin { to { transform:rotate(360deg); } }

  /* BLOBS */
  .blob-wrap { position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:0; }
  .blob { position:absolute; border-radius:50%; filter:blur(80px); opacity:0.22; animation:blobFloat 8s ease-in-out infinite alternate; }
  .blob-1 { width:500px; height:500px; background:var(--pink);   top:-100px; left:-100px; animation-delay:0s; }
  .blob-2 { width:400px; height:400px; background:var(--yellow); top:30%; right:-80px;  animation-delay:-3s; }
  .blob-3 { width:350px; height:350px; background:var(--sky);    bottom:-80px; left:40%; animation-delay:-5s; }
  @keyframes blobFloat { from { transform:translate(0,0) scale(1); } to { transform:translate(30px,-40px) scale(1.08); } }

  /* PULSE RINGS */
  .pulse-ring { position:absolute; width:300px; height:300px; border-radius:50%; border:2px solid rgba(255,107,157,0.15); top:50%; left:50%; transform:translate(-50%,-50%); animation:ringPulse 3s ease-out infinite; }
  .pulse-ring:nth-child(2) { animation-delay:1s; }
  .pulse-ring:nth-child(3) { animation-delay:2s; }
  @keyframes ringPulse { 0% { transform:translate(-50%,-50%) scale(1); opacity:0.4; } 100% { transform:translate(-50%,-50%) scale(2.5); opacity:0; } }

  /* BADGE */
  .badge { display:inline-flex; align-items:center; gap:8px; background:linear-gradient(135deg,#ffe4f0,#fff5cc); border:1.5px solid var(--pink-light); color:var(--pink); padding:8px 20px; border-radius:50px; font-size:0.82rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:32px; }
  .badge-dot { width:8px; height:8px; background:var(--pink); border-radius:50%; animation:pulseDot 1.5s ease-in-out infinite; }
  @keyframes pulseDot { 0%,100% { transform:scale(1); opacity:1; } 50% { transform:scale(1.5); opacity:0.6; } }

  /* ======================== PAGE 1 — LANDING ======================== */
  #page-landing {
    flex-direction:column; align-items:center; justify-content:center;
    position:relative; background:linear-gradient(135deg,#fff5fa 0%,#fffef0 50%,#f0f9ff 100%);
    min-height:100vh; padding:40px 20px; text-align:center;
  }
  .landing-content { position:relative; z-index:1; max-width:800px; }
  .hero-title {
    font-size:clamp(2.8rem,6vw,5rem); font-weight:900; line-height:1.1; margin-bottom:24px;
    background:linear-gradient(135deg,var(--pink) 0%,#ff9a44 40%,var(--sky-mid) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  }
  .hero-desc { font-size:1.15rem; color:var(--text-mid); line-height:1.8; margin-bottom:48px; max-width:620px; margin-left:auto; margin-right:auto; }
  .hero-features { display:flex; gap:16px; justify-content:center; margin-bottom:56px; flex-wrap:wrap; }
  .feature-pill { display:flex; align-items:center; gap:8px; background:white; border:1.5px solid var(--border); padding:10px 20px; border-radius:50px; font-size:0.88rem; font-weight:600; color:var(--text-dark); box-shadow:0 4px 16px rgba(255,107,157,0.08); }
  .scroll-hint { margin-top:40px; color:var(--text-light); font-size:0.85rem; font-weight:500; animation:bounce 2s infinite; display:flex; flex-direction:column; align-items:center; gap:6px; }
  @keyframes bounce { 0%,100% { transform:translateY(0); } 50% { transform:translateY(8px); } }

  /* ======================== PAGE 2 — UPLOAD ======================== */
  #page-upload {
    flex-direction:column; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#fff9f0 0%,#fff5fa 50%,#f5f5ff 100%);
    min-height:100vh; padding:60px 20px; position:relative;
  }
  .upload-card { position:relative; z-index:1; background:white; border-radius:32px; padding:52px 56px; width:100%; max-width:620px; box-shadow:var(--shadow-lg); border:1.5px solid var(--border); text-align:center; }
  .upload-icon-wrap { width:96px; height:96px; background:linear-gradient(135deg,var(--pink-soft),var(--yellow-light)); border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 24px; font-size:2.4rem; border:2px solid var(--pink-light); }
  .upload-card h1 { font-size:1.9rem; font-weight:800; color:var(--text-dark); margin-bottom:10px; }
  .upload-card p  { color:var(--text-mid); margin-bottom:32px; font-size:0.96rem; line-height:1.7; }
  .drop-zone { border:2.5px dashed var(--pink-light); border-radius:20px; padding:44px 20px; cursor:pointer; transition:all 0.3s ease; background:linear-gradient(135deg,#fff9fc,#fffef5); margin-bottom:20px; }
  .drop-zone:hover { border-color:var(--pink); background:var(--pink-soft); transform:scale(1.01); }
  .drop-zone .dz-icon { font-size:2.8rem; margin-bottom:10px; }
  .drop-zone h3 { font-size:1.05rem; font-weight:700; color:var(--text-dark); margin-bottom:6px; }
  .drop-zone small { color:var(--text-light); font-size:0.83rem; }
  #uploadStatus { display:none; margin-top:18px; padding:14px 20px; border-radius:14px; font-weight:600; font-size:0.93rem; }
  #uploadStatus.success { background:linear-gradient(135deg,#f0fff8,#e8f5ff); border:1.5px solid #a8e6cf; color:#1a7a4a; }
  #uploadStatus.error   { background:#fff5f5; border:1.5px solid #ffb3b3; color:#cc3333; }
  #uploadStatus.loading { background:linear-gradient(135deg,var(--pink-soft),var(--yellow-light)); border:1.5px solid var(--pink-light); color:var(--pink); }
  .back-btn { margin-top:16px; background:none; border:1.5px solid var(--border); color:var(--text-light); padding:10px 24px; border-radius:50px; font-size:0.86rem; font-weight:600; cursor:pointer; transition:all 0.2s; font-family:'Inter',sans-serif; }
  .back-btn:hover { border-color:var(--pink-light); color:var(--pink); }

  /* ======================== PAGE 3 — CHAT ======================== */
  #page-chat { flex-direction:row; min-height:100vh; background:var(--warm-bg); }

  /* SIDEBAR */
  .sidebar { width:252px; flex-shrink:0; background:white; border-right:1.5px solid var(--border); display:flex; flex-direction:column; padding:24px 16px; box-shadow:4px 0 20px rgba(255,107,157,0.06); }
  .sidebar-logo { display:flex; align-items:center; gap:10px; padding:8px 10px; margin-bottom:24px; }
  .logo-icon { width:38px; height:38px; background:linear-gradient(135deg,var(--pink),#ff9a44); border-radius:11px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; flex-shrink:0; }
  .logo-text { font-size:1.05rem; font-weight:800; color:var(--text-dark); }
  .logo-sub  { font-size:0.68rem; font-weight:600; color:var(--pink); letter-spacing:0.5px; }
  .sidebar-section-title { font-size:0.68rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--text-light); padding:0 10px; margin-bottom:6px; margin-top:16px; }
  .nav-link { display:flex; align-items:center; gap:10px; padding:11px 12px; border-radius:13px; border:none; background:none; font-family:'Inter',sans-serif; font-size:0.88rem; font-weight:600; color:var(--text-mid); cursor:pointer; width:100%; text-align:left; transition:all 0.2s ease; }
  .nav-link:hover { background:var(--pink-soft); color:var(--pink); }
  .nav-link.active { background:linear-gradient(135deg,var(--pink-soft),var(--yellow-light)); color:var(--pink); box-shadow:0 3px 14px rgba(255,107,157,0.12); }
  .nav-icon { width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:0.95rem; background:rgba(255,107,157,0.08); flex-shrink:0; }
  .nav-link.active .nav-icon { background:rgba(255,107,157,0.18); }
  .sidebar-doc-info { margin-top:auto; padding:14px; background:linear-gradient(135deg,var(--pink-soft),var(--sky-light)); border-radius:14px; border:1.5px solid var(--pink-light); }
  .doc-label  { font-size:0.68rem; font-weight:700; color:var(--text-light); letter-spacing:1px; text-transform:uppercase; margin-bottom:5px; }
  .doc-name   { font-size:0.85rem; font-weight:700; color:var(--text-dark); word-break:break-all; }
  .doc-chunks { font-size:0.75rem; color:var(--text-mid); margin-top:3px; }

  /* MAIN PANEL */
  .main-panel { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  .top-bar { background:white; border-bottom:1.5px solid var(--border); padding:14px 26px; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }
  .top-bar-title { font-size:1rem; font-weight:700; color:var(--text-dark); }
  .top-bar-status { display:flex; align-items:center; gap:8px; font-size:0.8rem; font-weight:600; color:#1a7a4a; background:linear-gradient(135deg,#f0fff8,#e8f5ff); border:1px solid #a8e6cf; padding:5px 14px; border-radius:50px; }
  .status-dot { width:7px; height:7px; background:#22c55e; border-radius:50%; animation:pulseDot 1.5s infinite; }

  /* PANELS */
  .panel-section { display:none; flex:1; overflow:hidden; flex-direction:column; }
  .panel-section.active { display:flex; animation:pageIn 0.4s ease forwards; }

  /* CHAT */
  .chat-messages { flex:1; overflow-y:auto; padding:24px 28px; display:flex; flex-direction:column; gap:18px; }
  .chat-welcome { text-align:center; padding:50px 20px; }
  .chat-welcome .welcome-icon { font-size:3.2rem; margin-bottom:14px; }
  .chat-welcome h3 { font-size:1.3rem; font-weight:700; color:var(--text-dark); margin-bottom:8px; }
  .chat-welcome p { font-size:0.92rem; color:var(--text-mid); line-height:1.6; max-width:380px; margin:auto; }
  .suggest-chips { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:20px; }
  .chip { background:white; border:1.5px solid var(--border); color:var(--text-dark); padding:8px 16px; border-radius:50px; font-size:0.8rem; font-weight:600; cursor:pointer; transition:all 0.2s; font-family:'Inter',sans-serif; }
  .chip:hover { border-color:var(--pink); color:var(--pink); background:var(--pink-soft); }

  .msg-wrap { display:flex; flex-direction:column; gap:4px; }
  .msg-wrap.user { align-items:flex-end; }
  .msg-wrap.bot  { align-items:flex-start; }
  .msg-bubble { max-width:80%; padding:14px 20px; border-radius:20px; font-size:0.96rem; line-height:1.75; font-weight:500; }
  .msg-wrap.user .msg-bubble { background:linear-gradient(135deg,var(--pink) 0%,#ff9a44 100%); color:white; border-bottom-right-radius:5px; box-shadow:0 6px 20px rgba(255,107,157,0.3); }
  .msg-wrap.bot  .msg-bubble { background:white; color:var(--text-dark); border-bottom-left-radius:5px; border:1px solid var(--border); box-shadow:var(--shadow-lg); }
  
  /* Markdown Styles for Bot */
  .msg-wrap.bot .msg-bubble p { margin-bottom: 0.9em; }
  .msg-wrap.bot .msg-bubble p:last-child { margin-bottom: 0; }
  .msg-wrap.bot .msg-bubble a { color: var(--sky-mid); text-decoration: none; font-weight: 700; border-bottom: 1.5px solid transparent; transition: all 0.2s; }
  .msg-wrap.bot .msg-bubble a:hover { border-bottom-color: var(--sky-mid); color: #2b7da8; }
  .msg-wrap.bot .msg-bubble ul, .msg-wrap.bot .msg-bubble ol { margin-left: 24px; margin-bottom: 0.9em; padding-left: 0; }
  .msg-wrap.bot .msg-bubble li { margin-bottom: 0.4em; }
  .msg-wrap.bot .msg-bubble li::marker { color: var(--pink); font-weight: 800; }
  .msg-wrap.bot .msg-bubble code { background: var(--warm-bg); padding: 3px 6px; border-radius: 6px; font-family: monospace; font-size: 0.88rem; color: #d0306c; border: 1px solid var(--border); }
  .msg-wrap.bot .msg-bubble pre { background: var(--text-dark); color: #f8f8f2; padding: 14px 18px; border-radius: 12px; overflow-x: auto; margin-bottom: 0.9em; border: 1px solid rgba(0,0,0,0.1); }
  .msg-wrap.bot .msg-bubble pre code { background: none; color: inherit; padding: 0; border: none; font-size: 0.85rem; }
  .msg-wrap.bot .msg-bubble strong, .msg-wrap.bot .msg-bubble b { font-weight: 700; color: #1a1a1a; }
  .msg-wrap.bot .msg-bubble h1, .msg-wrap.bot .msg-bubble h2, .msg-wrap.bot .msg-bubble h3 { font-size: 1.1rem; margin-top: 1em; margin-bottom: 0.6em; font-weight: 800; color: var(--text-dark); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .msg-wrap.bot .msg-bubble h1:first-child, .msg-wrap.bot .msg-bubble h2:first-child, .msg-wrap.bot .msg-bubble h3:first-child { margin-top: 0; }

  .msg-meta { font-size:0.73rem; color:var(--text-light); font-weight:500; display:flex; gap:6px; align-items:center; }
  .latency-badge { background:linear-gradient(135deg,var(--yellow-light),var(--sky-light)); border:1px solid var(--yellow-warm); color:#9a6700; padding:2px 10px; border-radius:50px; font-size:0.7rem; font-weight:700; }
  .chunks-badge  { background:var(--sky-light); border:1px solid #aadaff; color:var(--sky-mid); padding:2px 10px; border-radius:50px; font-size:0.7rem; font-weight:700; }
  .typing-bubble { background:white; border:1.5px solid var(--border); padding:13px 18px; border-radius:20px; border-bottom-left-radius:5px; box-shadow:var(--shadow); display:flex; gap:6px; align-items:center; }
  .typing-dot { width:8px; height:8px; background:var(--pink-light); border-radius:50%; animation:typingBounce 1.4s ease-in-out infinite; }
  .typing-dot:nth-child(2) { animation-delay:0.2s; }
  .typing-dot:nth-child(3) { animation-delay:0.4s; }
  @keyframes typingBounce { 0%,80%,100% { transform:translateY(0); } 40% { transform:translateY(-8px); } }

  .chat-input-area { padding:18px 24px; background:white; border-top:1.5px solid var(--border); display:flex; gap:12px; align-items:flex-end; flex-shrink:0; }
  .chat-input-wrap { flex:1; background:var(--warm-bg); border:2px solid var(--border); border-radius:18px; padding:11px 16px; transition:border-color 0.2s; }
  .chat-input-wrap:focus-within { border-color:var(--pink-light); }
  #chatInput { width:100%; background:none; border:none; outline:none; font-family:'Inter',sans-serif; font-size:0.93rem; color:var(--text-dark); resize:none; max-height:120px; overflow-y:auto; min-height:22px; }
  #chatInput::placeholder { color:var(--text-light); }
  .send-btn { width:46px; height:46px; background:linear-gradient(135deg,var(--pink),#ff9a44); border:none; border-radius:14px; color:white; font-size:1.15rem; cursor:pointer; box-shadow:0 4px 16px rgba(255,107,157,0.35); transition:all 0.2s; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
  .send-btn:hover { transform:scale(1.06); }
  .send-btn:disabled { opacity:0.5; cursor:not-allowed; transform:none; }

  /* METRICS */
  .metrics-wrap { flex:1; overflow-y:auto; padding:28px; }
  .metrics-header { margin-bottom:24px; }
  .metrics-header h2 { font-size:1.5rem; font-weight:800; color:var(--text-dark); margin-bottom:5px; }
  .metrics-header p  { color:var(--text-mid); font-size:0.9rem; }
  .metrics-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:18px; margin-bottom:28px; }
  .metric-card { background:white; border-radius:22px; padding:26px 20px; border:1.5px solid var(--border); box-shadow:var(--shadow); text-align:center; transition:transform 0.2s; }
  .metric-card:hover { transform:translateY(-4px); }
  .metric-card .m-icon  { font-size:1.9rem; margin-bottom:10px; }
  .metric-card .m-label { font-size:0.7rem; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:var(--text-light); margin-bottom:7px; }
  .metric-card .m-value { font-size:2.4rem; font-weight:900; line-height:1; background:linear-gradient(135deg,var(--pink),#ff9a44); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
  .metric-card .m-value.sky { background:linear-gradient(135deg,var(--sky-mid),var(--sky)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
  .metric-card .m-unit  { font-size:0.9rem; font-weight:600; color:var(--text-light); margin-top:3px; }
  .metrics-history { background:white; border-radius:22px; padding:24px; border:1.5px solid var(--border); box-shadow:var(--shadow); }
  .metrics-history h3 { font-size:1rem; font-weight:700; color:var(--text-dark); margin-bottom:14px; }
  .history-table { width:100%; border-collapse:collapse; }
  .history-table th { text-align:left; font-size:0.7rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:var(--text-light); padding:7px 11px; border-bottom:1.5px solid var(--border); }
  .history-table td { padding:11px 11px; font-size:0.86rem; font-weight:500; color:var(--text-dark); border-bottom:1px solid var(--border); vertical-align:top; }
  .history-table tr:last-child td { border-bottom:none; }
  .no-data { text-align:center; color:var(--text-light); padding:36px; font-size:0.88rem; }

  /* ======================== PAGE 4 — ABOUT TEAM ======================== */
  #page-about {
    flex-direction:column; align-items:center;
    min-height:100vh; padding:60px 24px 80px;
    background:linear-gradient(135deg,#fff5fa 0%,#fffef0 50%,#f0f9ff 100%);
    position:relative;
  }
  .about-back { position:absolute; top:28px; left:32px; z-index:10; cursor:pointer; background:white; border:1.5px solid var(--border); color:var(--text-mid); padding:10px 22px; border-radius:50px; font-size:0.86rem; font-weight:600; font-family:'Inter',sans-serif; transition:all 0.2s; box-shadow:0 3px 12px rgba(0,0,0,0.05); }
  .about-back:hover { border-color:var(--pink-light); color:var(--pink); }
  .about-hero { position:relative; z-index:1; text-align:center; margin-bottom:56px; margin-top:20px; }
  .about-hero .badge { margin-bottom:20px; }
  .about-hero h1 { font-size:clamp(2rem,4vw,3.2rem); font-weight:900; color:var(--text-dark); margin-bottom:14px; }
  .about-hero h1 span { background:linear-gradient(135deg,var(--pink),#ff9a44); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
  .about-hero p { color:var(--text-mid); font-size:1rem; line-height:1.75; max-width:560px; margin:auto; }
  .team-grid { display:flex; gap:28px; justify-content:center; flex-wrap:wrap; position:relative; z-index:1; }
  .team-card { background:white; border-radius:28px; padding:44px 36px; border:1.5px solid var(--border); box-shadow:var(--shadow-lg); width:320px; text-align:center; transition:all 0.35s ease; }
  .team-card:hover { transform:translateY(-8px); box-shadow:0 28px 70px rgba(255,107,157,0.18); }
  .avatar { width:110px; height:110px; border-radius:50%; margin:0 auto 20px; border:4px solid var(--pink-light); box-shadow:0 8px 28px rgba(255,107,157,0.22); }
  .team-card h3 { font-size:1.35rem; font-weight:800; color:var(--text-dark); margin-bottom:6px; }
  .team-card .role { font-size:0.76rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:var(--pink); margin-bottom:10px; }
  .team-card .bio  { font-size:0.88rem; color:var(--text-mid); line-height:1.6; margin-bottom:24px; }
  .team-links { display:flex; gap:12px; justify-content:center; }
  .team-link { padding:10px 24px; border-radius:12px; font-size:0.84rem; font-weight:700; text-decoration:none; transition:all 0.2s; border:1.5px solid var(--border); color:var(--text-dark); }
  .team-link.github:hover   { border-color:#1a1a1a; background:#1a1a1a; color:white; }
  .team-link.linkedin        { color:#0a66c2; }
  .team-link.linkedin:hover  { border-color:#0a66c2; background:#0a66c2; color:white; }

  .project-strip {
    margin-top: 64px; position:relative; z-index:1;
    background:white; border:1.5px solid var(--border);
    border-radius:28px; padding:40px 48px;
    max-width:720px; width:100%;
    box-shadow:var(--shadow);
    text-align:center;
  }
  .project-strip h3 { font-size:1.25rem; font-weight:800; color:var(--text-dark); margin-bottom:16px; }
  .tech-pills { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
  .tech-pill { background:linear-gradient(135deg,var(--pink-soft),var(--yellow-light)); border:1.5px solid var(--pink-light); color:var(--text-dark); padding:8px 18px; border-radius:50px; font-size:0.82rem; font-weight:700; }
</style>
</head>
<body>

<!-- ======================================================= -->
<!-- PAGE 1: LANDING                                          -->
<!-- ======================================================= -->
<div id="page-landing" class="page active">
  <div class="blob-wrap">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
    <div class="pulse-ring"></div>
    <div class="pulse-ring"></div>
    <div class="pulse-ring"></div>
  </div>

  <div class="landing-content">
    <div class="badge"><div class="badge-dot"></div>RAG-Powered · Live AI · Medical Grade</div>
    <h1 class="hero-title">MedRAG<br>Doctor Assistant</h1>
    <p class="hero-desc">An intelligent assistant for medical professionals. Upload prescriptions or clinical documents and get instant, AI-powered answers — grounded strictly in your document with zero hallucinations.</p>
    <div class="hero-features">
      <div class="feature-pill"><span>🧠</span> RAG Pipeline</div>
      <div class="feature-pill"><span>⚡</span> Low Latency</div>
      <div class="feature-pill"><span>🔒</span> Grounded Answers</div>
      <div class="feature-pill"><span>💊</span> Medical AI</div>
    </div>
    <button class="btn-primary" onclick="goToPage('page-upload')">Let's Start &nbsp;→</button>
    <div style="margin-top:20px;">
      <button onclick="goToPage('page-about')" style="background:none;border:none;cursor:pointer;color:var(--text-light);font-size:0.88rem;font-weight:600;font-family:'Inter',sans-serif;text-decoration:underline;text-underline-offset:3px;">👥 Meet the Team</button>
    </div>
    <div class="scroll-hint"><span>scroll to explore</span><span style="font-size:1.1rem;">↓</span></div>
  </div>
</div>

<!-- ======================================================= -->
<!-- PAGE 2: UPLOAD                                           -->
<!-- ======================================================= -->
<div id="page-upload" class="page" style="flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(135deg,#fff9f0 0%,#fff5fa 50%,#f5f5ff 100%);min-height:100vh;padding:60px 20px;position:relative;">
  <div class="blob-wrap" style="opacity:0.45;">
    <div class="blob blob-1" style="opacity:0.10;"></div>
    <div class="blob blob-2" style="opacity:0.09;"></div>
  </div>

  <div class="upload-card">
    <div class="upload-icon-wrap">📄</div>
    <h1>Upload Your Document</h1>
    <p>Upload a prescription or medical document (.txt). It will be indexed into the RAG pipeline and become the AI knowledge base.</p>

    <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept=".txt" style="display:none" onchange="handleUpload(event)">
      <div class="dz-icon">📋</div>
      <h3>Drop document here or click to browse</h3>
      <small>Supports .txt · Prescription, clinical notes, patient records</small>
    </div>

    <div id="uploadStatus"></div>
    <div style="margin-top:20px;display:flex;gap:12px;justify-content:center;">
      <button class="back-btn" onclick="goToPage('page-landing')">← Back to Home</button>
      <button class="back-btn" onclick="goToPage('page-about')">👥 About Team</button>
    </div>
  </div>
</div>

<!-- ======================================================= -->
<!-- PAGE 3: CHAT                                             -->
<!-- ======================================================= -->
<div id="page-chat" class="page">
  <!-- SIDEBAR -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon">🩺</div>
      <div>
        <div class="logo-text">MedRAG</div>
        <div class="logo-sub">Doctor Assistant</div>
      </div>
    </div>

    <div class="sidebar-section-title">Navigation</div>
    <button class="nav-link active" data-panel="chat-panel" onclick="switchPanel(this, 'RAG Question & Answer')">
      <div class="nav-icon">💬</div>Ask RAG
    </button>
    <button class="nav-link" data-panel="metrics-panel" onclick="switchPanel(this, 'Metrics Analysis')">
      <div class="nav-icon">📊</div>Metrics Analysis
    </button>

    <div class="sidebar-section-title">More</div>
    <button class="nav-link" onclick="goToPage('page-about')">
      <div class="nav-icon">👥</div>About Team
    </button>
    <button class="nav-link" onclick="goToPage('page-upload')" style="color:var(--text-light);font-weight:500;">
      <div class="nav-icon" style="background:rgba(135,206,235,0.15);">↑</div>Upload New Doc
    </button>

    <div class="sidebar-doc-info" id="docInfoBox" style="display:none;">
      <div class="doc-label">📁 Active Document</div>
      <div class="doc-name" id="docNameDisplay">—</div>
      <div class="doc-chunks" id="docChunksDisplay"></div>
    </div>
  </aside>

  <!-- MAIN PANEL -->
  <div class="main-panel">
    <div class="top-bar">
      <span class="top-bar-title" id="panelTitle">RAG Question & Answer</span>
      <div class="top-bar-status">
        <div class="status-dot"></div>RAG Engine Online
      </div>
    </div>

    <!-- CHAT PANEL -->
    <div id="chat-panel" class="panel-section active" style="flex:1;overflow:hidden;flex-direction:column;display:flex;">
      <div class="chat-messages" id="chatMessages">
        <div class="chat-welcome" id="chatWelcome">
          <div class="welcome-icon">🩺</div>
          <h3>Ready to assist you</h3>
          <p>Your document is loaded into the RAG pipeline. Ask any clinical question and get precise, grounded answers.</p>
          <div class="suggest-chips">
            <div class="chip" onclick="useChip(this)">What medication is prescribed?</div>
            <div class="chip" onclick="useChip(this)">What is the recommended dosage?</div>
            <div class="chip" onclick="useChip(this)">Are there any drug interactions?</div>
            <div class="chip" onclick="useChip(this)">What is the patient's diagnosis?</div>
          </div>
        </div>
      </div>
      <div class="chat-input-area">
        <div class="chat-input-wrap">
          <textarea id="chatInput" placeholder="Ask a clinical question about the document…" rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        </div>
        <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
      </div>
    </div>

    <!-- METRICS PANEL -->
    <div id="metrics-panel" class="panel-section">
      <div class="metrics-wrap">
        <div class="metrics-header">
          <h2>📊 Metrics Analysis</h2>
          <p>Live performance telemetry from your RAG pipeline queries.</p>
        </div>
        <div class="metrics-grid">
          <div class="metric-card"><div class="m-icon">⚡</div><div class="m-label">Avg Latency</div><div class="m-value" id="mAvgLatency">--</div><div class="m-unit">ms</div></div>
          <div class="metric-card"><div class="m-icon">🔥</div><div class="m-label">Last Latency</div><div class="m-value" id="mLastLatency">--</div><div class="m-unit">ms</div></div>
          <div class="metric-card"><div class="m-icon">🚀</div><div class="m-label">Token Speed</div><div class="m-value sky" id="mTPS">--</div><div class="m-unit">tokens/sec</div></div>
          <div class="metric-card"><div class="m-icon">🧮</div><div class="m-label">Total Tokens</div><div class="m-value sky" id="mTokens">--</div><div class="m-unit">tokens</div></div>
          <div class="metric-card"><div class="m-icon">🎯</div><div class="m-label">Avg Confidence</div><div class="m-value" id="mConfidence">--</div><div class="m-unit">%</div></div>
          <div class="metric-card"><div class="m-icon">💬</div><div class="m-label">Queries Asked</div><div class="m-value" id="mQueries">0</div><div class="m-unit">total</div></div>
        </div>
        <div class="metrics-history">
          <h3>Query History Log</h3>
          <table class="history-table">
            <thead><tr><th>#</th><th>Question</th><th>Latency</th><th>Tokens</th><th>Confidence</th></tr></thead>
            <tbody id="metricsTableBody">
              <tr><td colspan="5" class="no-data">No queries yet. Ask something in the Chat tab!</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div><!-- /main-panel -->
</div><!-- /page-chat -->

<!-- ======================================================= -->
<!-- PAGE 4: ABOUT TEAM (FULL PAGE)                          -->
<!-- ======================================================= -->
<div id="page-about" class="page" style="background:linear-gradient(135deg,#fff5fa 0%,#fffef0 55%,#f0f9ff 100%);flex-direction:column;align-items:center;min-height:100vh;position:relative;padding:70px 24px 80px;">
  <div class="blob-wrap" style="opacity:0.6;">
    <div class="blob blob-1" style="opacity:0.13;"></div>
    <div class="blob blob-2" style="opacity:0.11;"></div>
    <div class="blob blob-3" style="opacity:0.10;"></div>
  </div>

  <button class="about-back" id="aboutBackBtn" onclick="goBackFromAbout()">← Go Back</button>

  <div class="about-hero">
    <div class="badge"><div class="badge-dot"></div>Development Team</div>
    <h1>Meet the <span>Builders</span></h1>
    <p>Passionate AI engineers building production-grade RAG pipelines for real-world medical document intelligence.</p>
  </div>

  <div class="team-grid">
    <div class="team-card">
      <img class="avatar" src="https://ui-avatars.com/api/?name=Vaibhav+Sharma&background=ff6b9d&color=fff&size=200&bold=true" alt="Vaibhav Sharma">
      <h3>Vaibhav Sharma</h3>
      <div class="role">B.Tech · CSE AI/ML</div>
      <p class="bio">Final-year AI/ML engineer passionate about RAG systems, LLMs, and building intelligent medical applications that make a real difference.</p>
      <div class="team-links">
        <a class="team-link github"   href="https://github.com/7vaibhav31" target="_blank">GitHub</a>
        <a class="team-link linkedin" href="https://www.linkedin.com/in/vaibhav731/" target="_blank">LinkedIn</a>
      </div>
    </div>

    <div class="team-card">
      <img class="avatar" src="https://ui-avatars.com/api/?name=Bhaskar+Mishra&background=5ba3c9&color=fff&size=200&bold=true" alt="Bhaskar Mishra">
      <h3>Bhaskar Mishra</h3>
      <div class="role">3rd Year · B.Tech</div>
      <p class="bio">Backend developer and ML enthusiast focused on building scalable AI systems. Loves working on challenging real-world problems with data and models.</p>
      <div class="team-links">
        <a class="team-link github"   href="https://github.com/Bhaskar7462" target="_blank">GitHub</a>
        <a class="team-link linkedin" href="https://www.linkedin.com/in/kumar-bhaskar-3727162b3" target="_blank">LinkedIn</a>
      </div>
    </div>
  </div>

  <div class="project-strip">
    <h3>🛠️ Built With</h3>
    <div class="tech-pills">
      <div class="tech-pill">🐍 Python + Flask</div>
      <div class="tech-pill">🧠 RAG Pipeline</div>
      <div class="tech-pill">🤖 OpenRouter API</div>
      <div class="tech-pill">📄 TF-IDF Retrieval</div>
      <div class="tech-pill">💅 Vanilla CSS</div>
      <div class="tech-pill">⚡ Low Latency Design</div>
    </div>
  </div>
</div>

<!-- ======================================================= -->
<!-- JAVASCRIPT                                              -->
<!-- ======================================================= -->
<script>
// ----------------------------------------------------------------
// PAGE NAV
// ----------------------------------------------------------------
let previousPage = 'page-landing';

function goToPage(pageId) {
  const current = document.querySelector('.page.active');
  if (current) previousPage = current.id;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(pageId).classList.add('active');
  window.scrollTo(0, 0);
  if (pageId === 'page-chat') setTimeout(() => document.getElementById('chatInput').focus(), 350);
}

function goBackFromAbout() {
  goToPage(previousPage || 'page-landing');
}

// ----------------------------------------------------------------
// PANEL SWITCH (inside page 3)
// ----------------------------------------------------------------
function switchPanel(btn, title) {
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  btn.classList.add('active');
  const panelId = btn.getAttribute('data-panel');
  document.querySelectorAll('.panel-section').forEach(p => p.classList.remove('active'));
  document.getElementById(panelId).classList.add('active');
  document.getElementById('panelTitle').textContent = title;
}

// ----------------------------------------------------------------
// UPLOAD
// ----------------------------------------------------------------
let loadedDocName = '', loadedChunks = 0;

async function handleUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const statusEl = document.getElementById('uploadStatus');
  statusEl.style.display = 'block';
  statusEl.className = 'loading';
  statusEl.innerHTML = `<div style="display:flex;align-items:center;gap:10px;justify-content:center;"><div class="spinner" style="border-color:rgba(255,107,157,0.3);border-top-color:var(--pink);"></div>&nbsp;Indexing <strong>${file.name}</strong> into RAG pipeline…</div>`;

  const reader = new FileReader();
  reader.onload = async (ev) => {
    try {
      const res  = await fetch('/api/upload', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: ev.target.result }) });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      loadedDocName = file.name;
      loadedChunks  = data.chunks;
      statusEl.className = 'success';
      statusEl.innerHTML = `✅ <strong>${file.name}</strong> indexed into <strong>${data.chunks} context fragments</strong>. Routing to AI assistant…`;
      setTimeout(() => {
        goToPage('page-chat');
        document.getElementById('docInfoBox').style.display = 'block';
        document.getElementById('docNameDisplay').textContent  = loadedDocName;
        document.getElementById('docChunksDisplay').textContent = `${loadedChunks} context chunks`;
      }, 1600);
    } catch (err) {
      statusEl.className = 'error';
      statusEl.innerHTML = `❌ Upload failed: ${err.message}`;
    }
  };
  reader.readAsText(file);
}

// ----------------------------------------------------------------
// CHAT
// ----------------------------------------------------------------
let metricsHistory = [], totalTokensAll = 0, latenciesAll = [], confidencesAll = [];

function useChip(el) {
  document.getElementById('chatInput').value = el.textContent;
  sendMessage();
}
function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }
function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }

function appendMsg(role, text, meta) {
  const welcome = document.getElementById('chatWelcome');
  if (welcome) welcome.remove();
  const msgs = document.getElementById('chatMessages');
  const wrap = document.createElement('div');
  wrap.className = `msg-wrap ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  if (role === 'bot') {
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }
  wrap.appendChild(bubble);
  if (meta) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'msg-meta';
    let html = `<span>${new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span>`;
    if (meta.latency_ms)  html += `<span class="latency-badge">⚡ ${meta.latency_ms}ms</span>`;
    if (meta.chunks_used) html += `<span class="chunks-badge">📄 ${meta.chunks_used} chunks</span>`;
    metaDiv.innerHTML = html;
    wrap.appendChild(metaDiv);
  }
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
}

function showTyping() {
  const msgs = document.getElementById('chatMessages');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap bot'; wrap.id = 'typingIndicator';
  wrap.innerHTML = `<div class="typing-bubble"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
  msgs.appendChild(wrap); msgs.scrollTop = msgs.scrollHeight;
}
function removeTyping() { const t = document.getElementById('typingIndicator'); if(t) t.remove(); }

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const query = input.value.trim();
  if (!query) return;
  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;
  input.value = ''; input.style.height = 'auto';
  appendMsg('user', query);
  showTyping();
  try {
    const res  = await fetch('/api/qa', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ query }) });
    const data = await res.json();
    removeTyping();
    if (data.error) { appendMsg('bot', '⚠️ ' + data.error); }
    else { appendMsg('bot', data.answer, { latency_ms: data.latency_ms, chunks_used: data.chunks_used }); updateMetrics(query, data); }
  } catch (err) {
    removeTyping(); appendMsg('bot', '⚠️ Network error: ' + err.message);
  }
  sendBtn.disabled = false; input.focus();
}

// ----------------------------------------------------------------
// METRICS
// ----------------------------------------------------------------
function updateMetrics(query, data) {
  latenciesAll.push(data.latency_ms);
  confidencesAll.push(data.confidence);
  totalTokensAll += data.total_tokens || 0;
  metricsHistory.push({ query, ...data });
  const avg = arr => arr.length ? Math.round(arr.reduce((s,v) => s+v, 0) / arr.length) : '--';
  document.getElementById('mAvgLatency').textContent  = avg(latenciesAll);
  document.getElementById('mLastLatency').textContent = data.latency_ms || '--';
  document.getElementById('mTPS').textContent         = data.tokens_per_sec || '--';
  document.getElementById('mTokens').textContent      = totalTokensAll;
  document.getElementById('mConfidence').textContent  = avg(confidencesAll);
  document.getElementById('mQueries').textContent     = metricsHistory.length;
  const tbody = document.getElementById('metricsTableBody');
  tbody.innerHTML = '';
  metricsHistory.slice().reverse().forEach((row, i) => {
    const n = metricsHistory.length - i;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--text-light);font-weight:700;">${n}</td>
      <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${row.query.length>58 ? row.query.slice(0,58)+'…' : row.query}</td>
      <td><span class="latency-badge">⚡ ${row.latency_ms}ms</span></td>
      <td>${row.total_tokens || '--'}</td>
      <td><span class="chunks-badge">${row.confidence || '--'}%</span></td>`;
    tbody.appendChild(tr);
  });
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
