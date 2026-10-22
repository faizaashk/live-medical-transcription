/* Live Medical Transcription - frontend
 * mic -> Silero VAD (browser) -> speech segment (WAV) -> /api/transcribe
 * full transcript -> /api/summarize -> structured summary panel
 */
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------ state ----
const state = {
  micVad: null,
  recording: false,
  segments: [],        // array of transcribed strings
  pending: 0,          // chunks still being transcribed
  summary: null,       // last structured summary
  processing: false,
};

const SAMPLE_TEXT = `Doctor: Good morning, what brings you in today? Patient: Good morning doctor, my name is Tushar, I'm thirty years old, male. I've had a bad headache and fever for the last three days. Doctor: Did it start suddenly? Patient: Yes, it started on Sunday night. It's getting worse in the evening and light makes it worse. Paracetamol helps a little. Doctor: Any vomiting or neck stiffness? Patient: No vomiting and no neck stiffness, but I have body aches and a runny nose. Doctor: Any past medical problems? Patient: I have mild asthma since childhood, and I had my appendix removed in 2015. Doctor: Are you taking any medicines and do you have allergies? Patient: I use a salbutamol inhaler when needed, and I'm allergic to penicillin. Doctor: Let me check. Temperature is 101 degrees Fahrenheit, pulse 92, blood pressure 120 over 80. Throat is mildly red, chest is clear. I think this is a viral fever, possibly influenza. I'll order a CBC and a dengue NS1 test. Take paracetamol 500 milligrams three times a day for three days, drink plenty of fluids, and come back in three days if the fever persists.`;

// --------------------------------------------------------- helpers ----
const fullText = () => state.segments.join(" ").trim();
const wordCount = (t) => (t ? t.split(/\s+/).length : 0);

function setStatus(label, st) {
  $("status").textContent = label;
  $("status").dataset.state = st;
}
function showError(msg) {
  $("error").textContent = msg || "";
  $("error").hidden = !msg;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const list = (a) => (a && a.length ? a : []);

function render() {
  // transcript box
  const box = $("transcript");
  if (!state.segments.length && !state.pending) {
    box.innerHTML = '<p class="placeholder">Ready. Click Start mic and speak naturally. Voice activity detection will split your speech into chunks automatically.</p>';
  } else {
    box.innerHTML =
      state.segments.map((s) => `<p>${esc(s)}</p>`).join("") +
      (state.pending ? '<p class="pending">Transcribing…</p>' : "");
    box.scrollTop = box.scrollHeight;
  }
  $("words").textContent = wordCount(fullText());
  $("chunks").textContent = state.segments.length;

  // buttons
  $("btn-start").disabled = state.recording;
  $("btn-stop").disabled = !state.recording;
  $("btn-process").disabled = state.processing || wordCount(fullText()) < 5;
  $("btn-process").textContent = state.processing ? "Analysing…" : "Process transcript with AI";
  $("btn-copy").disabled = !state.summary;
}

// ------------------------------------------------ audio + VAD + ASR ----
async function startMic() {
  showError("");
  try {
    setStatus("Loading VAD…", "working");
    state.micVad = await vad.MicVAD.new({
      baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
      onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
      onSpeechStart: () => setStatus("Hearing speech", "speech"),
      onFrameProcessed: (probs) => {
        $("level-bar").style.width = Math.round(probs.isSpeech * 100) + "%";
      },
      onVADMisfire: () => setStatus("Listening…", "listening"),
      onSpeechEnd: (audio) => {
        setStatus("Listening…", "listening");
        sendSegment(audio);
      },
    });
    state.micVad.start();
    state.recording = true;
    setStatus("Listening…", "listening");
  } catch (err) {
    console.error(err);
    setStatus("Idle", "idle");
    showError(
      err && err.name === "NotAllowedError"
        ? "Microphone access was blocked. Allow it in the browser address bar and try again."
        : "Could not start the microphone: " + (err.message || err)
    );
  }
  render();
}

function stopMic() {
  if (state.micVad) { state.micVad.pause(); state.micVad.destroy?.(); state.micVad = null; }
  state.recording = false;
  $("level-bar").style.width = "0";
  setStatus("Idle", "idle");
  render();
}

async function sendSegment(float32Audio) {
  // Only speech segments reach this point, so silence/noise never hits the ASR.
  const wav = vad.utils.encodeWAV(float32Audio);        // 16 kHz mono WAV
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "chunk.wav");

  const slot = state.segments.length + state.pending; // keep chunks in order
  state.pending++; render();
  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { text } = await res.json();
    if (text) state.segments.push(text);
  } catch (err) {
    showError("Transcription failed: " + err.message + ". Is the backend running?");
  } finally {
    state.pending--; render();
  }
}

// -------------------------------------------------------- LLM step ----
async function processTranscript() {
  const text = fullText();
  if (wordCount(text) < 5) return;
  if (state.recording) stopMic();           // session is complete
  state.processing = true; showError(""); render();
  setStatus("Analysing…", "working");
  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.summary = data;
    renderSummary(data);
  } catch (err) {
    showError("Summary failed: " + err.message);
  } finally {
    state.processing = false; setStatus("Idle", "idle"); render();
  }
}

// ------------------------------------------------- summary renderer ----
const tags = (arr, cls = "") => list(arr).map((x) => `<span class="tag ${cls}">${esc(x)}</span>`).join("");
function setSection(id, html) {
  const el = $(id);
  if (html) { el.className = ""; el.innerHTML = html; }
  else { el.className = "empty"; }
}

function renderSummary(d) {
  const p = d.patient || {};
  $("p-name").textContent = p.name || "—";
  $("p-age").textContent = p.age || "—";
  $("p-sex").textContent = p.sex || "—";
  $("p-ids").textContent = list(p.identifiers).join(", ");
  $("s-summary").value = d.clinical_summary || "";

  setSection("s-cc", d.chief_complaint ? `<p>${esc(d.chief_complaint)}</p>` : "");

  const h = d.hpi || {};
  const hpiRows = [["Onset", h.onset], ["Duration", h.duration], ["Progression", h.progression],
    ["Aggravating", list(h.aggravating).join(", ")], ["Relieving", list(h.relieving).join(", ")]]
    .filter(([, v]) => v);
  setSection("s-hpi", hpiRows.length ? `<dl class="kv">${hpiRows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>` : "");

  const s = d.symptoms || {};
  setSection("s-symptoms",
    list(s.positive).length || list(s.negative).length
      ? (list(s.positive).length ? `<div class="sub">Present</div><div class="tags">${tags(s.positive)}</div>` : "") +
        (list(s.negative).length ? `<div class="sub">Denied</div><div class="tags">${tags(s.negative, "neg")}</div>` : "")
      : "");

  const pm = d.past_medical_history || {};
  const pmRows = [["Conditions", pm.conditions], ["Surgeries", pm.surgeries], ["Hospitalisations", pm.hospitalisations]]
    .filter(([, v]) => list(v).length);
  setSection("s-pmh", pmRows.length ? `<dl class="kv">${pmRows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v.join(", "))}</dd>`).join("")}</dl>` : "");

  const m = d.medications || {};
  const meds = list(m.current).map((x) => `<li>${esc([x.name, x.dose, x.adherence].filter(Boolean).join(" — "))}</li>`).join("");
  setSection("s-meds",
    meds || list(m.allergies).length
      ? (meds ? `<ul class="plain">${meds}</ul>` : "") +
        (list(m.allergies).length ? `<div class="sub">Allergies</div><div class="tags">${tags(m.allergies, "neg")}</div>` : "")
      : "");

  const o = d.observations || {};
  setSection("s-obs",
    list(o.vitals).length || list(o.exam_findings).length
      ? (list(o.vitals).length ? `<div class="sub">Vitals</div><div class="tags">${tags(o.vitals)}</div>` : "") +
        (list(o.exam_findings).length ? `<div class="sub">Examination</div><ul class="plain">${o.exam_findings.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "")
      : "");

  const a = d.assessment || {};
  setSection("s-assess",
    a.provisional || list(a.differentials).length
      ? (a.provisional ? `<p><b>${esc(a.provisional)}</b></p>` : "") +
        (list(a.differentials).length ? `<div class="sub">Differentials</div><div class="tags">${tags(a.differentials)}</div>` : "")
      : "");

  const pl = d.plan || {};
  const planRows = [["Investigations", list(pl.investigations).join(", ")], ["Prescriptions", list(pl.prescriptions).join("; ")],
    ["Advice", list(pl.advice).join("; ")], ["Follow-up", pl.follow_up]].filter(([, v]) => v);
  setSection("s-plan", planRows.length ? `<dl class="kv">${planRows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>` : "");
}

function clearSummary() {
  state.summary = null;
  ["p-name", "p-age", "p-sex"].forEach((id) => ($(id).textContent = "—"));
  $("p-ids").textContent = ""; $("s-summary").value = "";
  renderSummary({});
  render();
}

// ------------------------------------------------ copy + save + misc ----
function summaryAsText() {
  const d = state.summary; if (!d) return "";
  const L = [];
  const add = (t, v) => { if (v && String(v).length) L.push(`${t}: ${v}`); };
  const p = d.patient || {};
  add("Patient", [p.name, p.age && `${p.age} yrs`, p.sex].filter(Boolean).join(", "));
  add("Summary", $("s-summary").value);              // includes any manual edits
  add("Chief complaint", d.chief_complaint);
  const h = d.hpi || {};
  add("HPI", [h.onset && `onset ${h.onset}`, h.duration && `duration ${h.duration}`, h.progression,
    list(h.aggravating).length && `worse with ${h.aggravating.join(", ")}`,
    list(h.relieving).length && `better with ${h.relieving.join(", ")}`].filter(Boolean).join("; "));
  add("Symptoms", list(d.symptoms?.positive).join(", "));
  add("Denies", list(d.symptoms?.negative).join(", "));
  add("PMH", [...list(d.past_medical_history?.conditions), ...list(d.past_medical_history?.surgeries), ...list(d.past_medical_history?.hospitalisations)].join(", "));
  add("Medications", list(d.medications?.current).map((x) => [x.name, x.dose, x.adherence].filter(Boolean).join(" ")).join(", "));
  add("Allergies", list(d.medications?.allergies).join(", "));
  add("Vitals", list(d.observations?.vitals).join(", "));
  add("Exam", list(d.observations?.exam_findings).join("; "));
  add("Assessment", [d.assessment?.provisional, ...list(d.assessment?.differentials)].filter(Boolean).join("; "));
  add("Investigations", list(d.plan?.investigations).join(", "));
  add("Rx", list(d.plan?.prescriptions).join("; "));
  add("Advice", list(d.plan?.advice).join("; "));
  add("Follow-up", d.plan?.follow_up);
  return L.join("\n");
}

async function copySummary() {
  try { await navigator.clipboard.writeText(summaryAsText()); $("btn-copy").textContent = "Copied"; }
  catch { showError("Could not copy. Select the text manually."); }
  setTimeout(() => ($("btn-copy").textContent = "Copy to prescription"), 1500);
}

function saveRecording() {
  if (!fullText()) return showError("Nothing to save yet.");
  const blob = new Blob([JSON.stringify({ saved_at: new Date().toISOString(), transcript: fullText(), summary: state.summary }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `consultation-${Date.now()}.json`;
  a.click(); URL.revokeObjectURL(a.href);
}

function clearAll() {
  if (state.recording) stopMic();
  state.segments = []; state.pending = 0; showError(""); render();
}

// ------------------------------------------------------------ wiring ----
$("btn-start").onclick = startMic;
$("btn-stop").onclick = stopMic;
$("btn-clear").onclick = clearAll;
$("btn-save").onclick = saveRecording;
$("btn-sample").onclick = () => { state.segments = [SAMPLE_TEXT]; render(); };
$("btn-process").onclick = processTranscript;
$("btn-clear-summary").onclick = clearSummary;
$("btn-copy").onclick = copySummary;
render();
