/* Live Medical Transcription - frontend
 * mic -> Silero VAD (browser) -> speech segment (WAV) -> /api/transcribe
 * While you are still speaking, the audio so far is re-transcribed every ~1.2 s
 * and shown as a grey "live" line, so text keeps appearing as you talk.
 * When you pause, the final text replaces the live line.
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
  // live (interim) transcription while the user is speaking
  liveText: "",        // text shown in grey while speech is in progress
  frames: [],          // audio frames of the current utterance
  preBuf: [],          // last few frames before speech starts (so words aren't clipped)
  speaking: false,
  liveTimer: null,
  liveBusy: false,
  utt: 0,              // utterance counter, used to drop stale live results
  stream: null, ctx: null, proc: null,   // our own audio capture (feeds the live text)
};
const LIVE_EVERY_MS = 1200;

// Sample consultation: the patient speaks alone (no doctor).
const SAMPLE_TEXT = `Hello, my name is Ayesha Khan, I'm twenty eight years old and I'm a woman. For the past five days I've had a sore throat and a dry cough that keeps me awake at night. It started after I got caught in the rain last weekend. The cough gets worse when I lie down and a bit better when I drink warm water with honey. I also have a mild fever in the evenings, but I haven't checked it with a thermometer. I don't have any chest pain and I'm not short of breath, and I haven't lost my sense of smell. I had typhoid when I was ten and I was admitted to hospital for a week. I have no other conditions and I've never had surgery. I'm taking cetirizine at night when my nose is blocked, and I took paracetamol twice yesterday, but I'm not sure of the dose. I'm allergic to sulfa drugs, they gave me a rash once.`;

// --------------------------------------------------------- helpers ----
async function errorText(res) {
  // Server errors may be JSON ({detail}) or plain text ("Internal Server Error").
  const body = await res.text();
  try { return JSON.parse(body).detail || res.statusText; }
  catch { return `${res.status} ${body || res.statusText} (see the uvicorn terminal for details)`; }
}
const fullText = () => state.segments.filter(Boolean).join(" ").trim();
const finalSegments = () => state.segments.filter(Boolean);
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
  const finals = finalSegments();
  if (!finals.length && !state.pending && !state.liveText) {
    box.innerHTML = '<p class="placeholder">Ready. Click Start mic and speak naturally. Your words appear live as you talk; a pause finalises each chunk.</p>';
  } else {
    box.innerHTML =
      finals.map((t) => `<p>${esc(t)}</p>`).join("") +
      (state.liveText
        ? `<p class="pending">${esc(state.liveText)} …</p>`
        : state.pending ? '<p class="pending">Transcribing…</p>' : "");
    box.scrollTop = box.scrollHeight;
  }
  $("words").textContent = wordCount(fullText());
  $("chunks").textContent = finalSegments().length;

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
    await startCapture();
    state.micVad = await vad.MicVAD.new({
      baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
      onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
      redemptionMs: 900,      // pause length that ends a chunk (shorter = chunks finalise sooner)
      preSpeechPadMs: 500,
      onSpeechStart: () => {
        setStatus("Hearing speech", "speech");
        state.speaking = true;
        state.utt++;
        state.frames = state.preBuf.slice();
        clearInterval(state.liveTimer);
        state.liveTimer = setInterval(sendInterim, LIVE_EVERY_MS);
      },
      onFrameProcessed: (probs) => {
        $("level-bar").style.width = Math.round(probs.isSpeech * 100) + "%";
      },
      onVADMisfire: () => {
        setStatus("Listening…", "listening");
        endLive(true);
      },
      onSpeechEnd: (audio) => {
        setStatus("Listening…", "listening");
        endLive(false);          // live text stays on screen until the final text arrives
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
  stopCapture();
  if (state.micVad) { state.micVad.pause(); state.micVad.destroy?.(); state.micVad = null; }
  state.recording = false;
  endLive(true);
  state.preBuf = [];
  $("level-bar").style.width = "0";
  setStatus("Idle", "idle");
  render();
}

// Our own 16 kHz mic capture. It keeps the raw audio so we can show live text
// while you are still speaking (the VAD alone only hands over audio after a pause).
async function startCapture() {
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  state.ctx = new AudioContext({ sampleRate: 16000 });
  await state.ctx.resume();
  const src = state.ctx.createMediaStreamSource(state.stream);
  state.proc = state.ctx.createScriptProcessor(2048, 1, 1);   // ~128 ms per chunk
  state.proc.onaudioprocess = (e) => {
    const chunk = new Float32Array(e.inputBuffer.getChannelData(0));   // copy
    if (state.speaking) state.frames.push(chunk);
    else { state.preBuf.push(chunk); if (state.preBuf.length > 8) state.preBuf.shift(); }
  };
  src.connect(state.proc);
  state.proc.connect(state.ctx.destination);   // needed for the callback to fire; output is silent
}

function stopCapture() {
  try { state.proc && state.proc.disconnect(); } catch {}
  try { state.stream && state.stream.getTracks().forEach((t) => t.stop()); } catch {}
  try { state.ctx && state.ctx.close(); } catch {}
  state.proc = state.stream = state.ctx = null;
}

function endLive(clearText) {
  state.speaking = false;
  state.utt++;                       // any in-flight live result is now stale
  clearInterval(state.liveTimer);
  state.liveTimer = null;
  state.frames = [];
  if (clearText) { state.liveText = ""; render(); }
}

function concatFrames(frames) {
  const total = frames.reduce((n, f) => n + f.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const f of frames) { out.set(f, off); off += f.length; }
  return out;
}

function wavForm(float32Audio) {
  const wav = vad.utils.encodeWAV(float32Audio);        // 16 kHz mono WAV
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "chunk.wav");
  return form;
}

// Live text: re-transcribe everything said so far in the current utterance.
async function sendInterim() {
  if (!state.speaking || state.liveBusy || state.frames.length < 5) return;
  const myUtt = state.utt;
  state.liveBusy = true;
  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: wavForm(concatFrames(state.frames)) });
    if (!res.ok) throw new Error(await errorText(res));
    const { text } = await res.json();
    if (text && myUtt === state.utt && state.speaking) { state.liveText = text; render(); }
  } catch (err) {
    console.warn("Live transcription skipped:", err.message);   // final text will still arrive
  } finally {
    state.liveBusy = false;
  }
}

async function sendSegment(float32Audio) {
  // Only speech segments reach this point, so silence/noise never hits the ASR.
  const slot = state.segments.length;   // reserve a slot so chunks stay in order
  state.segments.push("");
  state.pending++; render();
  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: wavForm(float32Audio) });
    if (!res.ok) throw new Error(await errorText(res));
    const { text } = await res.json();
    state.segments[slot] = text || "";
  } catch (err) {
    showError("Transcription failed: " + err.message + ". Is the backend running?");
  } finally {
    state.pending--;
    if (!state.speaking) state.liveText = "";   // final text replaces the live line
    render();
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
    if (!res.ok) throw new Error(await errorText(res));
    const data = await res.json();
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
  state.segments = []; state.pending = 0; state.liveText = ""; showError(""); render();
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
