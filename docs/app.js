const API_URL = "https://student-sucess.onrender.com";

const NAMES = {attendance_rate:"Attendance", avg_quiz_score:"Period 1 score", assignment_attempts:"Study time band",
  time_spent_minutes:"Study minutes", prev_term_gpa:"Previous GPA", topic_algebra_score:"Algebra score",
  topic_calculus_score:"Calculus score", topic_stats_score:"Statistics score", late_submissions:"Past failures",
  forum_participation:"Support received", engagement_index:"Engagement index"};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = s => document.querySelector(s);
const tone = r => r >= 70 ? "hi" : r >= 40 ? "md" : "lo";
const color = r => r >= 70 ? css("--risk") : r >= 40 ? css("--warn") : css("--safe");
let students = [];
Chart.defaults.font.family = "Instrument Sans, system-ui, sans-serif";
Chart.defaults.color = css("--muted");

const load = f => fetch(`data/${f}.json`).then(r => r.json());

async function init() {
  try {
    const [stats, st, metrics, imp] = await Promise.all([load("stats"), load("students"), load("metrics"), load("importance")]);
    students = st;
    renderStats(stats); renderTable(); renderModel(metrics, imp);
  } catch (e) {
    $("#overview").insertAdjacentHTML("beforeend", `<p class="err">Could not load data files. Open the site through a web server (or GitHub Pages), not by double-clicking index.html.</p>`);
  }
}

function renderStats(s) {
  const items = [[s.total, "Students"], [s.predicted_at_risk, "Predicted at risk (50%+)"],
    [(s.at_risk_rate * 100).toFixed(1) + "%", "Actually failed (G3 under 10)"], [s.avg_g3.toFixed(1), "Average final grade (of 20)"],
    [s.avg_absences.toFixed(1), "Average absences"]];
  $("#stats").innerHTML = items.map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  new Chart($("#distChart"), {type: "bar", data: {labels: s.risk_distribution.labels,
    datasets: [{data: s.risk_distribution.counts, backgroundColor: [css("--safe"), css("--safe"), css("--warn"), css("--risk"), css("--risk")]}]},
    options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {title: {display: true, text: "Students"}}, x: {title: {display: true, text: "Predicted risk"}}}}});
  const f = s.risk_by_failures;
  new Chart($("#failChart"), {type: "bar", data: {labels: Object.keys(f).map(k => k + " failures"),
    datasets: [{data: Object.values(f).map(v => +(v * 100).toFixed(1)), backgroundColor: css("--ink")}]},
    options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {max: 100, title: {display: true, text: "% at risk"}}}}});
}

function renderTable() {
  const q = $("#search").value.trim().toUpperCase(), onlyRisk = $("#filter").value === "risk";
  const rows = students.filter(s => (!onlyRisk || s.risk >= 50) && s.id.includes(q));
  $("#tbl tbody").innerHTML = rows.map(s => `<tr><td>${s.id}</td>
    <td><span class="bar"><i style="width:${s.risk}%;background:${color(s.risk)}"></i></span><b class="${tone(s.risk)}">${s.risk}%</b></td>
    <td>${s.G1}</td><td>${s.G2}</td><td>${s.failures}</td><td>${s.absences}</td><td>${NAMES[s.top_factor] || s.top_factor}</td></tr>`).join("")
    || `<tr><td colspan="7">No students match. Clear the search or switch the filter to All students.</td></tr>`;
  $("#count").textContent = `Showing ${Math.min(rows.length, 200)} of ${rows.length} students. Scores come from the model applied to every student, including the ones it was trained on.`;
}
$("#search").addEventListener("input", renderTable);
$("#filter").addEventListener("change", renderTable);

function renderModel(m, imp) {
  const names = ["RandomForest", "XGBoost"], keys = ["accuracy", "roc_auc", "precision", "recall", "f1"];
  $("#best").textContent = `Best model by ROC-AUC: ${m.best_model}. Evaluated on ${m.test_size} held-out students (25% split).`;
  new Chart($("#cmpChart"), {type: "bar", data: {labels: ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"],
    datasets: names.map((n, i) => ({label: n, data: keys.map(k => +m[n][k].toFixed(3)), backgroundColor: i ? css("--warn") : css("--safe")}))},
    options: {maintainAspectRatio: false, scales: {y: {min: 0.6, max: 1}}}});
  const [[tn, fp], [fn, tp]] = m[m.best_model].confusion_matrix;
  const cell = (v, c) => `<div class="c" style="background:${c}22;color:${c}">${v}</div>`;
  $("#cm").innerHTML = `<span></span><span class="l">Predicted safe</span><span class="l">Predicted at risk</span>
    <span class="l">Actually safe</span>${cell(tn, css("--safe"))}${cell(fp, css("--warn"))}
    <span class="l">Actually at risk</span>${cell(fn, css("--risk"))}${cell(tp, css("--safe"))}`;
  const top = imp.shap_mean_abs.slice(0, 8);
  new Chart($("#impChart"), {type: "bar", data: {labels: top.map(d => NAMES[d.feature] || d.feature),
    datasets: [{data: top.map(d => +d.value.toFixed(4)), backgroundColor: css("--ink")}]},
    options: {indexAxis: "y", maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {title: {display: true, text: "Mean absolute SHAP value"}}}}});
}

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = $("#go"), out = $("#result");
  const body = Object.fromEntries(new FormData(e.target));
  ["G1", "G2", "studytime", "failures", "absences"].forEach(k => body[k] = +body[k]);
  btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const res = await fetch(`${API_URL}/predict`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (!res.ok) throw new Error("Check that every value is within its allowed range.");
    showResult(await res.json());
  } catch (err) {
    out.innerHTML = `<p class="err">Could not get a prediction. ${err.message.startsWith("Check") ? err.message : `The prediction server at ${API_URL} is not responding. Start the backend (or wait about a minute if it is waking up on free hosting), then try again.`}</p>`;
  }
  btn.disabled = false; btn.textContent = "Predict risk";
});

function showResult(r) {
  const p = r.risk_percent, c = color(p), arc = 251 * p / 100;
  const max = Math.max(...r.factors.map(f => Math.abs(f.impact)), 1e-6);
  $("#result").innerHTML = `<div class="gauge">
    <svg viewBox="0 0 200 115" role="img" aria-label="Risk ${p} percent"><path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="${css("--line")}" stroke-width="16" stroke-linecap="round"/>
    <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="${c}" stroke-width="16" stroke-linecap="round" stroke-dasharray="${arc} 251"/></svg>
    <div class="pct" style="color:${c}">${p}%</div>
    <span class="badge" style="background:${c}">${r.at_risk ? "At risk" : "Not at risk"}</span></div>
    <h3>Why this score</h3>
    ${r.factors.map(f => `<div class="factor"><span>${NAMES[f.feature] || f.feature}</span>
      <div class="t"><i style="background:${f.impact > 0 ? css("--risk") : css("--safe")};${f.impact > 0 ? "left:50%" : "right:50%"};width:${Math.abs(f.impact) / max * 50}%"></i></div>
      <span>${f.impact > 0 ? "+" : ""}${f.impact.toFixed(2)}</span></div>`).join("")}
    <p class="note">Red bars push the risk up; teal bars push it down. Model: ${r.model}.</p>`;
}
init();
