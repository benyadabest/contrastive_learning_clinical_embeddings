// ============================================================
// Clinical Embedding Studio — frontend script
// ============================================================

// ---- Utilities ----
function $(id) { return document.getElementById(id); }
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

async function postJSON(url, body) {
  const t0 = performance.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const elapsedMs = performance.now() - t0;
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  const data = await res.json();
  return { data, elapsedMs };
}

function loadingHTML(label) {
  return `<div class="loading-row"><span class="spinner" aria-hidden="true"></span> ${escapeHtml(label)}</div>`;
}

function errorHTML(message) {
  return `<div class="error-state"><strong>Error:</strong>&nbsp;${escapeHtml(message)}</div>`;
}

function setMeta(el, parts) {
  el.hidden = false;
  el.innerHTML = parts.map((p) => p).join('<span class="appfoot-sep">·</span>');
}

function setSubmitting(form, submitting) {
  const btn = form.querySelector('button[type="submit"]');
  if (!btn) return;
  btn.disabled = submitting;
}

function similarityClass(s) {
  if (s >= 0.55) return 'sim sim-high';
  if (s >= 0.4) return 'sim sim-mid';
  return 'sim sim-low';
}

// ---- Popovers ----
function closeAllPopovers(except = null) {
  document.querySelectorAll('.popover-trigger[aria-expanded="true"]').forEach((b) => {
    if (b !== except) b.setAttribute('aria-expanded', 'false');
  });
}

document.querySelectorAll('.popover-trigger').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = btn.getAttribute('aria-expanded') === 'true';
    closeAllPopovers(btn);
    btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  });
});

document.addEventListener('click', (e) => {
  // Click anywhere outside a popover closes them; clicks inside a panel are fine.
  if (e.target.closest('.popover-panel')) return;
  closeAllPopovers();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeAllPopovers();
});

// ---- Tabs ----
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    const t = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${t}`));
    closeAllPopovers();
  });
});

// ---- Health / status ----
async function loadHealth() {
  const dot = document.querySelector('#status .status-dot');
  const text = document.querySelector('#status .status-text');
  const footerCorpus = $('footer-corpus');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (!data.ok) {
      dot.dataset.state = 'error';
      text.textContent = `Corpus offline · ${data.reason || 'unknown'}`;
      return;
    }
    dot.dataset.state = 'ok';
    text.textContent = `${data.n_notes.toLocaleString()} notes · ${data.n_patients} patients · model: ${data.model_safe_name} · ${data.has_openai_key ? 'LLM connected' : 'no LLM key'}`;
    if (footerCorpus) footerCorpus.textContent = `corpus: ${data.model_safe_name} · ${data.n_notes.toLocaleString()} notes · ${data.n_patients} patients · ${data.n_categories} categories`;
  } catch (e) {
    dot.dataset.state = 'error';
    text.textContent = `Health check failed: ${e.message}`;
  }
}

// ---- Patient dropdowns ----
async function loadPatients() {
  const cohortSel = $('cohort-patient');
  const trajSel = $('trajectory-patient');
  try {
    const res = await fetch('/api/patients?limit=300');
    if (!res.ok) throw new Error(await res.text());
    const patients = await res.json();
    const opts = patients
      .map((p) => `<option value="${p.subject_id}">Subject ${p.subject_id} · ${p.n_notes} notes</option>`)
      .join('');
    cohortSel.innerHTML = opts;
    trajSel.innerHTML = opts;
  } catch (e) {
    const msg = `<option>error: ${escapeHtml(e.message)}</option>`;
    cohortSel.innerHTML = msg;
    trajSel.innerHTML = msg;
  }
}

// ---- Render helpers ----
function renderSearchTable(rows) {
  if (!rows || !rows.length) return `<div class="empty-state">No matching notes for that query.</div>`;
  const head = `
    <thead><tr>
      <th class="num">#</th>
      <th class="num">Sim</th>
      <th>Patient</th>
      <th>Category</th>
      <th>Date</th>
      <th>ICD-9</th>
      <th>Snippet</th>
    </tr></thead>`;
  const body = rows.map((r) => {
    const icd = (r.icd_codes || []).slice(0, 5).map((c) => `<span class="chip chip-icd">${escapeHtml(c)}</span>`).join('');
    return `<tr>
      <td class="num">${r.rank}</td>
      <td class="num"><span class="${similarityClass(r.similarity)}">${(r.similarity ?? 0).toFixed(3)}</span></td>
      <td class="num">${r.subject_id ?? ''}</td>
      <td>${escapeHtml(r.category)}</td>
      <td>${escapeHtml(r.date)}</td>
      <td>${icd}</td>
      <td class="snippet">${escapeHtml(r.snippet || '')}</td>
    </tr>`;
  }).join('');
  return `<div class="data-card"><table class="data">${head}<tbody>${body}</tbody></table></div>`;
}

function renderCohortTable(rows) {
  if (!rows || !rows.length) return `<div class="empty-state">No similar patients found.</div>`;
  const head = `
    <thead><tr>
      <th class="num">#</th>
      <th class="num">Sim</th>
      <th>Patient</th>
      <th class="num">Shared</th>
      <th class="num">Jaccard</th>
      <th>Shared ICD chapters</th>
    </tr></thead>`;
  const body = rows.map((r) => {
    const ch = (r.shared_chapters || []).map((c) => `<span class="chip chip-chap">${escapeHtml(c)}</span>`).join('');
    return `<tr>
      <td class="num">${r.rank}</td>
      <td class="num"><span class="${similarityClass(r.similarity)}">${(r.similarity ?? 0).toFixed(3)}</span></td>
      <td class="num">${r.subject_id}</td>
      <td class="num">${r.n_shared}</td>
      <td class="num">${(r.jaccard_chapter ?? 0).toFixed(3)}</td>
      <td>${ch || '<span class="chip">none</span>'}</td>
    </tr>`;
  }).join('');
  return `<div class="data-card"><table class="data">${head}<tbody>${body}</tbody></table></div>`;
}

function renderCohortByTextTable(rows) {
  if (!rows || !rows.length) return `<div class="empty-state">No matching patients for that description.</div>`;
  const head = `
    <thead><tr>
      <th class="num">#</th>
      <th class="num">Sim</th>
      <th>Patient</th>
      <th class="num">Notes</th>
      <th class="num">Chapters</th>
      <th>ICD-9 chapters</th>
    </tr></thead>`;
  const body = rows.map((r) => {
    const ch = (r.chapters || []).map((c) => `<span class="chip chip-chap">${escapeHtml(c)}</span>`).join('');
    return `<tr>
      <td class="num">${r.rank}</td>
      <td class="num"><span class="${similarityClass(r.similarity)}">${(r.similarity ?? 0).toFixed(3)}</span></td>
      <td class="num">${r.subject_id}</td>
      <td class="num">${r.n_notes}</td>
      <td class="num">${r.n_chapters}</td>
      <td>${ch || '<span class="chip">none</span>'}</td>
    </tr>`;
  }).join('');
  return `<div class="data-card"><table class="data">${head}<tbody>${body}</tbody></table></div>`;
}

// Tiny markdown subset: **bold**, [#N] / Note #N -> styled inline code, preserve newlines.
function renderMarkdownLite(s) {
  if (!s) return '';
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\bNote #(\d+)\b/g, '<code class="note-ref">Note #$1</code>')
    .replace(/\[#(\d+)\]/g, '<code class="note-ref">#$1</code>');
}

// ---- Search ----
$('search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  closeAllPopovers();
  const query = $('search-query').value;
  const k = +$('search-k').value;
  const snippet_chars = +$('search-snippet').value;
  const out = $('search-result');
  const meta = $('search-meta');
  out.innerHTML = loadingHTML('Encoding query and ranking notes…');
  meta.hidden = true;
  setSubmitting(e.currentTarget, true);
  try {
    const { data, elapsedMs } = await postJSON('/api/search', { query, k, snippet_chars });
    setMeta(meta, [
      `<strong>${data.length}</strong> results`,
      `query: <code>${escapeHtml(query)}</code>`,
      `${Math.round(elapsedMs)} ms`,
    ]);
    out.innerHTML = renderSearchTable(data);
  } catch (err) {
    out.innerHTML = errorHTML(err.message);
  } finally {
    setSubmitting(e.currentTarget, false);
  }
});

// ---- Ask ----
$('ask-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  closeAllPopovers();
  const question = $('ask-question').value;
  const k = +$('ask-k').value;
  const model = $('ask-model').value;
  const mode = $('ask-mode').value;
  const out = $('ask-result');
  const meta = $('ask-meta');
  out.innerHTML = loadingHTML(mode === 'strict' ? 'Retrieving notes (strict mode)…' : 'Synthesizing answer from retrieved notes…');
  meta.hidden = true;
  setSubmitting(e.currentTarget, true);
  try {
    const { data, elapsedMs } = await postJSON('/api/ask', { question, k, model, mode });
    setMeta(meta, [
      `mode: <strong>${escapeHtml(data.mode || mode)}</strong>`,
      `retrieved: <strong>${data.retrieved.length}</strong>`,
      data.answer ? `model: ${escapeHtml(data.model || model)}` : '',
      `${Math.round(elapsedMs)} ms`,
    ].filter(Boolean));

    let html = '';
    if (data.answer) {
      html += `
        <article class="answer-card">
          <div class="answer-card-header">
            <span class="label">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
              Grounded answer
            </span>
            <span class="meta">${escapeHtml(data.model || model)} · ${escapeHtml(data.mode || mode)}</span>
          </div>
          <p class="answer-body">${escapeHtml(data.answer)}</p>
        </article>`;
    } else if (data.note) {
      html += `<div class="warn-state"><strong>LLM disabled.</strong>&nbsp;${escapeHtml(data.note)} Retrieval is still shown below.</div>`;
    }
    html += `
      <div class="retrieval-header">
        <h3>Retrieved evidence</h3>
        <span class="hint">ranked by cosine similarity</span>
      </div>`;
    html += renderSearchTable(data.retrieved);
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = errorHTML(err.message);
  } finally {
    setSubmitting(e.currentTarget, false);
  }
});

// ---- Cohort: mode switch ----
let cohortMode = 'patient';
document.querySelectorAll('[data-cohort-mode]').forEach((btn) => {
  btn.addEventListener('click', () => {
    cohortMode = btn.dataset.cohortMode;
    document.querySelectorAll('[data-cohort-mode]').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('[data-cohort-input]').forEach((el) => {
      el.hidden = el.dataset.cohortInput !== cohortMode;
    });
    // Clear stale results when switching mode.
    $('cohort-result').innerHTML = '';
    $('cohort-meta').hidden = true;
  });
});

// ---- Cohort: submit ----
$('cohort-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  closeAllPopovers();
  const k = +$('cohort-k').value;
  const out = $('cohort-result');
  const meta = $('cohort-meta');
  setSubmitting(e.currentTarget, true);
  try {
    if (cohortMode === 'text') {
      const query = $('cohort-text').value.trim();
      if (!query) {
        out.innerHTML = errorHTML('Enter a clinical description first.');
        return;
      }
      out.innerHTML = loadingHTML('Encoding description and ranking patient embeddings…');
      meta.hidden = true;
      const { data, elapsedMs } = await postJSON('/api/similar-patients-by-text', { query, k });
      setMeta(meta, [
        `mode: <strong>by description</strong>`,
        `query: <code>${escapeHtml(query)}</code>`,
        `patients: <strong>${data.length}</strong>`,
        `${Math.round(elapsedMs)} ms`,
      ]);
      out.innerHTML = renderCohortByTextTable(data);
    } else {
      const subject_id = +$('cohort-patient').value;
      out.innerHTML = loadingHTML('Computing patient-level similarities…');
      meta.hidden = true;
      const { data, elapsedMs } = await postJSON('/api/similar-patients', { subject_id, k });
      setMeta(meta, [
        `mode: <strong>by patient</strong>`,
        `seed: subject <strong>${subject_id}</strong>`,
        `neighbors: <strong>${data.length}</strong>`,
        `${Math.round(elapsedMs)} ms`,
      ]);
      out.innerHTML = renderCohortTable(data);
    }
  } catch (err) {
    out.innerHTML = errorHTML(err.message);
  } finally {
    setSubmitting(e.currentTarget, false);
  }
});

// ---- Trajectory ----
const PLOT_LAYOUT_BASE = {
  margin: { t: 36, l: 60, r: 18, b: 50 },
  font: { family: 'Inter, -apple-system, sans-serif', size: 11.5, color: '#4a5876' },
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  xaxis: { gridcolor: '#eef2f8', zerolinecolor: '#e2e8f0', linecolor: '#cbd5e2' },
  yaxis: { gridcolor: '#eef2f8', zerolinecolor: '#e2e8f0', linecolor: '#cbd5e2' },
  hoverlabel: { bgcolor: '#0d1424', font: { color: '#fff', size: 11.5, family: 'Inter, sans-serif' } },
  height: 340,
  showlegend: true,
  legend: { orientation: 'h', y: -0.2, x: 0, font: { size: 11 } },
};

async function loadTrajectoryInterpretation(subjectId) {
  const card = $('trajectory-interpret');
  card.hidden = false;
  card.innerHTML = `
    <div class="interpret-header">
      <span class="interpret-label">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 4 14h7l-1 8 9-14h-7z"/></svg>
        Trajectory interpretation
      </span>
      <span class="interpret-meta">drafting&hellip;</span>
    </div>
    <div class="interpret-body">${loadingHTML('Generating natural-language summary of inflection points…')}</div>`;
  try {
    const { data, elapsedMs } = await postJSON('/api/trajectory/interpret', { subject_id: subjectId });
    if (data.answer) {
      card.querySelector('.interpret-meta').textContent =
        `${escapeHtml(data.model || 'gpt-4o-mini')} · ${Math.round(elapsedMs)} ms`;
      card.querySelector('.interpret-body').innerHTML = renderMarkdownLite(data.answer);
    } else {
      card.querySelector('.interpret-meta').textContent = 'LLM disabled';
      card.querySelector('.interpret-body').innerHTML =
        `<div class="warn-state">${escapeHtml(data.note || 'No interpretation available.')}</div>`;
    }
  } catch (err) {
    card.querySelector('.interpret-meta').textContent = 'failed';
    card.querySelector('.interpret-body').innerHTML = errorHTML(err.message);
  }
}

$('trajectory-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  closeAllPopovers();
  const subject_id = +$('trajectory-patient').value;
  const statsEl = $('trajectory-stats');
  const velocityEl = $('trajectory-velocity');
  const pcaEl = $('trajectory-pca');
  const interpretEl = $('trajectory-interpret');
  statsEl.hidden = true;
  statsEl.innerHTML = '';
  interpretEl.hidden = true;
  interpretEl.innerHTML = '';
  Plotly.purge(velocityEl);
  Plotly.purge(pcaEl);
  velocityEl.innerHTML = loadingHTML('Loading trajectory…');
  pcaEl.innerHTML = '';
  setSubmitting(e.currentTarget, true);
  try {
    const { data } = await postJSON('/api/trajectory', { subject_id });
    const s = data.stats;
    statsEl.hidden = false;
    statsEl.innerHTML = `
      <div class="stat-block"><span class="stat-label">Patient</span><span class="stat-value">${data.subject_id}</span></div>
      <div class="stat-block"><span class="stat-label">Notes</span><span class="stat-value">${s.n_notes}</span></div>
      <div class="stat-block"><span class="stat-label">Median L2</span><span class="stat-value">${s.median_l2.toFixed(3)}</span></div>
      <div class="stat-block"><span class="stat-label">Max L2</span><span class="stat-value">${s.max_l2.toFixed(3)}</span></div>
      <div class="stat-block"><span class="stat-label">p95 spike threshold</span><span class="stat-value">${s.spike_threshold_p95.toFixed(3)}</span></div>
    `;

    const notes = data.notes;
    const x = notes.map((_, i) => i);
    const l2 = notes.map((n) => n.l2_prev);
    const spikeX = [];
    const spikeY = [];
    notes.forEach((n, i) => {
      if (i > 0 && n.l2_prev > s.spike_threshold_p95) {
        spikeX.push(i);
        spikeY.push(n.l2_prev);
      }
    });

    velocityEl.innerHTML = '';
    Plotly.newPlot(velocityEl, [
      {
        x, y: l2, mode: 'lines+markers', name: 'L2 to prev',
        line: { color: '#1d3a6f', width: 1.8 },
        marker: { color: '#1d3a6f', size: 5 },
        hovertemplate: 'note %{x}<br>L2=%{y:.3f}<extra></extra>',
      },
      {
        x: spikeX, y: spikeY, mode: 'markers', name: 'p95 spike',
        marker: { color: '#b8362d', size: 10, symbol: 'circle-open', line: { width: 2 } },
        hovertemplate: 'note %{x}<br>L2=%{y:.3f}<extra>spike</extra>',
      },
    ], {
      ...PLOT_LAYOUT_BASE,
      title: { text: 'Embedding velocity over time', font: { size: 13, color: '#0d1424' }, x: 0.02, xanchor: 'left' },
      xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'Note index (chronological)' },
      yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: 'L2 to previous note embedding' },
    }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toggleSpikelines'] });

    Plotly.newPlot(pcaEl, [{
      x: notes.map((n) => n.pca1),
      y: notes.map((n) => n.pca2),
      mode: 'markers',
      marker: {
        color: x,
        colorscale: [[0, '#a8d3c5'], [0.5, '#1d3a6f'], [1, '#0d1424']],
        size: 11,
        line: { color: '#fff', width: 1 },
        showscale: true,
        colorbar: { title: { text: 'Note order', side: 'right', font: { size: 11 } }, thickness: 10, len: 0.8 },
      },
      text: notes.map((n, i) => `Note ${i}<br>cat: ${escapeHtml(n.anchor_category)}<br>date: ${escapeHtml(n.anchor_date)}<br>L2 prev: ${(n.l2_prev ?? 0).toFixed(3)}`),
      hovertemplate: '%{text}<extra></extra>',
    }], {
      ...PLOT_LAYOUT_BASE,
      title: { text: '2-D PCA of patient note embeddings', font: { size: 13, color: '#0d1424' }, x: 0.02, xanchor: 'left' },
      xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: 'PC1' },
      yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: 'PC2' },
      showlegend: false,
    }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toggleSpikelines'] });

    // Auto-fire the LLM interpretation after the charts are up.
    loadTrajectoryInterpretation(subject_id);
  } catch (err) {
    statsEl.hidden = false;
    statsEl.innerHTML = errorHTML(err.message);
  } finally {
    setSubmitting(e.currentTarget, false);
  }
});

// ---- Init ----
loadHealth();
loadPatients();
