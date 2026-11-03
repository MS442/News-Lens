'use strict';

// Python performs extraction, retrieval, model inference and evaluation.
// The browser displays the returned evidence and measured metrics without
// substituting demo predictions, fabricated sources or placeholder scores.
const byId = (id) => document.getElementById(id);
const state = { view: 'check', mode: 'text', metrics: null, metricsLoading: false, requestNumber: 0, extractionNumber: 0, predictController: null, extractController: null, checkTimer: null, analyzing: false, extracting: false, source: null, sourceEdited: false };
const number = (value) => typeof value === 'number' && Number.isFinite(value) ? value : null;
const percentage = (value, places = 1) => number(value) === null ? '—' : `${(value * 100).toFixed(places)}%`;
const integer = (value) => number(value) === null ? '—' : Math.round(value).toLocaleString('en-US');
const decimal = (value, places = 3) => number(value) === null ? '—' : value.toFixed(places);
const words = (text) => text.trim().match(/\S+/g)?.length || 0;
const CLASS_ORDER = ['contradiction', 'entailment', 'neutral'];
const CLASS_NAMES = { contradiction: 'Contradiction', entailment: 'Entailment', neutral: 'Neutral' };
const VERDICTS = { supported: { label: 'Supported', tone: 'supported', icon: '✓' }, contradicted: { label: 'Contradicted', tone: 'contradicted', icon: '!' }, conflicting: { label: 'Conflicting evidence', tone: 'review', icon: '↔' }, mixed: { label: 'Mixed findings', tone: 'review', icon: '↔' }, insufficient_evidence: { label: 'Insufficient evidence', tone: 'review', icon: '?' } };

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function note(container, text) { container.replaceChildren(element('p', 'empty-note', text)); }
function safeUrl(value) {
  try { const url = new URL(value); return ['https:', 'http:'].includes(url.protocol) ? url.href : null; } catch (_) { return null; }
}
function dateLabel(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}
function externalLink(text, url, className = '') {
  const href = safeUrl(url);
  if (!href) return element('span', className, text);
  const link = element('a', className, text);
  link.href = href; link.target = '_blank'; link.rel = 'noopener noreferrer';
  return link;
}
function errorMessage(data) {
  if (typeof data.detail === 'string') return data.detail;
  if (typeof data.error === 'string') return data.error;
  if (typeof data.message === 'string') return data.message;
  if (data.detail?.message) return data.detail.message;
  if (Array.isArray(data.detail)) return data.detail.map((row) => row.msg).filter(Boolean).join(' ');
  return 'The service is temporarily unavailable. Please try again shortly.';
}
async function requestJson(path, options = {}) {
  const controller = options.controller || new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, options.timeout || 55000);
  try {
    const response = await fetch(path, { method: options.body ? 'POST' : 'GET', headers: options.body ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' }, body: options.body ? JSON.stringify(options.body) : undefined, signal: controller.signal, cache: 'no-store' });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error('The service did not return a readable result. Please try again shortly.'); }
    if (!response.ok) throw new Error(errorMessage(data));
    return data;
  } catch (error) {
    if (timedOut) throw new Error('The request timed out. Try again, or add a source passage and turn off web search for a direct comparison.');
    if (error instanceof TypeError) throw new Error('Could not reach the service. Check your connection and try again.');
    throw error;
  } finally { clearTimeout(timeout); }
}
function showToast(message) {
  const toast = byId('toast'); toast.textContent = message; toast.hidden = false;
  clearTimeout(showToast.timer); showToast.timer = setTimeout(() => { toast.hidden = true; }, 3000);
}
function switchView(view, updateHash = true) {
  state.view = view === 'results' ? 'results' : 'check';
  for (const tab of document.querySelectorAll('[data-view]')) {
    const selected = tab.dataset.view === state.view;
    tab.classList.toggle('is-active', selected); tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
  }
  byId('panel-check').hidden = state.view !== 'check'; byId('panel-results').hidden = state.view !== 'results';
  if (updateHash && window.location.hash !== `#${state.view}`) history.replaceState(null, '', `#${state.view}`);
  if (state.view === 'results' && !state.metrics && !state.metricsLoading) loadMetrics();
}
function switchInputMode(mode) {
  state.mode = mode === 'url' ? 'url' : 'text';
  byId('url-entry').hidden = state.mode !== 'url';
  document.querySelectorAll('[data-input-mode]').forEach((button) => { const active = button.dataset.inputMode === state.mode; button.classList.toggle('is-active', active); button.setAttribute('aria-pressed', String(active)); });
}
function showResultState(name) {
  for (const key of ['empty', 'loading', 'error', 'content']) byId(`result-${key}`).hidden = key !== name;
}
function stopCheckTimer() {
  if (state.checkTimer !== null) clearInterval(state.checkTimer);
  state.checkTimer = null;
}
function startCheckTimer(id) {
  stopCheckTimer();
  const started = performance.now();
  const update = () => {
    if (id !== state.requestNumber || !state.analyzing) { stopCheckTimer(); return; }
    const seconds = Math.floor((performance.now() - started) / 1000);
    byId('check-elapsed').textContent = `${seconds} s elapsed`;
    byId('check-wait-note').textContent = seconds >= 25
      ? 'Still waiting for the result. If this times out, try a source passage with web search off.'
      : 'Waiting for the server to finish the comparison.';
  };
  update(); state.checkTimer = setInterval(update, 1000);
}
function updateBusyUi() {
  if (!state.analyzing) stopCheckTimer();
  byId('analyze-button').disabled = state.analyzing || state.extracting;
  byId('analyze-button').querySelector('.button-label').textContent = state.analyzing ? 'Checking claims…' : 'Check claims';
  byId('analyze-button').querySelector('.button-arrow').hidden = state.analyzing;
  byId('analyze-button').querySelector('.spinner').hidden = !state.analyzing;
  byId('extract-button').disabled = state.extracting || state.analyzing;
  byId('extract-button').textContent = state.extracting ? 'Loading…' : 'Load article';
  byId('article-text').readOnly = state.extracting;
  document.querySelector('.result-card').setAttribute('aria-busy', String(state.analyzing));
}
function resetAnalysis() {
  state.requestNumber += 1; state.predictController?.abort(); state.predictController = null; state.analyzing = false;
  byId('result-badge').textContent = 'Ready when you are'; showResultState('empty'); updateBusyUi();
}
function cancelExtraction() {
  state.extractionNumber += 1; state.extractController?.abort(); state.extractController = null; state.extracting = false; updateBusyUi();
}
function feedback(message, isError = false) {
  const box = byId('input-feedback'); box.textContent = message; box.classList.toggle('is-error', isError); box.hidden = !message;
}
function updateInput() {
  const count = words(byId('article-text').value);
  byId('input-counter').textContent = `${integer(count)} ${count === 1 ? 'word' : 'words'}`;
  byId('clear-button').disabled = ![byId('article-text').value, byId('article-url').value, byId('evidence-text').value, byId('evidence-url').value].some(Boolean);
}
function showLoadedSource() {
  const box = byId('loaded-source'); box.hidden = !state.source;
  if (!state.source) { box.replaceChildren(); return; }
  const title = state.source.title || state.source.domain || 'Loaded article';
  box.replaceChildren(externalLink(title, state.source.source_url, 'loaded-source-title'));
  const detail = `${state.source.domain || 'Source page'}${state.sourceEdited ? ' · Text edited after import' : number(state.source.word_count) !== null ? ` · ${integer(state.source.word_count)} extracted words` : ''}`;
  box.append(element('p', '', detail), element('p', 'field-hint', 'This is the article’s origin, not independent supporting evidence.'));
}
async function extractArticle() {
  const url = safeUrl(byId('article-url').value.trim());
  if (!url) { feedback('Enter a complete http:// or https:// article link.', true); byId('article-url').focus(); return; }
  resetAnalysis(); cancelExtraction(); const id = state.extractionNumber;
  const controller = new AbortController(); state.extractController = controller; state.extracting = true; updateBusyUi(); feedback('Loading the article text…');
  try {
    const data = await requestJson('/api/extract', { body: { url }, controller, timeout: 55000 });
    if (id !== state.extractionNumber) return;
    if (typeof data.text !== 'string' || !data.text.trim()) throw new Error('No readable article text was found. Please paste the article instead.');
    state.source = { title: data.title, source_url: data.source_url || url, requested_url: url, domain: data.domain, word_count: data.word_count };
    state.sourceEdited = false; byId('article-text').value = data.text.slice(0, 30000); showLoadedSource(); updateInput();
    feedback(data.text.length > 30000 ? 'Article loaded. Only the first 30,000 characters fit in this input; review the text before checking.' : 'Article loaded. Review the extracted text, then check its claims.');
  } catch (error) { if (id === state.extractionNumber && error.name !== 'AbortError') feedback(error.message, true); }
  finally { if (id === state.extractionNumber) { state.extracting = false; state.extractController = null; updateBusyUi(); } }
}
function scoreBars(scores, className = 'claim-scores') {
  const box = element('div', className);
  const definitions = [['entailment', 'Entailment · supports'], ['contradiction', 'Contradiction · conflicts'], ['neutral', 'Neutral · not established']];
  for (const [key, label] of definitions) {
    if (number(scores?.[key]) === null) continue;
    const row = element('div', `nli-score nli-${key}`); const heading = element('div', 'nli-score-heading');
    heading.append(element('span', '', label), element('strong', '', percentage(scores[key])));
    const track = element('div', 'score-track'); const fill = element('span', 'score-fill'); fill.style.width = `${Math.min(1, Math.max(0, scores[key])) * 100}%`; track.append(fill);
    row.append(heading, track); box.append(row);
  }
  return box;
}
function sourceById(sources, id) { return sources.find((source) => String(source.id) === String(id)); }
function sourceNumber(sources, id) { const index = sources.findIndex((source) => String(source.id) === String(id)); return index < 0 ? null : index + 1; }
function sourceName(source) { return source?.title || source?.domain || (['provided', 'supplied'].includes(source?.kind) ? 'Supplied source passage' : 'Source passage'); }
function countLabel(count, singular, plural = `${singular}s`) { return `${integer(count)} ${count === 1 ? singular : plural}`; }
function sourceTextKind(source) {
  if (['provided', 'supplied'].includes(source?.kind)) return 'Supplied by you';
  if (source?.enrichment_status === 'full_text') return 'Article text loaded';
  if (source?.enrichment_status === 'search_excerpt') return 'Search excerpt';
  return source?.kind === 'web' || source?.kind === 'search' ? 'Retrieved source' : null;
}
function comparisonSummary(claim, sources) {
  const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
  const summary = claim.evidence_summary || {};
  const found = number(summary.sources_found) ?? sources.length;
  const compared = number(summary.sources_compared) ?? new Set(evidence.map((item) => item.source_id).filter(Boolean)).size;
  const passages = number(summary.passages_compared) ?? evidence.length;
  if (!found) return 'No related sources found.';
  if (!passages) return `${countLabel(found, 'related source')} found; no relevant passage compared.`;
  return `${countLabel(passages, 'passage')} compared from ${integer(compared)} of ${countLabel(found, 'related source')}.`;
}
function modelLean(relation) { return { entailment: 'Model leans toward support', contradiction: 'Model leans toward contradiction', neutral: 'Model leans toward insufficient context' }[relation] || null; }
function reviewReason(relation, reason) {
  if (typeof reason === 'string' && reason.trim()) return `Review required — ${reason.trim()}`;
  return relation === 'neutral' ? 'Review required — the model does not establish support or contradiction.' : 'Review required — threshold not met.';
}
function renderClaim(claim, index, sources) {
  const verdict = VERDICTS[claim.verdict] || VERDICTS.insufficient_evidence;
  const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
  const label = claim.verdict === 'insufficient_evidence' && evidence.length && (!claim.label || claim.label === verdict.label)
    ? 'Evidence found; review required' : claim.label || verdict.label;
  const details = element('details', `claim-card claim-${verdict.tone}`);
  const summary = element('summary');
  const heading = element('div', 'claim-summary-heading'); heading.append(element('span', 'claim-number', `${String(index + 1).padStart(2, '0')}`), element('span', `verdict-tag verdict-${verdict.tone}`, label), element('span', 'claim-chevron', '+'));
  summary.append(heading, element('p', 'claim-text', claim.text));
  summary.append(element('p', 'claim-comparison-summary', comparisonSummary(claim, sources)));
  const reviewLean = claim.verdict === 'insufficient_evidence' ? modelLean(claim.model_prediction) : null;
  if (reviewLean) summary.append(element('p', 'claim-model-lean', `${reviewLean} · review required`));
  details.append(summary);
  const body = element('div', 'claim-body');
  const reason = claim.reason || ({ no_sources: 'No source was available for comparison.', no_relevant_passages: 'Related sources were found, but no relevant passage was selected.', model_uncertain: 'Evidence was compared, but the model did not reach a decisive result.' }[claim.review_reason]);
  if (reason) body.append(element('p', 'claim-reason', reason));
  if (reviewLean) body.append(element('p', 'model-review-note', reviewReason(claim.model_prediction, claim.decision_reason)));
  const bars = scoreBars(claim.scores);
  if (bars.childElementCount) body.append(element('p', 'evidence-small-label', 'SELECTED PASSAGE · MODEL SCORES'), bars, element('p', 'score-caveat', 'These scores compare this claim with one selected passage, not the whole article. They are not probabilities of truth. Thresholds can require review even when one score is highest.'));
  if (!evidence.length && !reason) body.append(element('p', 'no-evidence-note', 'No passage was compared for this claim.'));
  evidence.forEach((item) => {
    const source = sourceById(sources, item.source_id); const citation = sourceNumber(sources, item.source_id);
    const card = element('div', 'evidence-passage');
    const name = `${citation ? `[${citation}] ` : ''}${sourceName(source)}`;
    card.append(externalLink(name, source?.url, 'evidence-source-link'));
    if (source?.domain || source?.kind) card.append(element('p', 'evidence-source-meta', [source.domain, sourceTextKind(source)].filter(Boolean).join(' · ')));
    card.append(element('blockquote', '', item.passage || 'No passage text was returned.'));
    const needsReview = item.decision === 'review' || (!item.decision && ['review', 'neutral'].includes(item.relation));
    if (needsReview) {
      const lean = modelLean(item.raw_relation);
      if (lean) card.append(element('p', 'passage-model-lean', lean));
      card.append(element('p', 'model-review-note', reviewReason(item.raw_relation, item.decision_reason)));
    } else if (item.relation) card.append(element('span', 'evidence-relation', `Passage relation: ${CLASS_NAMES[item.relation] || VERDICTS[item.relation]?.label || item.relation}`));
    if (item.truncated) card.append(element('p', 'passage-truncation', `Input truncated${number(item.token_count) !== null ? ` to ${integer(item.token_count)} model tokens` : ''}. The model did not see the complete passage and claim.`));
    if (item.scores && evidence.length > 1) { const pairDetails = element('details', 'passage-scores'); pairDetails.append(element('summary', '', 'This passage’s model scores'), scoreBars(item.scores)); card.append(pairDetails); }
    body.append(card);
  });
  details.append(body); return details;
}
function renderPrediction(data) {
  if (!Array.isArray(data.claims) || !data.assessment) throw new Error('The response does not contain a source-backed claim assessment. Please try again once the evidence model is ready.');
  const style = VERDICTS[data.assessment] || VERDICTS.insufficient_evidence;
  byId('assessment-banner').className = `assessment-banner${style.tone === 'contradicted' ? ' is-misleading' : style.tone === 'review' ? ' is-review' : ''}`;
  byId('assessment-icon').textContent = style.icon;
  byId('assessment-label').textContent = data.label || style.label;
  byId('assessment-reason').textContent = data.summary || 'Inspect each claim and its cited evidence below.';
  byId('result-badge').textContent = data.model?.name || 'Comparison complete';
  const coverage = data.coverage || {};
  const counts = [['Checked', coverage.checked_claims ?? data.claims.length], ['Supported', coverage.supported], ['Contradicted', coverage.contradicted], ['Unresolved', coverage.insufficient_evidence]];
  if (number(coverage.conflicting) !== null && coverage.conflicting > 0) counts.push(['Conflicting', coverage.conflicting]);
  byId('coverage-summary').replaceChildren(...counts.filter(([, value]) => number(value) !== null).map(([label, value]) => { const item = element('div'); item.append(element('strong', '', integer(value)), element('span', '', label)); return item; }));
  const coverageNotes = [];
  if (number(coverage.total_sentences) !== null) coverageNotes.push(`${integer(coverage.checked_claims ?? data.claims.length)} selected claims from ${integer(coverage.total_sentences)} sentences`);
  if (number(coverage.omitted) !== null && coverage.omitted > 0) coverageNotes.push(`${integer(coverage.omitted)} sentences not checked`);
  byId('coverage-note').textContent = coverageNotes.length ? `${coverageNotes.join(' · ')}. This is not a check of every statement in the article.` : 'The assessment covers the selected claims shown below.';
  const search = data.search || {};
  const warnings = Array.isArray(data.warnings) ? data.warnings.filter((warning) => typeof warning === 'string') : [];
  const searchMessages = [...new Set([search.message || (search.status ? `Web search: ${String(search.status).replaceAll('_', ' ')}.` : ''), ...warnings].filter(Boolean))];
  byId('search-status').hidden = !searchMessages.length;
  byId('search-status').classList.toggle('has-warning', warnings.length > 0 || search.status === 'unavailable');
  byId('search-status').textContent = searchMessages.join(' ');
  const sources = Array.isArray(data.sources) ? data.sources : [];
  const usage = data.source_usage || {};
  const allEvidence = data.claims.flatMap((claim) => Array.isArray(claim.evidence) ? claim.evidence : []);
  const comparedSources = number(usage.sources_compared) ?? new Set(allEvidence.map((item) => item.source_id).filter(Boolean)).size;
  const foundSources = number(usage.sources_found) ?? sources.length;
  const comparedPassages = number(usage.passages_compared) ?? allEvidence.length;
  byId('source-usage-summary').textContent = `${countLabel(foundSources, 'related source')} found · ${integer(comparedSources)} compared · ${countLabel(comparedPassages, 'passage')} checked. Source counts are not confirmations.`;
  byId('claims-list').replaceChildren(...data.claims.map((claim, index) => renderClaim(claim, index, sources)));
  if (!data.claims.length) note(byId('claims-list'), 'No checkable claims were identified. Try a specific factual claim or another article.');
  byId('sources-panel').hidden = !sources.length; byId('source-count').textContent = integer(sources.length);
  byId('sources-list').replaceChildren(...sources.map((source, index) => {
    const item = element('div', 'source-list-item'); item.append(externalLink(`[${index + 1}] ${sourceName(source)}`, source.url));
    const usedClaims = Array.isArray(source.used_for_claims) ? [...new Set(source.used_for_claims)] : data.claims.filter((claim) => (claim.evidence || []).some((row) => String(row.source_id) === String(source.id))).map((claim) => claim.id);
    item.append(element('span', `source-use-tag${usedClaims.length ? ' is-compared' : ''}`, usedClaims.length ? `Compared for ${countLabel(usedClaims.length, 'claim')}` : 'Not compared'));
    const published = dateLabel(source.published_at); const retrieved = dateLabel(source.retrieved_at);
    item.append(element('p', '', [source.domain, sourceTextKind(source), published ? `Published ${published}` : null, retrieved ? `Retrieved ${retrieved}` : null].filter(Boolean).join(' · ')));
    return item;
  }));
  const limitations = Array.isArray(data.limitations) ? data.limitations : [];
  byId('prediction-limitations').hidden = !limitations.length;
  byId('prediction-limitations').replaceChildren();
  if (limitations.length) { const details = element('details'); details.append(element('summary', '', 'Limits of this check')); const list = element('ul'); limitations.forEach((text) => list.append(element('li', '', text))); details.append(list); byId('prediction-limitations').append(details); }
  renderResultPerformance(); if (!state.metrics && !state.metricsLoading) loadMetrics(); showResultState('content');
}
async function analyzeArticle() {
  const text = byId('article-text').value.trim();
  const articleUrl = byId('article-url').value.trim();
  if (state.mode === 'url' && articleUrl && safeUrl(articleUrl) !== state.source?.requested_url) { await extractArticle(); return; }
  if (!text && state.mode === 'url' && byId('article-url').value.trim()) { await extractArticle(); return; }
  if (!text) { feedback('Paste an article or a specific claim, or load an article link.', true); byId('article-text').focus(); return; }
  if (/^https?:\/\/\S+$/i.test(text)) { byId('article-url').value = text; switchInputMode('url'); await extractArticle(); return; }
  const evidenceText = byId('evidence-text').value.trim(); const evidenceUrlText = byId('evidence-url').value.trim();
  if (evidenceUrlText && !safeUrl(evidenceUrlText)) { feedback('Enter a complete http:// or https:// evidence URL, or leave it empty.', true); byId('supplied-evidence').open = true; byId('evidence-url').focus(); return; }
  if (!byId('search-web').checked && !evidenceText && !evidenceUrlText) { feedback('Add a source passage or source URL, or enable web search so there is evidence to compare.', true); byId('supplied-evidence').open = true; byId('evidence-text').focus(); return; }
  resetAnalysis(); const id = state.requestNumber; const controller = new AbortController(); state.predictController = controller; state.analyzing = true; updateBusyUi(); feedback(''); showResultState('loading'); startCheckTimer(id); byId('result-badge').textContent = 'Check in progress';
  try {
    const data = await requestJson('/api/predict', { body: { text, source_url: state.source?.source_url || null, evidence_text: evidenceText || null, evidence_url: evidenceUrlText ? safeUrl(evidenceUrlText) : null, search_web: byId('search-web').checked }, controller });
    if (id !== state.requestNumber) return;
    state.analyzing = false; updateBusyUi(); renderPrediction(data);
    if (window.innerWidth < 761) byId('result-heading').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    if (id !== state.requestNumber || error.name === 'AbortError') return;
    state.analyzing = false; updateBusyUi(); byId('result-badge').textContent = 'Check not completed'; byId('result-error-message').textContent = error.message; showResultState('error');
  } finally { if (id === state.requestNumber) { state.predictController = null; state.analyzing = false; updateBusyUi(); } }
}
async function loadExamples() {
  try {
    const data = await requestJson('/api/examples', { timeout: 20000 }); const examples = Array.isArray(data) ? data : data.examples;
    if (!Array.isArray(examples)) return;
    const nodes = examples.filter((sample) => typeof sample.text === 'string' && sample.text.trim()).slice(0, 4).map((sample, index) => {
      const button = element('button', 'example-button', sample.title || `Example ${index + 1}`); button.type = 'button';
      button.addEventListener('click', () => {
        resetAnalysis(); cancelExtraction(); state.source = null; state.sourceEdited = false; showLoadedSource(); switchInputMode('text');
        byId('article-url').value = ''; byId('article-text').value = sample.text.slice(0, 30000); byId('evidence-text').value = (sample.evidence_text || '').slice(0, 20000); byId('evidence-url').value = sample.evidence_url || ''; byId('search-web').checked = sample.search_web ?? !sample.evidence_text; byId('supplied-evidence').open = Boolean(sample.evidence_text); updateInput(); feedback('');
        byId('example-caption').textContent = sample.evidence_text ? 'Example loaded with its source passage. Press Check claims to run the model.' : 'Example loaded. Press Check claims to retrieve and compare evidence.'; byId('example-caption').hidden = false; byId('article-text').focus();
      }); return button;
    });
    byId('example-buttons').replaceChildren(...nodes); byId('examples-block').hidden = !nodes.length;
  } catch (_) { /* Optional examples never block article analysis. */ }
}

function makeTable(headers, rows) {
  const table = element('table', 'data-table'); const head = element('thead'); const headRow = element('tr');
  for (const title of headers) { const cell = element('th', '', title); cell.scope = 'col'; headRow.append(cell); } head.append(headRow);
  const body = element('tbody'); rows.forEach((values) => { const row = element('tr'); values.forEach((value) => row.append(element('td', '', value))); body.append(row); }); table.append(head, body); return table;
}
function tableContainer(table) { const box = element('div', 'table-scroll'); box.tabIndex = 0; box.setAttribute('role', 'region'); box.setAttribute('aria-label', 'Data table; scroll horizontally if needed'); box.append(table); return box; }
function metricCards(metrics) {
  const definitions = [['macro_precision', 'Macro precision', 'How often each predicted relationship is correct, averaged equally across the three classes.', false], ['macro_recall', 'Macro recall', 'How many examples of each relationship are found, averaged equally across the three classes.', false], ['macro_f1', 'Macro F1', 'The average of the three class F1 scores, balancing precision and recall.', true], ['accuracy', 'Accuracy', 'The share of held-out evidence–claim pairs assigned the correct relationship.', false]];
  byId('metric-cards').replaceChildren(...definitions.map(([key, title, description, primary]) => { const card = element('div', `metric-card${primary ? ' is-primary' : ''}`); const heading = element('div', 'metric-name', title); const help = element('span', 'metric-help', '?'); help.title = description; help.setAttribute('aria-label', description); heading.append(help); card.append(heading, element('span', 'metric-value', percentage(metrics[key])), element('p', 'metric-description', description)); return card; }));
}
function renderResultPerformance() {
  const test = state.metrics?.test; byId('result-test-performance').hidden = !test;
  if (!test) return;
  byId('result-test-statistics').replaceChildren(...[['Macro precision', test.macro_precision], ['Macro recall', test.macro_recall], ['Macro F1', test.macro_f1]].map(([label, value]) => { const item = element('div'); item.append(element('span', '', label), element('strong', '', percentage(value))); return item; }));
}
function renderMatrix(test) {
  const matrix = test.confusion_matrix; const labels = Array.isArray(test.matrix_labels) && test.matrix_labels.length === 3 ? test.matrix_labels : CLASS_ORDER;
  if (!Array.isArray(matrix) || matrix.length !== 3 || matrix.some((row) => !Array.isArray(row) || row.length !== 3)) { note(byId('confusion-matrix'), 'The three-class matrix has not been included.'); return; }
  const table = element('table', 'nli-matrix'); const caption = element('caption', '', 'Rows: actual relationship · Columns: prediction'); table.append(caption);
  const head = element('thead'); const header = element('tr'); const corner = element('th', '', 'Actual ↓ / Predicted →'); corner.scope = 'col'; header.append(corner);
  labels.forEach((label) => { const cell = element('th', '', CLASS_NAMES[label] || label); cell.scope = 'col'; header.append(cell); }); head.append(header); table.append(head);
  const body = element('tbody'); matrix.forEach((values, rowIndex) => { const row = element('tr'); const label = element('th', '', CLASS_NAMES[labels[rowIndex]] || labels[rowIndex]); label.scope = 'row'; row.append(label); values.forEach((value, columnIndex) => row.append(element('td', rowIndex === columnIndex ? 'matrix-correct' : 'matrix-incorrect', integer(value)))); body.append(row); }); table.append(body);
  byId('confusion-matrix').replaceChildren(tableContainer(table), element('p', 'matrix-footnote', 'Diagonal cells are correct predictions. Other cells show which text relationships the model confuses.'));
}
function renderComparison(data) {
  const test = data.test; const baseline = test.baseline || {};
  byId('comparison-context').textContent = data.baseline?.context_note || 'A simpler text-feature model is tested on the same evidence–claim pairs. This checks whether the Transformer adds value.';
  const rows = [['Transformer', percentage(test.macro_precision), percentage(test.macro_recall), percentage(test.macro_f1)]];
  if (number(baseline.macro_f1) !== null) rows.push([data.baseline?.name || 'Text baseline', percentage(baseline.macro_precision), percentage(baseline.macro_recall), percentage(baseline.macro_f1)]);
  byId('baseline-table').replaceChildren(tableContainer(makeTable(['Model', 'Macro P', 'Macro R', 'Macro F1'], rows)));
  const perClass = Array.isArray(test.per_class) ? test.per_class : [];
  byId('class-metrics').replaceChildren();
  if (perClass.length) byId('class-metrics').append(element('h3', 'subsection-title', 'PER-CLASS PERFORMANCE'), tableContainer(makeTable(['Class', 'Precision', 'Recall', 'F1'], perClass.map((row) => [CLASS_NAMES[row.label] || row.label, percentage(row.precision), percentage(row.recall), percentage(row['f1-score'] ?? row.f1)]))));
}
function datasetName(name) { return String(name).replace(/^external_/, '').replaceAll('_', ' ').replace(/\b(mnli|snli|fever|anli|scifact)\b/gi, (part) => part.toUpperCase()); }
function renderDatasetPerformance(data) {
  const rows = Object.entries(data.test.by_dataset || {}).map(([name, metrics]) => [datasetName(name), integer(metrics.support), percentage(metrics.accuracy), percentage(metrics.macro_f1), percentage(metrics.baseline?.macro_f1)]);
  if (rows.length) byId('dataset-performance').replaceChildren(tableContainer(makeTable(['Held-out dataset', 'Pairs', 'Accuracy', 'Macro F1', 'Baseline F1'], rows)));
  else note(byId('dataset-performance'), 'Per-dataset results have not been included.');
  const externalRows = Object.entries(data.external || {}).filter(([, metrics]) => metrics && number(metrics.macro_f1) !== null).map(([name, metrics]) => [datasetName(name), integer(metrics.support), percentage(metrics.accuracy), percentage(metrics.macro_f1), percentage(metrics.baseline?.macro_f1)]);
  byId('external-performance').replaceChildren();
  if (externalRows.length) byId('external-performance').append(element('h3', 'subsection-title', 'ORIGINAL DATASET HOLDOUTS'), tableContainer(makeTable(['External dataset', 'Pairs', 'Accuracy', 'Macro F1', 'Baseline F1'], externalRows)), element('p', 'table-note', 'Original held-out partitions are shown separately. These dataset families also appear in training; this is not zero-shot evaluation. Pair scores do not measure web-search quality.'));
}
function svgNode(tag, attributes = {}, text) { const node = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value))); if (text !== undefined) node.textContent = String(text); return node; }
function renderLearning(historyData) {
  const history = Array.isArray(historyData) ? historyData.filter((row) => number(row.train_loss) !== null || number(row.val_loss) !== null) : [];
  if (!history.length) { note(byId('learning-chart'), 'Training curves have not been included.'); return; }
  const width = 480, height = 235, plot = { left: 44, right: 17, top: 14, bottom: 37 };
  const values = history.flatMap((row) => [row.train_loss, row.val_loss]).filter((value) => number(value) !== null); const maximum = Math.max(...values) * 1.15 || 1; const innerWidth = width - plot.left - plot.right, innerHeight = height - plot.top - plot.bottom;
  const x = (index) => plot.left + (history.length === 1 ? innerWidth / 2 : index / (history.length - 1) * innerWidth); const y = (value) => plot.top + innerHeight * (1 - value / maximum);
  const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `Training and validation loss across ${history.length} recorded epochs` }); svg.append(svgNode('title', {}, 'Training and validation loss by epoch'), svgNode('desc', {}, history.map((row, index) => `Epoch ${row.epoch ?? index + 1}: training loss ${decimal(row.train_loss, 4)}, validation loss ${decimal(row.val_loss, 4)}.`).join(' ')));
  for (let tick = 0; tick <= 4; tick += 1) { const value = maximum * tick / 4, position = y(value); svg.append(svgNode('line', { x1: plot.left, y1: position, x2: width - plot.right, y2: position, stroke: '#e8e9df', 'stroke-dasharray': tick ? '3 4' : 'none' }), svgNode('text', { x: plot.left - 9, y: position + 3.5, 'text-anchor': 'end', fill: '#777f6a', 'font-size': 10, 'font-family': 'Segoe UI,Arial,sans-serif' }, value.toFixed(2))); }
  history.forEach((row, index) => svg.append(svgNode('text', { x: x(index), y: height - 17, 'text-anchor': 'middle', fill: '#777f6a', 'font-size': 10, 'font-family': 'Segoe UI,Arial,sans-serif' }, row.epoch ?? index + 1)));
  svg.append(svgNode('text', { x: width / 2, y: height - 1, 'text-anchor': 'middle', fill: '#777f6a', 'font-size': 9, 'font-family': 'Segoe UI,Arial,sans-serif' }, 'Epoch'));
  for (const [key, color] of [['train_loss', '#596348'], ['val_loss', '#a75c41']]) { const points = history.map((row, index) => number(row[key]) === null ? null : [x(index), y(row[key])]).filter(Boolean); svg.append(svgNode('polyline', { points: points.map((point) => point.join(',')).join(' '), fill: 'none', stroke: color, 'stroke-width': 2.3, 'stroke-dasharray': key === 'val_loss' ? '5 4' : 'none' })); points.forEach(([cx, cy]) => svg.append(svgNode('circle', { cx, cy, r: 3.5, fill: color, stroke: '#fffefb', 'stroke-width': 1.5 }))); }
  const details = element('details', 'chart-values'); details.append(element('summary', '', 'View exact training values'), tableContainer(makeTable(['Epoch', 'Train loss', 'Val. loss', 'Val. macro F1'], history.map((row, index) => [row.epoch ?? index + 1, decimal(row.train_loss, 4), decimal(row.val_loss, 4), percentage(row.val_macro_f1)])))); byId('learning-chart').replaceChildren(svg, details);
}
function detailList(items) { const list = element('dl', 'detail-list'); items.forEach(([label, value, description]) => { const row = element('div'); const dd = element('dd', '', value); if (description) dd.append(element('small', '', description)); row.append(element('dt', '', label), dd); list.append(row); }); return list; }
function renderCalibration(data) {
  const calibration = data.calibration || {}; const items = [];
  for (const key of ['entailment', 'contradiction']) { const value = calibration.thresholds?.[key]; if (number(value) !== null) items.push([`${CLASS_NAMES[key]} threshold`, value > 1 ? 'Disabled' : percentage(value, 1), value > 1 ? 'Validation precision target was not met' : 'Chosen using validation pairs']); }
  if (number(calibration.minimum_margin) !== null) items.push(['Minimum winning margin', percentage(calibration.minimum_margin), 'Gap above the second-highest class score']);
  if (number(calibration.target_precision) !== null) items.push(['Validation precision target', percentage(calibration.target_precision), 'A tuning target, not a guarantee on new claims']);
  if (number(calibration.temperature) !== null) items.push(['Calibration temperature', decimal(calibration.temperature, 2), 'Fitted on validation data']);
  if (number(calibration.before?.ece) !== null) items.push(['Validation ECE before calibration', percentage(calibration.before.ece, 2)]);
  if (number(calibration.after?.ece) !== null) items.push(['Validation ECE after calibration', percentage(calibration.after.ece, 2), 'Calibration-fit data; not a new test']);
  const policy = data.test?.decision_policy || calibration.decision_policy;
  if (number(policy?.decisive_coverage) !== null) items.push(['Decisive pair coverage', percentage(policy.decisive_coverage), 'Pairs passing the model decision policy']);
  if (number(policy?.decisive_accuracy) !== null) items.push(['Accuracy on decisive pairs', percentage(policy.decisive_accuracy), `${number(policy.selected) !== null ? integer(policy.selected) : 'Selected'} held-out pairs`]);
  const container = byId('calibration-details'); if (!items.length) { note(container, 'Threshold and calibration results have not been included.'); return; }
  container.replaceChildren(detailList(items), element('p', 'table-note', 'These measures cover model decisions on labelled evidence pairs. Claim selection, retrieval relevance and source conflicts add further uncertainty in the app.'));
  const policyRows = [['Internal test', data.test?.decision_policy], ...Object.entries(data.external || {}).map(([name, report]) => [datasetName(name), report.decision_policy])].filter(([, report]) => report && number(report.decisive_coverage) !== null);
  if (policyRows.length) container.append(element('h3', 'subsection-title', 'THRESHOLD POLICY BY HELD-OUT SPLIT'), tableContainer(makeTable(['Split', 'Decisive pairs', 'Coverage', 'Accuracy'], policyRows.map(([name, report]) => [name, integer(report.selected), percentage(report.decisive_coverage), percentage(report.decisive_accuracy)]))), element('p', 'table-note', 'Accuracy here counts only decisive pairs. A validation precision target can fail badly on a different distribution; coverage shows how many pairs receive a decision.'));
  if (calibration.warning) container.append(element('p', 'table-note', calibration.warning));
}
function renderDatasets(data) {
  const manifest = data.data_manifest || {}; const container = byId('dataset-details'); const grid = element('div', 'dataset-grid');
  const sources = manifest.source_licenses && typeof manifest.source_licenses === 'object' ? Object.entries(manifest.source_licenses).map(([name, source]) => ({ ...source, name, url: source.primary })) : Array.isArray(manifest.datasets) ? manifest.datasets : Array.isArray(manifest.sources) ? manifest.sources : [];
  sources.forEach((source) => { const block = element('div', 'dataset-block'); block.append(element('h3', '', source.name || source.dataset || 'Dataset')); if (source.description || source.role) block.append(element('p', '', source.description || source.role)); if (source.license) block.append(element('p', '', `License: ${source.license}`)); if (safeUrl(source.url || source.homepage)) block.append(externalLink('Dataset source ↗', source.url || source.homepage)); grid.append(block); });
  container.replaceChildren(); if (grid.childElementCount) container.append(grid);
  if (manifest.splits && typeof manifest.splits === 'object') {
    const rows = Object.entries(manifest.splits).filter(([, split]) => split && number(split.records ?? split.count) !== null).map(([name, split]) => [datasetName(name), integer(split.records ?? split.count), integer(split.class_counts?.['0'] ?? split.class_counts?.contradiction), integer(split.class_counts?.['1'] ?? split.class_counts?.entailment), integer(split.class_counts?.['2'] ?? split.class_counts?.neutral)]);
    if (rows.length) container.append(element('h3', 'subsection-title', 'PREPARED EVIDENCE–CLAIM PAIRS'), tableContainer(makeTable(['Split', 'Pairs', 'Contradiction', 'Entailment', 'Neutral'], rows)));
  }
  const model = data.model || {}, training = data.training || {}; const items = [];
  if (model.architecture) items.push(['Architecture', model.architecture]);
  if (number(model.parameters) !== null) items.push(['Model parameters', integer(model.parameters)]);
  if (number(model.max_tokens) !== null) items.push(['Maximum pair tokens', integer(model.max_tokens)]);
  if (number(training.epochs_completed) !== null) items.push(['Epochs completed', integer(training.epochs_completed)]);
  if (number(training.best_epoch) !== null) items.push(['Selected checkpoint epoch', integer(training.best_epoch)]);
  if (training.optimizer) items.push(['Optimizer', training.optimizer]);
  if (number(training.learning_rate) !== null) items.push(['Initial learning rate', training.learning_rate.toExponential(1)]);
  if (number(training.batch_size) !== null) items.push(['Training batch size', integer(training.batch_size)]);
  if (items.length) container.append(detailList(items));
  const notes = [training.base_training_disclosure, manifest.split_strategy, manifest.cleaning, data.evaluation_note].filter((value) => typeof value === 'string' && value);
  if (notes.length) container.append(element('p', 'dataset-wide-note', notes.join(' ')));
  if (!container.childElementCount) note(container, 'Dataset preparation details have not been included.');
}
function renderConcepts(data) {
  const concepts = Array.isArray(data.concepts) && data.concepts.length ? data.concepts : [
    { name: 'Natural language inference', explanation: 'Compare an evidence passage with a claim: does it entail it, contradict it, or leave it unestablished?' },
    { name: 'Subword tokenization', explanation: 'The model’s tokenizer converts both texts into token IDs with special tokens, padding and attention masks.' },
    { name: 'Contextual embeddings', explanation: 'A pretrained Transformer represents words using the surrounding text and token positions.' },
    { name: 'Self-attention', explanation: 'Attention combines information across tokens so the model can compare the passage and the claim in context.' },
    { name: 'Fine-tuning', explanation: 'Training updates a pretrained language model for this three-class relationship task. It does not train a language model from scratch.' },
    { name: 'Loss and backpropagation', explanation: 'Cross-entropy measures label-prediction error; gradients guide updates to the model’s weights.' },
    { name: 'Optimization and regularization', explanation: 'The training setup controls learning rate, optimization and regularization. The saved validation checkpoint is evaluated on held-out data.' },
    { name: 'Calibration and abstention', explanation: 'Validation-fitted scores and thresholds allow the model to withhold a decisive relationship when support is weak.' },
    { name: 'Retrieval is a separate step', explanation: 'Python finds candidate source passages. Pair-classification metrics do not establish that retrieval found the best or most reliable evidence.' },
  ];
  byId('concepts-list').replaceChildren(...concepts.map((concept) => { const card = element('div', 'concept-card'); card.append(element('h3', '', concept.name), element('p', '', concept.explanation || concept.description)); return card; }));
}
function renderMetrics(data) {
  const test = data.test;
  if (!test || !Array.isArray(test.confusion_matrix) || test.confusion_matrix.length !== 3 || number(test.macro_precision) === null || number(test.macro_recall) === null || number(test.macro_f1) === null) throw new Error('The three-class evidence model’s evaluation is not available yet. Earlier article-pattern scores are not used for this model.');
  state.metrics = data; byId('evaluation-model').textContent = data.model?.name || 'Evidence comparison model'; byId('evaluation-count').textContent = `${integer(test.support)} held-out evidence–claim pairs`; const date = dateLabel(data.generated_at || data.evaluated_at); byId('evaluation-date').textContent = date ? `Evaluated ${date}` : '';
  metricCards(test); renderResultPerformance(); renderMatrix(test); renderComparison(data); renderDatasetPerformance(data); renderLearning(data.history); renderCalibration(data); renderDatasets(data); renderConcepts(data);
  const limitations = Array.isArray(data.limitations) && data.limitations.length ? data.limitations : ['Support means agreement with the cited passage, not proof that the source or claim is true.', 'Selected-claim coverage and web retrieval are separate from model evaluation on supplied evidence pairs.', 'A model can confidently misread evidence. Read the cited passage and check independent sources.']; byId('limitations-list').replaceChildren(...limitations.map((text) => element('li', '', text)));
  byId('metrics-content').hidden = false; byId('metrics-unavailable').hidden = true; byId('download-metrics').hidden = false;
}
async function loadMetrics() {
  if (state.metricsLoading) return; state.metricsLoading = true; byId('metrics-loading').hidden = false; byId('metrics-unavailable').hidden = true;
  try { renderMetrics(await requestJson('/api/metrics', { timeout: 30000 })); }
  catch (error) { if (!state.metrics) { byId('metrics-content').hidden = true; byId('metrics-unavailable').hidden = false; byId('metrics-error-message').textContent = error.message; byId('download-metrics').hidden = true; } else showToast('Could not refresh results. The previously loaded evaluation is still shown.'); }
  finally { state.metricsLoading = false; byId('metrics-loading').hidden = true; }
}

document.querySelectorAll('[data-view]').forEach((tab) => tab.addEventListener('click', () => switchView(tab.dataset.view)));
document.querySelectorAll('[data-open-results]').forEach((button) => button.addEventListener('click', () => { switchView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); }));
document.querySelectorAll('[data-input-mode]').forEach((button) => button.addEventListener('click', () => switchInputMode(button.dataset.inputMode)));
document.querySelector('[role="tablist"]').addEventListener('keydown', (event) => { const tabs = [...document.querySelectorAll('[data-view]')]; const current = tabs.indexOf(document.activeElement); if (current < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return; event.preventDefault(); const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length; tabs[next].focus(); switchView(tabs[next].dataset.view); });
window.addEventListener('hashchange', () => switchView(window.location.hash === '#results' ? 'results' : 'check', false));
byId('article-text').addEventListener('input', () => { resetAnalysis(); cancelExtraction(); if (state.source) state.sourceEdited = true; showLoadedSource(); updateInput(); feedback(''); byId('example-caption').hidden = true; });
byId('article-text').addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !byId('analyze-button').disabled) { event.preventDefault(); analyzeArticle(); } });
byId('article-url').addEventListener('input', () => { resetAnalysis(); cancelExtraction(); updateInput(); feedback(state.source && safeUrl(byId('article-url').value.trim()) !== state.source.requested_url ? 'Load this link to replace the current text preview.' : ''); });
byId('article-url').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !byId('extract-button').disabled) { event.preventDefault(); extractArticle(); } });
['evidence-text', 'evidence-url'].forEach((id) => byId(id).addEventListener('input', () => { resetAnalysis(); updateInput(); feedback(''); byId('example-caption').hidden = true; }));
byId('search-web').addEventListener('change', () => { resetAnalysis(); feedback(''); });
byId('clear-button').addEventListener('click', () => { resetAnalysis(); cancelExtraction(); ['article-text', 'article-url', 'evidence-text', 'evidence-url'].forEach((id) => { byId(id).value = ''; }); state.source = null; state.sourceEdited = false; showLoadedSource(); byId('supplied-evidence').open = false; byId('example-caption').hidden = true; updateInput(); feedback(''); byId('article-text').focus(); });
byId('extract-button').addEventListener('click', extractArticle); byId('analyze-button').addEventListener('click', analyzeArticle); byId('retry-button').addEventListener('click', analyzeArticle); byId('reload-metrics').addEventListener('click', loadMetrics);
switchView(window.location.hash === '#results' ? 'results' : 'check', false); updateInput(); loadExamples();
