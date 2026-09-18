(() => {
  const log = document.getElementById('log');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const resetBtn = document.getElementById('resetBtn');

  const SESSION_KEY = 'darukaa_session_id';
  let sessionId = sessionStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, sessionId);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function addUserEntry(text) {
    const entry = el('div', 'log-entry user');
    const meta = el('div', 'entry-meta');
    meta.appendChild(el('span', 'who', 'You'));
    const body = el('div', 'entry-body');
    body.appendChild(el('p', null, text));
    entry.appendChild(meta);
    entry.appendChild(body);
    log.appendChild(entry);
    scrollToBottom();
  }

  function addSystemEntry(children) {
    const entry = el('div', 'log-entry system');
    const meta = el('div', 'entry-meta');
    meta.appendChild(el('span', 'who', 'System'));
    const body = el('div', 'entry-body');
    children.forEach(c => body.appendChild(c));
    entry.appendChild(meta);
    entry.appendChild(body);
    log.appendChild(entry);
    scrollToBottom();
    return entry;
  }

  function addErrorEntry(text) {
    const entry = el('div', 'log-entry error system');
    const meta = el('div', 'entry-meta');
    meta.appendChild(el('span', 'who', 'System'));
    const body = el('div', 'entry-body');
    body.appendChild(el('p', null, text));
    entry.appendChild(meta);
    entry.appendChild(body);
    log.appendChild(entry);
    scrollToBottom();
  }

  function scrollToBottom() {
    const panel = log.parentElement;
    panel.scrollTop = panel.scrollHeight;
  }

  function renderClarification(data) {
    const children = [];
    children.push(el('p', null, data.message));
    if (data.clarifying_questions && data.clarifying_questions.length) {
      const ul = el('ul');
      data.clarifying_questions.forEach(q => ul.appendChild(el('li', null, q)));
      children.push(ul);
    }
    return children;
  }

  function renderRecommendations(data) {
    const children = [];

    if (data.cross_metric_notes && data.cross_metric_notes.length) {
      const traceDetails = document.createElement('details');
      traceDetails.className = 'trace-toggle';
      const summary = el('summary', null, `Cross-metric reasoning (${data.cross_metric_notes.length})`);
      traceDetails.appendChild(summary);
      const ul = el('ul');
      data.cross_metric_notes.forEach(n => ul.appendChild(el('li', null, n)));
      traceDetails.appendChild(ul);
      children.push(traceDetails);
    }

    if (data.reasoning_trace && data.reasoning_trace.length) {
      const traceDetails = document.createElement('details');
      traceDetails.className = 'trace-toggle';
      const summary = el('summary', null, 'Reasoning trace');
      traceDetails.appendChild(summary);
      const ul = el('ul');
      data.reasoning_trace.forEach(n => ul.appendChild(el('li', null, n)));
      traceDetails.appendChild(ul);
      children.push(traceDetails);
    }

    data.recommendations.forEach(rec => {
      const card = el('div', 'rec');
      card.dataset.horizon = rec.time_horizon;

      card.appendChild(el('p', 'rec-action', rec.recommendation));

      const meta = el('div', 'rec-meta');
      meta.appendChild(el('span', null, rec.time_horizon + ' term'));
      meta.appendChild(el('span', null, rec.confidence + ' confidence'));
      rec.impacted_metrics.slice(0, 3).forEach(m => {
        meta.appendChild(el('span', null, m.replace(/_/g, ' ')));
      });
      card.appendChild(meta);

      const details = document.createElement('details');
      const summary = el('summary', null, 'Why it works · expected effect');
      details.appendChild(summary);
      details.appendChild(el('p', null, rec.why_it_works));
      details.appendChild(el('p', null, rec.expected_effect));
      card.appendChild(details);

      card.appendChild(el('p', 'rec-source', 'Source: ' + rec.source));

      children.push(card);
    });

    return children;
  }

  async function sendMessage(text) {
    addUserEntry(text);
    input.value = '';
    sendBtn.disabled = true;

    const thinking = addSystemEntry([el('p', null, 'Retrieving and reasoning…')]);

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data = await res.json();
      thinking.remove();

      if (!res.ok) {
        addErrorEntry(data.error || 'Something went wrong on the server.');
        return;
      }

      if (data.status === 'needs_clarification') {
        addSystemEntry(renderClarification(data));
      } else {
        addSystemEntry(renderRecommendations(data));
      }
    } catch (err) {
      thinking.remove();
      addErrorEntry('Could not reach the server. Is app_api.py running?');
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    sendMessage(text);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  document.querySelectorAll('[data-fill]').forEach(btn => {
    btn.addEventListener('click', () => {
      input.value = btn.dataset.fill;
      input.focus();
    });
  });

  resetBtn.addEventListener('click', () => {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, sessionId);
    log.innerHTML = '';
    addSystemEntry([
      el('p', null, "Session cleared. Tell me about a piece of land — condition of the soil, rainfall, what's growing on it."),
    ]);
  });

  // ---- Knowledge base preview table ----
  async function loadKnowledgeBase() {
    const container = document.getElementById('kbTable');
    try {
      const res = await fetch('/api/knowledge');
      const data = await res.json();
      container.innerHTML = '';

      const head = el('div', 'kb-row kb-row-head');
      head.appendChild(el('span', null, 'Intervention'));
      head.appendChild(el('span', null, 'Mechanism'));
      head.appendChild(el('span', null, 'Source'));
      container.appendChild(head);

      data.interventions.forEach(iv => {
        const row = el('div', 'kb-row');

        const nameCol = el('div');
        nameCol.appendChild(el('div', 'kb-name', iv.name));
        const tag = el('span', 'kb-tag', iv.connected_variables.join(' · ').replace(/_/g, ' '));
        nameCol.querySelector('.kb-name').appendChild(tag);
        row.appendChild(nameCol);

        const mechCol = el('div');
        mechCol.appendChild(el('p', 'kb-mechanism', iv.mechanism));
        const metrics = el('div', 'kb-metrics');
        iv.impacted_metrics.forEach(m => metrics.appendChild(el('span', null, m.replace(/_/g, ' '))));
        mechCol.appendChild(metrics);
        row.appendChild(mechCol);

        const sourceCol = el('div', 'kb-source', iv.source);
        row.appendChild(sourceCol);

        container.appendChild(row);
      });
    } catch (err) {
      container.innerHTML = '';
      container.appendChild(el('p', 'kb-loading', 'Could not load the knowledge base. Is app_api.py running?'));
    }
  }

  loadKnowledgeBase();
})();
