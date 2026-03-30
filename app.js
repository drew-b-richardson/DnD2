'use strict';

// ===== CONSTANTS =====
const OLLAMA_BASE = 'http://localhost:11434';
const DEFAULT_MODEL = 'dolphin-mistral:latest';
const STORAGE_KEY = 'dnd-session-v1';
const STATUS_POLL_MS = 30_000;

// System prompt is defined in the Modelfile. Leave empty to use the Modelfile's
// SYSTEM directive as-is. Add text here only to override or extend it at runtime.
const DEFAULT_SYSTEM_PROMPT = '';

// ===== STATE =====
const state = {
  messages: [],         // [{role, content, timestamp}]
  isStreaming: false,
  currentModel: DEFAULT_MODEL,
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  adventureLog: [],     // [{timestamp, text}]
  diceHistory: [],      // [{notation, result, breakdown}]
};

// ===== DOM REFS =====
const $ = id => document.getElementById(id);
const chatWindow        = $('chat-window');
const chatInput         = $('chat-input');
const btnSend           = $('btn-send');
const btnSettings       = $('btn-settings');
const btnToggleLog      = $('btn-toggle-log');
const btnToggleDice     = $('btn-toggle-dice');
const modelSelectHeader = $('model-select-header');
const modelSelectModal  = $('model-select-modal');
const systemPromptInput = $('system-prompt-input');
const statusDot         = $('status-dot');
const adventureLogPanel = $('adventure-log-panel');
const dicePanel         = $('dice-panel');
const logEntries        = $('adventure-log-entries');
const logNoteInput      = $('log-note-input');
const btnAddNote        = $('btn-add-note');
const btnSummarize      = $('btn-summarize');
const diceResultValue   = $('dice-result-value');
const diceResultNotation= $('dice-result-notation');
const diceResultBreakdown=$('dice-result-breakdown');
const diceCustomInput   = $('dice-custom-input');
const btnRollCustom     = $('btn-roll-custom');
const diceHistoryEl     = $('dice-history');
const resumeModal       = $('resume-modal');
const btnResume         = $('btn-resume');
const btnNewAdventure   = $('btn-new-adventure');
const resumeInfo        = $('resume-info');
const settingsModal     = $('settings-modal');
const btnSettingsSave   = $('btn-settings-save');
const btnSettingsCancel = $('btn-settings-cancel');
const btnExport         = $('btn-export');
const btnClearHistory   = $('btn-clear-history');
const chatEmptyState    = $('chat-empty-state');
const quickStarts       = $('quick-starts');

// ===== UTILITIES =====
function formatTime(ts) {
  return new Date(ts || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDateTime(ts) {
  return new Date(ts).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit'
  });
}

function escapeHTML(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Lightweight markdown renderer for DM messages
function renderMarkdown(text) {
  const lines = text.split('\n');
  const out = [];
  let inParagraph = false;

  const closeParagraph = () => {
    if (inParagraph) { out.push('</p>'); inParagraph = false; }
  };

  const inlineFormat = str =>
    escapeHTML(str)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Highlight dice cue phrases
      .replace(/\b(roll(?:s)? (?:a |an )?(?:\d+)?d\d+(?:[+-]\d+)?|make (?:a |an )?\w+ (?:check|save|saving throw)|roll (?:for )?initiative)/gi,
        m => `<span class="dice-cue" data-dice="${parseDiceFromPhrase(m)}">${m}</span>`);

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (/^#{1,3}\s+/.test(line)) {
      closeParagraph();
      const heading = line.replace(/^#{1,3}\s+/, '');
      out.push(`<h3>${inlineFormat(heading)}</h3>`);
    } else if (line === '') {
      closeParagraph();
    } else {
      if (!inParagraph) { out.push('<p>'); inParagraph = true; }
      else out.push(' ');
      out.push(inlineFormat(line));
    }
  }
  closeParagraph();
  return out.join('');
}

// Extract the most relevant die from a dice cue phrase
function parseDiceFromPhrase(phrase) {
  const match = phrase.match(/(\d*d\d+(?:[+-]\d+)?)/i);
  if (match) return match[1].toLowerCase();
  if (/initiative/i.test(phrase)) return 'd20';
  if (/\bcheck\b/i.test(phrase)) return 'd20';
  if (/\bsave\b|\bsaving\b/i.test(phrase)) return 'd20';
  return 'd20';
}

// ===== SESSION PERSISTENCE =====
function saveSession() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      messages: state.messages,
      adventureLog: state.adventureLog,
      systemPrompt: state.systemPrompt,
      currentModel: state.currentModel,
      savedAt: Date.now(),
    }));
  } catch (e) { /* Storage full or unavailable */ }
}

function loadSavedSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch { return null; }
}

function clearSession() {
  localStorage.removeItem(STORAGE_KEY);
  state.messages = [];
  state.adventureLog = [];
  chatWindow.innerHTML = '';
  chatWindow.appendChild(chatEmptyState);
  chatEmptyState.style.display = '';
  logEntries.innerHTML = '';
}

// ===== OLLAMA CONNECTION =====
async function checkOllamaStatus() {
  try {
    const res = await fetch(`${OLLAMA_BASE}/`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      statusDot.className = 'status-indicator connected';
      statusDot.title = 'Ollama is running';
    } else {
      throw new Error('non-ok');
    }
  } catch {
    statusDot.className = 'status-indicator disconnected';
    statusDot.title = 'Ollama not responding — run: ollama serve';
  }
}

async function fetchModels() {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return;
    const data = await res.json();
    const models = (data.models || []).map(m => m.name);
    if (models.length === 0) return;

    // Prefer dnd-dm if available, else dolphin-mistral, else first model
    const preferred = ['dnd:latest', 'dolphin-mistral:latest'];
    let defaultModel = models[0];
    for (const p of preferred) {
      if (models.includes(p)) { defaultModel = p; break; }
    }
    state.currentModel = defaultModel;

    [modelSelectHeader, modelSelectModal].forEach(sel => {
      sel.innerHTML = '';
      models.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === state.currentModel) opt.selected = true;
        sel.appendChild(opt);
      });
    });
  } catch {
    // Ollama not available; leave default option
    [modelSelectHeader, modelSelectModal].forEach(sel => {
      sel.innerHTML = `<option value="${DEFAULT_MODEL}">${DEFAULT_MODEL}</option>`;
    });
  }
}

// ===== MESSAGE RENDERING =====
function hideChatEmptyState() {
  chatEmptyState.style.display = 'none';
}

function createMessageEl(role, content, timestamp) {
  const isDM = role === 'assistant';
  const wrapper = document.createElement('div');
  wrapper.className = `message ${isDM ? 'dm-message' : 'player-message'}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = isDM ? 'DM' : 'You';

  const body = document.createElement('div');
  body.className = 'message-body';

  const header = document.createElement('div');
  header.className = 'message-header';
  header.textContent = isDM ? 'Dungeon Master' : 'Adventurer';

  const contentEl = document.createElement('div');
  contentEl.className = 'message-content';
  if (isDM) {
    contentEl.innerHTML = renderMarkdown(content);
  } else {
    contentEl.textContent = content;
  }

  const ts = document.createElement('div');
  ts.className = 'message-timestamp';
  ts.textContent = formatTime(timestamp);

  body.appendChild(header);
  body.appendChild(contentEl);
  body.appendChild(ts);
  wrapper.appendChild(avatar);
  wrapper.appendChild(body);

  // Wire up dice cues
  contentEl.querySelectorAll('.dice-cue').forEach(el => {
    el.addEventListener('click', () => {
      const notation = el.dataset.dice || 'd20';
      rollDice(notation);
      diceCustomInput.value = notation;
    });
  });

  return wrapper;
}

function appendMessage(role, content, timestamp) {
  hideChatEmptyState();
  const el = createMessageEl(role, content, timestamp || Date.now());
  chatWindow.appendChild(el);
  scrollToBottom();
  return el;
}

function createStreamingMessageEl() {
  hideChatEmptyState();
  const wrapper = document.createElement('div');
  wrapper.className = 'message dm-message';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = 'DM';

  const body = document.createElement('div');
  body.className = 'message-body';

  const header = document.createElement('div');
  header.className = 'message-header';
  header.textContent = 'Dungeon Master';

  const contentEl = document.createElement('div');
  contentEl.className = 'message-content';

  // Typing dots while waiting for first token
  const dots = document.createElement('div');
  dots.className = 'typing-dots';
  dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  contentEl.appendChild(dots);

  const ts = document.createElement('div');
  ts.className = 'message-timestamp';
  ts.textContent = formatTime();

  body.appendChild(header);
  body.appendChild(contentEl);
  body.appendChild(ts);
  wrapper.appendChild(avatar);
  wrapper.appendChild(body);

  chatWindow.appendChild(wrapper);
  scrollToBottom();

  return { wrapper, contentEl, dots, ts };
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showError(msg) {
  const el = document.createElement('div');
  el.className = 'message-error';
  el.innerHTML = msg;
  chatWindow.appendChild(el);
  scrollToBottom();
}

// ===== OLLAMA STREAMING =====
async function streamFromOllama(messagesSnapshot) {
  state.isStreaming = true;
  btnSend.disabled = true;

  const { contentEl, dots, ts } = createStreamingMessageEl();

  const messages = [
    ...(state.systemPrompt ? [{ role: 'system', content: state.systemPrompt }] : []),
    ...messagesSnapshot.map(m => ({ role: m.role, content: m.content })),
  ];

  let fullText = '';
  let firstToken = false;
  let cursor = null;

  try {
    const response = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: state.currentModel,
        messages,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Ollama returned ${response.status}: ${errText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete trailing fragment

      for (const line of lines) {
        if (!line.trim()) continue;
        let data;
        try { data = JSON.parse(line); } catch { continue; }

        if (data.message?.content) {
          if (!firstToken) {
            firstToken = true;
            dots.remove();
            // Add blinking cursor
            cursor = document.createElement('span');
            cursor.className = 'streaming-cursor';
            contentEl.innerHTML = '';
            contentEl.appendChild(cursor);
          }
          fullText += data.message.content;
          // Update rendered content, keep cursor at end
          contentEl.innerHTML = renderMarkdown(fullText);
          contentEl.appendChild(cursor);
          scrollToBottom();
        }

        if (data.done && data.done_reason === 'load') {
          // Model was just loaded into memory — retry the request
          dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
          contentEl.innerHTML = '';
          contentEl.appendChild(dots);
          state.isStreaming = false;
          btnSend.disabled = false;
          const wrapper = contentEl.closest('.message');
          if (wrapper) wrapper.remove();
          await streamFromOllama(messagesSnapshot);
          return;
        }

        if (data.done) {
          // Finalize
          if (cursor) cursor.remove();
          contentEl.innerHTML = renderMarkdown(fullText);

          // Wire dice cues
          contentEl.querySelectorAll('.dice-cue').forEach(el => {
            el.addEventListener('click', () => {
              const notation = el.dataset.dice || 'd20';
              rollDice(notation);
              diceCustomInput.value = notation;
            });
          });

          ts.textContent = formatTime();
          const msgObj = { role: 'assistant', content: fullText, timestamp: Date.now() };
          state.messages.push(msgObj);
          saveSession();
        }
      }
    }

    // Handle any remaining buffer content
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer);
        if (data.message?.content) {
          fullText += data.message.content;
          contentEl.innerHTML = renderMarkdown(fullText);
        }
      } catch { /* incomplete JSON, ignore */ }
    }

    if (!fullText) {
      contentEl.innerHTML = '<em style="color:var(--text-dim)">No response received.</em>';
    }

  } catch (err) {
    if (cursor) cursor.remove();
    dots.style.display = 'none';
    const isNetworkErr = err.message.includes('Failed to fetch') || err.message.includes('NetworkError');
    if (isNetworkErr) {
      showError(
        '<strong>Cannot reach Ollama.</strong> Make sure it\'s running:<br>' +
        '<code>ollama serve</code>'
      );
      contentEl.closest('.message')?.remove();
    } else {
      contentEl.innerHTML = `<span style="color:var(--text-crimson)">Error: ${escapeHTML(err.message)}</span>`;
    }
  } finally {
    state.isStreaming = false;
    btnSend.disabled = false;
    chatInput.focus();
    scrollToBottom();
  }
}

// ===== SEND MESSAGE =====
async function sendMessage(text) {
  text = text.trim();
  if (!text || state.isStreaming) return;

  // Hide quick starts
  if (quickStarts) quickStarts.style.display = 'none';

  const msg = { role: 'user', content: text, timestamp: Date.now() };
  state.messages.push(msg);
  appendMessage('user', text, msg.timestamp);
  chatInput.value = '';
  chatInput.style.height = 'auto';
  saveSession();

  await streamFromOllama(state.messages);
}

// ===== DICE ROLLER =====
function rollDice(notation) {
  const trimmed = notation.trim().toLowerCase();
  const match = trimmed.match(/^(\d*)d(\d+)([+-]\d+)?$/);
  if (!match) {
    diceResultValue.textContent = '?';
    diceResultNotation.textContent = 'Invalid notation';
    diceResultBreakdown.textContent = '';
    return null;
  }

  const count    = Math.max(1, Math.min(100, parseInt(match[1] || '1', 10)));
  const sides    = Math.min(1000, parseInt(match[2], 10));
  const modifier = parseInt(match[3] || '0', 10);

  const rolls = Array.from({ length: count }, () => Math.floor(Math.random() * sides) + 1);
  const subtotal = rolls.reduce((a, b) => a + b, 0);
  const total = subtotal + modifier;

  // Animate result
  diceResultValue.classList.remove('rolling');
  void diceResultValue.offsetWidth; // reflow to restart animation
  diceResultValue.classList.add('rolling');
  diceResultValue.textContent = total;

  const notationDisplay = notation.trim().toLowerCase();
  diceResultNotation.textContent = notationDisplay;

  if (count > 1 || modifier !== 0) {
    const parts = rolls.join(' + ');
    diceResultBreakdown.textContent =
      modifier !== 0
        ? `[${parts}] ${modifier > 0 ? '+' : ''}${modifier} = ${total}`
        : `[${parts}] = ${total}`;
  } else {
    diceResultBreakdown.textContent = '';
  }

  // Add to history
  const entry = { notation: notationDisplay, result: total, breakdown: rolls.join(', ') };
  state.diceHistory.unshift(entry);
  if (state.diceHistory.length > 10) state.diceHistory.pop();
  renderDiceHistory();

  return entry;
}

function renderDiceHistory() {
  diceHistoryEl.innerHTML = '';
  state.diceHistory.slice(0, 8).forEach(entry => {
    const el = document.createElement('div');
    el.className = 'dice-history-entry';
    el.innerHTML = `
      <span class="dice-history-notation">${escapeHTML(entry.notation)}</span>
      <span class="dice-history-result">${entry.result}</span>
    `;
    diceHistoryEl.appendChild(el);
  });
}

// ===== ADVENTURE LOG =====
function addLogEntry(text, timestamp) {
  const ts = timestamp || Date.now();
  const entry = { text, timestamp: ts };
  state.adventureLog.push(entry);
  renderLogEntry(entry);
  saveSession();
}

function renderLogEntry(entry) {
  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `
    <span class="log-entry-time">${formatDateTime(entry.timestamp)}</span>
    <span class="log-entry-text">${escapeHTML(entry.text)}</span>
  `;
  logEntries.appendChild(el);
  logEntries.scrollTop = logEntries.scrollHeight;
}

async function summarizeSession() {
  if (state.messages.length < 2 || state.isStreaming) return;
  const recent = state.messages.slice(-10);
  const prompt = 'In one or two sentences, summarize what just happened in our D&D session. Be concise and in-world.';

  btnSummarize.disabled = true;
  btnSummarize.textContent = '…';

  try {
    const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: state.currentModel,
        messages: [
          { role: 'system', content: 'You summarize D&D sessions briefly.' },
          ...recent.map(m => ({ role: m.role, content: m.content })),
          { role: 'user', content: prompt },
        ],
        stream: false,
      }),
      signal: AbortSignal.timeout(20000),
    });
    if (res.ok) {
      const data = await res.json();
      const summary = data.message?.content?.trim();
      if (summary) addLogEntry(summary);
    }
  } catch { /* ignore */ }

  btnSummarize.disabled = false;
  btnSummarize.textContent = 'Summarize';
}

// ===== EXPORT =====
function exportSession() {
  const lines = [`D&D Session Export — ${new Date().toLocaleString()}`, '='.repeat(60), ''];
  state.messages.forEach(m => {
    const who = m.role === 'assistant' ? 'DUNGEON MASTER' : 'PLAYER';
    lines.push(`[${formatDateTime(m.timestamp)}] ${who}`);
    lines.push(m.content);
    lines.push('');
  });
  if (state.adventureLog.length) {
    lines.push('='.repeat(60));
    lines.push('ADVENTURE LOG');
    lines.push('');
    state.adventureLog.forEach(e => {
      lines.push(`[${formatDateTime(e.timestamp)}] ${e.text}`);
    });
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `dnd-session-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ===== SETTINGS MODAL =====
function openSettings() {
  systemPromptInput.value = state.systemPrompt;
  // Sync model selects
  if (modelSelectModal.value !== state.currentModel) {
    modelSelectModal.value = state.currentModel;
  }
  settingsModal.classList.remove('hidden');
}

function closeSettings() {
  settingsModal.classList.add('hidden');
}

function saveSettings() {
  state.systemPrompt = systemPromptInput.value.trim();
  const newModel = modelSelectModal.value;
  if (newModel) {
    state.currentModel = newModel;
    modelSelectHeader.value = newModel;
  }
  saveSession();
  closeSettings();
}

// ===== AUTO-RESIZE TEXTAREA =====
function autoResizeTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
}

// ===== PANEL TOGGLES (responsive) =====
function togglePanel(panel) {
  panel.classList.toggle('open');
  panel.classList.toggle('collapsed');
}

// ===== RESTORE SESSION UI =====
function restoreMessagesFromState() {
  chatWindow.innerHTML = '';
  chatWindow.appendChild(chatEmptyState);

  if (state.messages.length === 0) {
    chatEmptyState.style.display = '';
    return;
  }

  chatEmptyState.style.display = 'none';
  state.messages.forEach(m => {
    chatWindow.appendChild(createMessageEl(m.role, m.content, m.timestamp));
  });
  scrollToBottom();
}

function restoreLogFromState() {
  logEntries.innerHTML = '';
  state.adventureLog.forEach(e => renderLogEntry(e));
}

// ===== INIT =====
async function init() {
  // 1. Check for saved session
  const saved = loadSavedSession();
  if (saved && saved.messages && saved.messages.length > 0) {
    const count = saved.messages.length;
    const savedAt = formatDateTime(saved.savedAt);
    resumeInfo.textContent = `${count} message${count !== 1 ? 's' : ''} saved on ${savedAt}`;
    resumeModal.classList.remove('hidden');

    btnResume.onclick = () => {
      state.messages    = saved.messages;
      state.adventureLog= saved.adventureLog || [];
      state.systemPrompt= saved.systemPrompt || '';
      state.currentModel= saved.currentModel || DEFAULT_MODEL;
      resumeModal.classList.add('hidden');
      restoreMessagesFromState();
      restoreLogFromState();
      if (quickStarts) quickStarts.style.display = 'none';
    };

    btnNewAdventure.onclick = () => {
      clearSession();
      resumeModal.classList.add('hidden');
    };
  }

  // 2. Check Ollama & fetch models
  await Promise.all([checkOllamaStatus(), fetchModels()]);

  // 3. Poll status
  setInterval(checkOllamaStatus, STATUS_POLL_MS);

  // 4. Wire up events

  // Send button
  btnSend.addEventListener('click', () => sendMessage(chatInput.value));

  // Enter key
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(chatInput.value);
    }
  });

  // Auto-resize
  chatInput.addEventListener('input', autoResizeTextarea);

  // Model selector in header
  modelSelectHeader.addEventListener('change', () => {
    state.currentModel = modelSelectHeader.value;
    modelSelectModal.value = state.currentModel;
  });

  // Quick starts
  document.querySelectorAll('.quick-chip').forEach(chip => {
    chip.addEventListener('click', () => sendMessage(chip.textContent));
  });

  // Settings
  btnSettings.addEventListener('click', openSettings);
  btnSettingsSave.addEventListener('click', saveSettings);
  btnSettingsCancel.addEventListener('click', closeSettings);
  settingsModal.addEventListener('click', e => {
    if (e.target === settingsModal) closeSettings();
  });

  // Clear history
  btnClearHistory.addEventListener('click', () => {
    if (confirm('Clear all messages? This cannot be undone.')) {
      clearSession();
      closeSettings();
    }
  });

  // Export
  btnExport.addEventListener('click', exportSession);

  // Adventure log
  btnAddNote.addEventListener('click', () => {
    const text = logNoteInput.value.trim();
    if (text) { addLogEntry(text); logNoteInput.value = ''; }
  });
  logNoteInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const text = logNoteInput.value.trim();
      if (text) { addLogEntry(text); logNoteInput.value = ''; }
    }
  });
  btnSummarize.addEventListener('click', summarizeSession);

  // Dice buttons
  document.querySelectorAll('.dice-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      rollDice(btn.dataset.dice);
      diceCustomInput.value = btn.dataset.dice;
    });
  });

  btnRollCustom.addEventListener('click', () => {
    const notation = diceCustomInput.value.trim();
    if (notation) rollDice(notation);
  });

  diceCustomInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const notation = diceCustomInput.value.trim();
      if (notation) rollDice(notation);
    }
  });

  // Panel toggles (responsive)
  btnToggleLog.addEventListener('click', () => {
    if (window.innerWidth <= 900) {
      adventureLogPanel.classList.toggle('open');
    } else {
      adventureLogPanel.classList.toggle('collapsed');
    }
  });

  btnToggleDice.addEventListener('click', () => {
    if (window.innerWidth <= 900) {
      dicePanel.classList.toggle('open');
    } else {
      dicePanel.classList.toggle('collapsed');
    }
  });

  // Escape closes modals
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeSettings();
      resumeModal.classList.add('hidden');
    }
  });

  // Ctrl+D focuses dice input
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
      e.preventDefault();
      diceCustomInput.focus();
    }
  });

  // Click outside to close mobile panels
  document.addEventListener('click', e => {
    if (window.innerWidth > 900) return;
    if (!adventureLogPanel.contains(e.target) && !btnToggleLog.contains(e.target)) {
      adventureLogPanel.classList.remove('open');
    }
    if (!dicePanel.contains(e.target) && !btnToggleDice.contains(e.target)) {
      dicePanel.classList.remove('open');
    }
  });

  // Focus input on load
  chatInput.focus();
}

document.addEventListener('DOMContentLoaded', init);
