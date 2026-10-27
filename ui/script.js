/* ============================================================
   Corporate Law Assistant — Main Application Script
   Modules: Auth · Chat · UI
   ============================================================ */

// ── Global State ────────────────────────────────────────────
let supabaseClient = null;
let currentUser = null;
let authEnabled = false;
let isLoading = false;
let chatHistory = [];
let lastQuestion = '';

// ══════════════════════════════════════════════════════════════
//  AUTH MODULE
// ══════════════════════════════════════════════════════════════

const Auth = {
  /**
   * Fetch /api/auth/config and initialize Supabase if configured.
   * Returns true if auth is enabled, false otherwise.
   */
  async init() {
    try {
      const res = await fetch('/api/auth/config');
      if (!res.ok) throw new Error('Config fetch failed');
      const config = await res.json();

      if (!config.supabase_url) {
        authEnabled = false;
        UI.updateAuthUI();
        return false;
      }

      authEnabled = true;
      supabaseClient = window.supabase.createClient(
        config.supabase_url,
        config.supabase_anon_key
      );

      // Listen for auth state changes
      supabaseClient.auth.onAuthStateChange((event, session) => {
        if (event === 'SIGNED_OUT') {
          currentUser = null;
          window.location.replace('/ui/auth.html');
        }
        if (event === 'SIGNED_IN' && session) {
          currentUser = session.user;
          UI.updateAuthUI();
        }
      });

      return true;
    } catch (err) {
      console.warn('Auth config not available, running without auth:', err);
      authEnabled = false;
      UI.updateAuthUI();
      return false;
    }
  },

  /**
   * Check current session. Redirect to auth.html if not authenticated.
   */
  async checkSession() {
    if (!authEnabled || !supabaseClient) return true;

    try {
      const { data: { session }, error } = await supabaseClient.auth.getSession();
      if (error) throw error;

      if (!session) {
        window.location.replace('/ui/auth.html');
        return false;
      }

      currentUser = session.user;
      UI.updateAuthUI();
      return true;
    } catch (err) {
      console.error('Session check failed:', err);
      window.location.replace('/ui/auth.html');
      return false;
    }
  },

  /**
   * Returns authorization headers for API calls.
   */
  async getAuthHeaders() {
    if (!authEnabled || !supabaseClient) return {};

    try {
      const { data: { session } } = await supabaseClient.auth.getSession();
      if (session?.access_token) {
        return { 'Authorization': `Bearer ${session.access_token}` };
      }
    } catch (err) {
      console.error('Failed to get auth headers:', err);
    }
    return {};
  },

  /**
   * Sign out and redirect.
   */
  async checkAdmin() {
    const adminLink = document.getElementById('admin-link');
    const analyticsLink = document.getElementById('nav-analytics');

    if (adminLink) adminLink.hidden = true;
    if (analyticsLink) analyticsLink.hidden = true;

    if (!authEnabled || !supabaseClient) {
        return false;
    }

    try {
        const response = await fetch('/api/auth/me', {
            headers: await Auth.getAuthHeaders()
        });

        if (!response.ok) return false;

        const data = await response.json();

        if (data.is_admin === true) {
            if (adminLink) adminLink.hidden = false;
            if (analyticsLink) analyticsLink.hidden = false;
            return true;
        }

        return false;
    } catch (error) {
        console.error('Admin verification failed:', error);
        return false;
    }
},
  async signOut() {
    if (supabaseClient) {
      await supabaseClient.auth.signOut();
    }
    currentUser = null;
    window.location.replace('/ui/auth.html');
  }
};

// ══════════════════════════════════════════════════════════════
//  CHAT MODULE
// ══════════════════════════════════════════════════════════════

const Chat = {
  /**
   * Send a question to the /ask endpoint.
   */
  async askQuestion(question) {
    const headers = {
      'Content-Type': 'application/json',
      ...(await Auth.getAuthHeaders())
    };

    const res = await fetch('/ask', {
      method: 'POST',
      headers,
      body: JSON.stringify({ question })
    });

    if (!res.ok) {
      const errBody = await res.text();
      let msg = `Server error (${res.status})`;
      try {
        const parsed = JSON.parse(errBody);
        msg = parsed.detail || parsed.message || msg;
      } catch (_) {}
      throw new Error(msg);
    }

    return await res.json();
  },

  /**
   * Handle form submission.
   */
  async handleSubmit(e) {
    if (e) e.preventDefault();

    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question || isLoading) return;

    lastQuestion = question;
    input.value = '';
    UI.updateSendButton();

    // Hide welcome state
    UI.hideWelcome();

    // Render user message
    UI.renderUserMessage(question);
    UI.scrollToBottom();

    // Show loading
    isLoading = true;
    UI.setInputDisabled(true);
    UI.showLoading();
    UI.scrollToBottom();

    try {
      const data = await Chat.askQuestion(question);
      UI.hideLoading();
      UI.renderAssistantMessage(data);
      Chat.addToHistory('user', question);
      Chat.addToHistory('assistant', data.answer, data.sources);
    } catch (err) {
      UI.hideLoading();
      UI.renderError(err.message || 'Something went wrong. Please try again.');
    } finally {
      isLoading = false;
      UI.setInputDisabled(false);
      UI.scrollToBottom();
      // Auto-focus input
      input.focus();
    }
  },

  /**
   * Track conversation history.
   */
  addToHistory(role, content, sources = null) {
    chatHistory.push({ role, content, sources, timestamp: Date.now() });
  }
};

// ══════════════════════════════════════════════════════════════
//  UI MODULE
// ══════════════════════════════════════════════════════════════

const UI = {
  /**
   * Render a user message bubble.
   */
  renderUserMessage(text) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = 'message message-user';
    msg.innerHTML = `
      <div class="message-content">
        <div class="message-bubble">${UI.escapeHtml(text)}</div>
      </div>
    `;
    container.appendChild(msg);
  },

  /**
   * Render an assistant message with markdown + source cards.
   */
  renderAssistantMessage(data) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = 'message message-assistant';

    // Render markdown safely
    const rawHtml = marked.parse(data.answer || '');
    const cleanHtml = DOMPurify.sanitize(rawHtml);

    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
      sourcesHtml = UI.renderSourceCards(data.sources);
    }

    msg.innerHTML = `
      <div class="message-avatar" aria-hidden="true">⚖️</div>
      <div class="message-content">
        <div class="message-bubble markdown-body">${cleanHtml}</div>
        ${sourcesHtml}
      </div>
    `;

    container.appendChild(msg);

    // Attach source toggle handler
    const toggle = msg.querySelector('.sources-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const list = msg.querySelector('.sources-list');
        const isExpanded = toggle.classList.contains('expanded');
        toggle.classList.toggle('expanded');
        list.classList.toggle('visible');
        toggle.setAttribute('aria-expanded', !isExpanded);
      });
    }
  },

  /**
   * Build source cards HTML.
   *
   * Each card links to /docs/<source_file>#page=<page>, which opens the
   * PDF in a new browser tab and jumps to the exact page. Works in Chrome,
   * Edge, and Firefox via the native #page=N fragment.
   */
  renderSourceCards(sources) {
    const sourceItems = sources.map((src) => {
      const relevance = src.relevance_score != null
        ? `<span class="source-relevance">${Math.round(src.relevance_score * 100)}%</span>`
        : '';
      const page = src.page != null
        ? `<span class="source-page">Page ${src.page}</span>`
        : '';
      const section = src.section
        ? `<span class="source-section">§ ${UI.escapeHtml(src.section)}</span>`
        : '';
      const snippet = src.snippet
        ? `<p class="source-snippet">${UI.escapeHtml(src.snippet)}</p>`
        : '';

      // Build the PDF URL. Prefer source_file (actual disk filename);
      // fall back to the display name if source_file is missing.
      const fileForUrl = src.source_file || src.file || '';
      const canLink = fileForUrl && src.page != null;
      const pdfUrl = canLink
        ? `/docs/${encodeURIComponent(fileForUrl)}#page=${src.page}`
        : null;

      const headerInner = `
        <span class="source-icon">📄</span>
        <span class="source-filename" title="${UI.escapeHtml(src.file || 'Unknown')}">
          ${UI.escapeHtml(src.file || 'Unknown source')}
        </span>
        ${section}
        ${page}
        ${relevance}
      `;

      // If we have a link, wrap header + snippet in <a>; otherwise render plain.
      const inner = pdfUrl
        ? `<a class="source-link" href="${pdfUrl}" target="_blank" rel="noopener noreferrer">
             <div class="source-header">${headerInner}</div>
             ${snippet}
           </a>`
        : `<div class="source-header">${headerInner}</div>${snippet}`;

      return `<div class="source-card">${inner}</div>`;
    }).join('');

    return `
      <div class="sources-container">
        <button class="sources-toggle" aria-expanded="false" aria-label="Toggle source references">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
          ${sources.length} source${sources.length !== 1 ? 's' : ''} referenced
        </button>
        <div class="sources-list" role="list" aria-label="Source documents">
          ${sourceItems}
        </div>
      </div>
    `;
  },

  /**
   * Render an error message with retry option.
   */
  renderError(message) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = 'message message-assistant message-error';
    msg.innerHTML = `
      <div class="message-avatar" aria-hidden="true">⚠️</div>
      <div class="message-content">
        <div class="message-bubble">
          <p>${UI.escapeHtml(message)}</p>
          <button class="btn-retry" id="btn-retry" type="button" aria-label="Retry question">
            ↻ Retry
          </button>
        </div>
      </div>
    `;
    container.appendChild(msg);

    // Attach retry handler
    const retryBtn = msg.querySelector('.btn-retry');
    retryBtn.addEventListener('click', () => {
      // Remove error message
      msg.remove();
      // Re-ask last question
      if (lastQuestion) {
        const input = document.getElementById('question-input');
        input.value = lastQuestion;
        Chat.handleSubmit();
      }
    });
  },

  /**
   * Show loading indicator.
   */
  showLoading() {
    const container = document.getElementById('chat-messages');
    const loader = document.createElement('div');
    loader.className = 'loading-indicator';
    loader.id = 'loading-indicator';
    loader.innerHTML = `
      <div class="message-avatar" aria-hidden="true">⚖️</div>
      <div class="loading-bubble">
        <div class="loading-dots" aria-label="Loading">
          <span></span><span></span><span></span>
        </div>
        <span class="loading-text">Analyzing legal documents…</span>
      </div>
    `;
    container.appendChild(loader);
  },

  /**
   * Hide loading indicator.
   */
  hideLoading() {
    const loader = document.getElementById('loading-indicator');
    if (loader) loader.remove();
  },

  /**
   * Scroll chat to bottom smoothly.
   */
  scrollToBottom() {
    const area = document.getElementById('chat-area');
    requestAnimationFrame(() => {
      area.scrollTo({ top: area.scrollHeight, behavior: 'smooth' });
    });
  },

  /**
   * Hide welcome state.
   */
  hideWelcome() {
    const welcome = document.getElementById('welcome-state');
    if (welcome) {
      welcome.style.display = 'none';
    }
  },

  /**
   * Handle chip click.
   */
  handleChipClick(e) {
    const chip = e.target.closest('.suggestion-chip');
    if (!chip) return;
    const question = chip.dataset.question;
    if (question) {
      const input = document.getElementById('question-input');
      input.value = question;
      Chat.handleSubmit();
    }
  },

  /**
   * Enable/disable input and send button during loading.
   */
  setInputDisabled(disabled) {
    const input = document.getElementById('question-input');
    const btn = document.getElementById('btn-send');
    input.disabled = disabled;
    btn.disabled = disabled;
  },

  /**
   * Update send button state based on input content.
   */
  updateSendButton() {
    const input = document.getElementById('question-input');
    const btn = document.getElementById('btn-send');
    btn.disabled = !input.value.trim() || isLoading;
  },

  /**
   * Update header with user info.
   */
  updateAuthUI() {
    const emailEl = document.getElementById('user-email');
    const signOutBtn = document.getElementById('btn-signout');

    if (!emailEl || !signOutBtn) return;

    if (authEnabled && currentUser) {
      emailEl.textContent = currentUser.email || '';
      emailEl.title = currentUser.email || '';
      signOutBtn.classList.remove('hidden');
    } else if (!authEnabled) {
      emailEl.textContent = '';
      signOutBtn.classList.add('hidden');
    }
  },

  /**
   * Show a toast notification.
   */
  showToast(message, type = 'error') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? '✓' : type === 'warning' ? '⚠' : '✕';
    toast.innerHTML = `<span>${icon}</span><span>${UI.escapeHtml(message)}</span>`;
    container.appendChild(toast);

    // Auto-remove after 5s
    setTimeout(() => {
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  },

  /**
   * Escape HTML to prevent XSS.
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
};

// ══════════════════════════════════════════════════════════════
//  INITIALIZATION
// ══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
  // ── Auth ──
  const authIsEnabled = await Auth.init();
  if (authIsEnabled) {
    const sessionOk = await Auth.checkSession();
    if (!sessionOk) return; // Redirecting…
  }
  await Auth.checkAdmin();

  // ── DOM References ──
  const form = document.getElementById('question-form');
  const input = document.getElementById('question-input');
  const signOutBtn = document.getElementById('btn-signout');
  const welcomeChips = document.getElementById('welcome-state');

  // ── Form Submission ──
  if (form) {
    form.addEventListener('submit', (e) => Chat.handleSubmit(e));
  }

  // ── Input Change → Enable/Disable Send ──
  if (input) {
    input.addEventListener('input', () => UI.updateSendButton());
    input.focus();
  }

  // ── Sign Out ──
  if (signOutBtn) {
    signOutBtn.addEventListener('click', () => Auth.signOut());
  }

  // ── Suggestion Chips ──
  if (welcomeChips) {
    welcomeChips.addEventListener('click', (e) => UI.handleChipClick(e));
  }

  // ── Configure marked.js ──
  if (typeof marked !== 'undefined') {
    marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false
    });
  }
});
