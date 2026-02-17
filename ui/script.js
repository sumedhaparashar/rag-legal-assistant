/**
 * Legal RAG Assistant – Frontend Logic
 *
 * Connects to the FastAPI backend, manages conversation history in-memory,
 * and renders a ChatGPT-style chat interface.
 */

// ── API configuration ────────────────────────────────
// When served via FastAPI's /ui mount the origin is the same server,
// so a simple relative path works in both local dev and production.
const API_URL = "/ask";

// ── DOM references ───────────────────────────────────
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");

// ── Conversation history (kept for future multi-turn) ──
const history = [];

// ── Suggestion chips (auto-fill the input) ───────────
document.getElementById("suggestions").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  input.value = chip.dataset.q;
  input.focus();
});

// ── Form submit handler ─────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = input.value.trim();
  if (!question) return;               // guard: empty input

  // Remove the welcome screen on the first question
  if (welcome && welcome.parentNode) {
    welcome.remove();
  }

  // Save to history & render the user bubble
  history.push({ role: "user", content: question });
  renderMessage(question, "user");

  // Reset input and show loading state
  input.value = "";
  setLoading(true);
  const loader = showLoader();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    // Remove loader before rendering anything
    loader.remove();

    if (!res.ok) {
      // Server returned an HTTP error (e.g. 500, 503)
      const errBody = await res.json().catch(() => null);
      const detail = errBody?.detail || `Status ${res.status}`;
      throw new Error(detail);
    }

    const data = await res.json();

    // Guard: empty / missing answer
    if (!data.answer) {
      throw new Error("The server returned an empty answer.");
    }

    // Save to history & render assistant response with sources
    history.push({ role: "assistant", content: data.answer, sources: data.sources });
    renderAssistant(data);

  } catch (err) {
    loader.remove();
    renderMessage(
      `⚠ ${err.message || "Network error — is the backend running?"}`,
      "error"
    );
  } finally {
    setLoading(false);
    input.focus();
  }
});

// ── Enter-to-send (Shift+Enter reserved for future textarea) ──
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ═══════════════════════════════════════════════════════
//  Rendering helpers
// ═══════════════════════════════════════════════════════

/**
 * Render a simple text message (user or error).
 * @param {string} text  – message content
 * @param {"user"|"error"} role
 */
function renderMessage(text, role) {
  const group = document.createElement("div");
  group.className = `msg-group ${role}`;

  // Role label
  const label = document.createElement("div");
  label.className = "msg-role";
  label.textContent = role === "user" ? "You" : "Error";
  group.appendChild(label);

  // Bubble
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  group.appendChild(bubble);

  // Timestamp
  group.appendChild(makeTimestamp());

  chat.appendChild(group);
  scrollToBottom();
}

/**
 * Render the assistant's answer along with an optional sources card.
 * @param {{ answer: string, sources: Array }} data
 */
function renderAssistant(data) {
  const group = document.createElement("div");
  group.className = "msg-group assistant";

  // Role label
  const label = document.createElement("div");
  label.className = "msg-role";
  label.textContent = "Assistant";
  group.appendChild(label);

  // Answer bubble
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = data.answer;

  // Sources card (rendered inside the bubble)
  if (data.sources && data.sources.length > 0) {
    bubble.appendChild(buildSourcesCard(data.sources));
  }

  group.appendChild(bubble);

  // Timestamp
  group.appendChild(makeTimestamp());

  chat.appendChild(group);
  scrollToBottom();
}

/**
 * Build a sources card element from the sources array.
 * @param {Array<{file:string, page:number, snippet:string}>} sources
 * @returns {HTMLElement}
 */
function buildSourcesCard(sources) {
  const card = document.createElement("div");
  card.className = "sources-card";

  // Heading with a small icon
  const heading = document.createElement("div");
  heading.className = "sources-heading";
  heading.innerHTML =
    `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
       <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
       <polyline points="14 2 14 8 20 8"/>
     </svg>
     Sources (${sources.length})`;
  card.appendChild(heading);

  // Individual source items
  sources.forEach((src) => {
    const item = document.createElement("div");
    item.className = "src-item";

    // File name
    const file = document.createElement("span");
    file.className = "src-file";
    file.textContent = src.file || "Unknown file";
    item.appendChild(file);

    // Page number
    if (src.page != null) {
      const page = document.createElement("span");
      page.className = "src-page";
      page.textContent = `— p. ${src.page}`;
      item.appendChild(page);
    }

    // Snippet
    if (src.snippet) {
      const snippet = document.createElement("div");
      snippet.className = "src-snippet";
      snippet.textContent = `"${src.snippet}"`;
      item.appendChild(snippet);
    }

    card.appendChild(item);
  });

  return card;
}

// ── Loading dots ────────────────────────────────────
function showLoader() {
  const group = document.createElement("div");
  group.className = "msg-group loader-group";

  const label = document.createElement("div");
  label.className = "msg-role";
  label.textContent = "Assistant";
  label.style.color = "var(--green)";
  group.appendChild(label);

  const loader = document.createElement("div");
  loader.className = "loader";
  loader.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  group.appendChild(loader);

  chat.appendChild(group);
  scrollToBottom();
  return group;
}

// ── Timestamp helper ────────────────────────────────
function makeTimestamp() {
  const el = document.createElement("div");
  el.className = "msg-time";
  const now = new Date();
  el.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return el;
}

// ── Toggle loading state ────────────────────────────
function setLoading(on) {
  sendBtn.disabled = on;
  input.disabled = on;
}

// ── Scroll chat to bottom ───────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    chat.scrollTop = chat.scrollHeight;
  });
}
