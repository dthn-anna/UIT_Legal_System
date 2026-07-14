const API_BASE = "/api/v1";
const DEFAULT_TOP_K = 6;

const messagesContainer = document.getElementById("messages-container");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-btn");
const newChatButton = document.getElementById("new-chat-btn");
const statusIndicator = document.getElementById("status-indicator");

document.addEventListener("DOMContentLoaded", () => {
  resetChat();
  checkBackendHealth();
  userInput.focus();
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage();
});

userInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    await sendMessage();
  }
});

userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = `${Math.min(userInput.scrollHeight, 160)}px`;
});

newChatButton.addEventListener("click", () => {
  resetChat();
  userInput.focus();
});

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAnswer(value) {
  return escapeHtml(value)
    .replaceAll(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replaceAll(/\n/g, "<br>");
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function setSubmitting(isSubmitting) {
  userInput.disabled = isSubmitting;
  sendButton.disabled = isSubmitting;
  sendButton.textContent = isSubmitting ? "Đang xử lý..." : "Gửi";
}

function welcomeMessageHtml() {
  return `
    <div class="message system-message">
      <div class="avatar">AI</div>

      <div class="message-body">
        <p>
          Xin chào! Tôi là trợ lý hỗ trợ tra cứu quy định học vụ.
        </p>

        <p>
          Hệ thống sử dụng E5, BM25 và RRF để tìm đúng điều khoản,
          sau đó Qwen qua Ollama tổng hợp câu trả lời.
        </p>

        <div class="quick-queries">
          <button
            type="button"
            data-query="Sinh viên bị cảnh báo học tập trong trường hợp nào?"
          >
            Khi nào sinh viên bị cảnh báo học tập?
          </button>

          <button
            type="button"
            data-query="Sinh viên được bảo lưu kết quả học tập trong trường hợp nào?"
          >
            Điều kiện bảo lưu kết quả học tập?
          </button>

          <button
            type="button"
            data-query="Sinh viên bị buộc thôi học trong trường hợp nào?"
          >
            Khi nào sinh viên bị buộc thôi học?
          </button>
        </div>
      </div>
    </div>
  `;
}

function bindQuickQueries() {
  document
    .querySelectorAll("[data-query]")
    .forEach((button) => {
      button.addEventListener("click", async () => {
        userInput.value = button.dataset.query || "";
        await sendMessage();
      });
    });
}

function resetChat() {
  messagesContainer.innerHTML = welcomeMessageHtml();
  bindQuickQueries();
  scrollToBottom();
}

function appendUserMessage(question) {
  const message = document.createElement("div");
  message.className = "message user-message";

  message.innerHTML = `
    <div class="avatar">U</div>

    <div class="message-body">
      <p>${escapeHtml(question)}</p>
    </div>
  `;

  messagesContainer.appendChild(message);
  scrollToBottom();
}

function appendLoadingMessage() {
  const message = document.createElement("div");
  message.className = "message system-message";
  message.id = "chat-loading-indicator";

  message.innerHTML = `
    <div class="avatar">AI</div>

    <div class="message-body">
      <div class="loader-content">
        <div class="loader-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>

        <span>Đang tìm quy định và tạo câu trả lời...</span>
      </div>
    </div>
  `;

  messagesContainer.appendChild(message);
  scrollToBottom();
}

function removeLoadingMessage() {
  document.getElementById("chat-loading-indicator")?.remove();
}

function buildSourcesHtml(sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return "";
  }

  const items = sources.map((source, index) => {
    const denseRank = source.dense_rank ?? "-";
    const bm25Rank = source.bm25_rank ?? "-";
    const denseScore = source.dense_score == null
      ? "-"
      : Number(source.dense_score).toFixed(4);
    const bm25Score = source.bm25_score == null
      ? "-"
      : Number(source.bm25_score).toFixed(4);

    return `
      <details class="source-card">
        <summary>
          <span class="source-number">${index + 1}</span>

          <span class="source-title">
            ${escapeHtml(source.document || "Không rõ văn bản")}
            ${source.article ? `— ${escapeHtml(source.article)}` : ""}
          </span>
        </summary>

        <div class="source-content">
          <p>${escapeHtml(source.content || "")}</p>

          <div class="source-metrics">
            <span>Dense rank: ${denseRank}</span>
            <span>BM25 rank: ${bm25Rank}</span>
            <span>Dense score: ${denseScore}</span>
            <span>BM25 score: ${bm25Score}</span>
          </div>
        </div>
      </details>
    `;
  }).join("");

  return `
    <div class="sources-box">
      <div class="sources-header">
        Nguồn quy định tham khảo
      </div>

      <div class="sources-list">
        ${items}
      </div>
    </div>
  `;
}

function appendAssistantMessage(answer, sources) {
  const message = document.createElement("div");
  message.className = "message system-message";

  message.innerHTML = `
    <div class="avatar">AI</div>

    <div class="message-body">
      <div class="answer-content">
        ${formatAnswer(answer || "Không nhận được câu trả lời.")}
      </div>

      ${buildSourcesHtml(sources)}
    </div>
  `;

  messagesContainer.appendChild(message);
  scrollToBottom();
}

function appendErrorMessage(messageText) {
  const message = document.createElement("div");
  message.className = "message system-message";

  message.innerHTML = `
    <div class="avatar error-avatar">!</div>

    <div class="message-body error-message">
      <strong>Không thể xử lý yêu cầu.</strong>
      <p>${escapeHtml(messageText)}</p>
    </div>
  `;

  messagesContainer.appendChild(message);
  scrollToBottom();
}

async function sendMessage() {
  const question = userInput.value.trim();

  if (!question || sendButton.disabled) {
    return;
  }

  appendUserMessage(question);

  userInput.value = "";
  userInput.style.height = "auto";

  setSubmitting(true);
  appendLoadingMessage();

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: DEFAULT_TOP_K,
        generate: true,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail
        || `Máy chủ trả về mã lỗi ${response.status}.`
      );
    }

    removeLoadingMessage();

    appendAssistantMessage(
      data.answer,
      data.sources,
    );
  } catch (error) {
    removeLoadingMessage();

    appendErrorMessage(
      error instanceof Error
        ? error.message
        : "Không thể kết nối tới FastAPI."
    );
  } finally {
    setSubmitting(false);
    userInput.focus();
  }
}

async function checkBackendHealth() {
  statusIndicator.innerHTML = `
    <span class="dot checking"></span>
    Đang kiểm tra hệ thống...
  `;

  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error("Health check failed.");
    }

    if (data.status === "ok") {
      statusIndicator.innerHTML = `
        <span class="dot online"></span>
        E5, BM25 và Ollama đã sẵn sàng
      `;
      return;
    }

    if (data.model_loaded && data.bm25_loaded && !data.ollama_reachable) {
      statusIndicator.innerHTML = `
        <span class="dot warning"></span>
        Retriever hoạt động, Ollama chưa kết nối
      `;
      return;
    }

    statusIndicator.innerHTML = `
      <span class="dot warning"></span>
      Hệ thống đang ở trạng thái degraded
    `;
  } catch (error) {
    statusIndicator.innerHTML = `
      <span class="dot offline"></span>
      Không kết nối được FastAPI
    `;
  }
}
