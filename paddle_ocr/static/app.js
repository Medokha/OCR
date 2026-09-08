(() => {
  const input = document.getElementById("pdf-input");
  const runBtn = document.getElementById("run-btn");
  const fileLabel = document.getElementById("file-label");
  const statusEl = document.getElementById("status");
  const statusTextEl = document.getElementById("status-text");
  const errorEl = document.getElementById("error");
  const resultsEl = document.getElementById("results");
  const outputEl = document.getElementById("output");
  const pagesEl = document.getElementById("pages");
  const metaEl = document.getElementById("meta");
  const copyBtn = document.getElementById("copy-btn");
  const downloadBtn = document.getElementById("download-btn");

  let selectedFile = null;
  let lastText = "";
  let lastName = "ocr-result.txt";
  let pageTexts = [];

  function showError(message) {
    errorEl.hidden = false;
    errorEl.textContent = message;
  }

  function clearError() {
    errorEl.hidden = true;
    errorEl.textContent = "";
  }

  function setBusy(busy) {
    runBtn.disabled = busy || !selectedFile;
    input.disabled = busy;
    if (!busy) {
      statusEl.hidden = true;
    }
  }

  function setStatus(message) {
    statusEl.hidden = false;
    statusTextEl.textContent = message;
  }

  function rebuildFullText() {
    lastText = pageTexts
      .map((text, idx) => {
        if (!text) return "";
        return `—— الصفحة ${idx + 1} ——\n${text}`;
      })
      .filter(Boolean)
      .join("\n\n");
    outputEl.textContent = lastText || "(لم يُعثر على نص قابل للقراءة بعد)";
  }

  function appendPage(page) {
    const idx = page.page_index;
    pageTexts[idx] = page.text || "";
    rebuildFullText();

    if (!page.text) return;

    const block = document.createElement("article");
    block.className = "page page-live";
    const title = document.createElement("h3");
    title.textContent = `الصفحة ${page.page_number}${
      page.avg_score != null ? ` · ثقة ${Math.round(page.avg_score * 100)}%` : ""
    }`;
    const body = document.createElement("p");
    body.textContent = page.text;
    block.append(title, body);
    pagesEl.appendChild(block);
    block.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  input.addEventListener("change", () => {
    clearError();
    resultsEl.hidden = true;
    const file = input.files && input.files[0];
    if (!file) {
      selectedFile = null;
      runBtn.disabled = true;
      fileLabel.textContent = "لم يتم اختيار ملف بعد";
      return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      selectedFile = null;
      runBtn.disabled = true;
      showError("يُقبل ملف PDF فقط.");
      return;
    }
    selectedFile = file;
    runBtn.disabled = false;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    fileLabel.textContent = `${file.name} · ${sizeMb} ميجابايت`;
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    clearError();
    pageTexts = [];
    pagesEl.innerHTML = "";
    outputEl.textContent = "";
    metaEl.textContent = "";
    resultsEl.hidden = false;
    setBusy(true);
    setStatus("جاري تجهيز الملف…");

    const form = new FormData();
    form.append("file", selectedFile);
    lastName = `${selectedFile.name.replace(/\.pdf$/i, "")}-ocr.txt`;

    try {
      const response = await fetch("/api/ocr/stream", {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = payload.detail;
        const message = Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join(" · ")
          : detail || "تعذر إكمال القراءة.";
        throw new Error(message);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let newlineIdx;
        while ((newlineIdx = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newlineIdx).trim();
          buffer = buffer.slice(newlineIdx + 1);
          if (!line) continue;

          const event = JSON.parse(line);
          if (event.type === "start") {
            setStatus(`بدء القراءة · ${event.page_count} صفحة`);
            metaEl.textContent = `${event.filename} · 0 / ${event.page_count}`;
          } else if (event.type === "page") {
            appendPage(event.page);
            setStatus(
              `جاري قراءة الصفحة ${event.pages_done} من ${event.page_count}…`
            );
            metaEl.textContent = `${event.filename} · ${event.pages_done} / ${event.page_count} · ${event.line_count} سطر`;
          } else if (event.type === "done") {
            setStatus("اكتملت القراءة");
            metaEl.textContent = `${event.filename} · ${event.page_count} صفحة · ${event.line_count} سطر`;
            rebuildFullText();
          } else if (event.type === "error") {
            throw new Error(event.detail || "فشل استخراج النص.");
          }
        }
      }
    } catch (err) {
      showError(err.message || "حدث خطأ غير متوقع.");
    } finally {
      setBusy(false);
    }
  });

  copyBtn.addEventListener("click", async () => {
    if (!lastText) return;
    try {
      await navigator.clipboard.writeText(lastText);
      copyBtn.textContent = "تم النسخ";
      setTimeout(() => {
        copyBtn.textContent = "نسخ النص";
      }, 1400);
    } catch {
      showError("تعذر النسخ إلى الحافظة.");
    }
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastText) return;
    const blob = new Blob([lastText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = lastName;
    a.click();
    URL.revokeObjectURL(url);
  });
})();
