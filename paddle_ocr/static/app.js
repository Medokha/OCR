(() => {
  const input = document.getElementById("pdf-input");
  const runBtn = document.getElementById("run-btn");
  const fileLabel = document.getElementById("file-label");
  const fieldInput = document.getElementById("field-input");
  const addFieldBtn = document.getElementById("add-field-btn");
  const fieldChips = document.getElementById("field-chips");
  const statusEl = document.getElementById("status");
  const statusTextEl = document.getElementById("status-text");
  const errorEl = document.getElementById("error");
  const resultsEl = document.getElementById("results");
  const outputEl = document.getElementById("output");
  const ocrListEl = document.getElementById("ocr-list");
  const pairsJsonEl = document.getElementById("pairs-json");
  const pagesEl = document.getElementById("pages");
  const fieldsEl = document.getElementById("fields");
  const fieldsMetaEl = document.getElementById("fields-meta");
  const mappedFieldsEl = document.getElementById("mapped-fields");
  const mappedMetaEl = document.getElementById("mapped-meta");
  const metaEl = document.getElementById("meta");
  const copyBtn = document.getElementById("copy-btn");
  const copyFieldsBtn = document.getElementById("copy-fields-btn");
  const copyListBtn = document.getElementById("copy-list-btn");
  const downloadBtn = document.getElementById("download-btn");

  let selectedFile = null;
  let requestedFields = [];
  let lastText = "";
  let lastName = "ocr-result.txt";
  let lastFields = [];
  let lastMapped = [];
  let lastOcrList = [];
  let lastPairs = [];
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
    fieldInput.disabled = busy;
    addFieldBtn.disabled = busy;
    if (!busy) statusEl.hidden = true;
  }

  function setStatus(message) {
    statusEl.hidden = false;
    statusTextEl.textContent = message;
  }

  function normalizeLabel(label) {
    return label.replace(/\s+/g, " ").trim();
  }

  function renderChips() {
    fieldChips.innerHTML = "";
    requestedFields.forEach((label, idx) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.title = "إزالة";
      chip.innerHTML = `<span>${label}</span><span class="chip-x" aria-hidden="true">×</span>`;
      chip.addEventListener("click", () => {
        requestedFields.splice(idx, 1);
        renderChips();
      });
      fieldChips.appendChild(chip);
    });
  }

  function addFieldFromInput() {
    const label = normalizeLabel(fieldInput.value || "");
    if (!label) return;
    const exists = requestedFields.some(
      (f) => f.toLocaleLowerCase("ar") === label.toLocaleLowerCase("ar")
    );
    if (!exists) {
      requestedFields.push(label);
      renderChips();
    }
    fieldInput.value = "";
    fieldInput.focus();
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

  function fieldsAsText() {
    const mappedBlock =
      lastMapped.length > 0
        ? [
            "—— نتائج الـ Mapping ——",
            ...lastMapped.map((f) => {
              const vals =
                f.values && f.values.length ? f.values.join(" | ") : "—";
              const idxs = (f.value_indices || []).join(",");
              return `${f.name || "—"}: ${vals} [name_index=${f.name_index ?? "—"} value_indices=${idxs || "—"}]`;
            }),
            "",
          ]
        : [];
    const autoBlock = lastFields.map((f) => {
      const status = f.found ? "موجود" : "غير موجود";
      const idxs =
        f.name_index != null || f.value_index != null
          ? ` name_index=${f.name_index} value_index=${f.value_index}`
          : "";
      return `${f.label}: ${f.value || "—"} [${status}]${idxs}`;
    });
    return [...mappedBlock, ...autoBlock].join("\n");
  }

  function listPayload() {
    return {
      ocr_list: lastOcrList,
      pairs: lastPairs,
      mapped: lastMapped,
    };
  }

  function saveSession() {
    try {
      sessionStorage.setItem(
        "qiraa_ocr_session",
        JSON.stringify({
          ocr_list: lastOcrList,
          pairs: lastPairs,
          fields: lastFields,
          saved_at: new Date().toISOString(),
        })
      );
    } catch {
      /* ignore quota */
    }
  }

  function renderOcrList(ocrList) {
    lastOcrList = ocrList || [];
    if (!ocrListEl) return;
    if (!lastOcrList.length) {
      ocrListEl.textContent = "(فارغ)";
      return;
    }
    ocrListEl.textContent = lastOcrList
      .map((item) => `[${item.index}] صفحة ${item.page}: ${item.text}`)
      .join("\n");
  }

  function renderPairs(extraction) {
    lastPairs = extraction.pairs || extraction.fields.map((f) => ({
      name: f.label,
      value: f.value,
      name_index: f.name_index,
      value_index: f.value_index,
      found: f.found,
    }));
    if (pairsJsonEl) {
      pairsJsonEl.textContent = JSON.stringify(
        { ocr_list: lastOcrList, pairs: lastPairs },
        null,
        2
      );
    }
  }

  function renderMapped(mapped) {
    if (!mappedFieldsEl) return;
    lastMapped = (mapped && mapped.results) || [];
    mappedFieldsEl.innerHTML = "";

    if (!mapped || !mapped.mapping_count) {
      if (mappedMetaEl) {
        mappedMetaEl.textContent =
          "لا يوجد قالب Mapping محفوظ — احفظه من شاشة /map مرة واحدة";
      }
      return;
    }

    const withValues = lastMapped.filter((r) => r.has_values).length;
    const foundNames = lastMapped.filter((r) => r.found_name).length;
    if (mappedMetaEl) {
      mappedMetaEl.textContent = `قالب ثابت · ${mapped.mapping_count} حقل · وُجد الاسم في ${foundNames} · قيم في ${withValues}`;
    }

    lastMapped.forEach((row) => {
      const card = document.createElement("article");
      const ok = row.found_name && row.has_values;
      const partial = row.found_name && !row.has_values;
      card.className = `field-card ${ok ? "is-valid" : partial ? "is-missing" : "is-missing"}`;
      const valuesText =
        row.values && row.values.length ? row.values.join(" | ") : "—";
      const badge = !row.found_name
        ? "الاسم غير موجود"
        : row.has_values
          ? `✓ ${row.values.length} قيمة`
          : "بدون قيمة";
      card.innerHTML = `
        <div class="field-top">
          <p class="field-label">${row.name || "—"}</p>
          <span class="field-badge">${badge}</span>
        </div>
        <p class="field-value ${row.has_values ? "" : "empty"}">${valuesText}</p>
        <p class="field-note">
          offsets = <b>${(row.value_offsets || []).join(", ") || "—"}</b>
          · name_index = <b>${row.name_index ?? "—"}</b>
          · value_indices = <b>${(row.value_indices || []).join(", ") || "—"}</b>
          ${row.page ? ` · صفحة ${row.page}` : ""}
        </p>
      `;
      mappedFieldsEl.appendChild(card);
    });
  }

  function renderFields(extraction) {
    if (!extraction || !extraction.fields) return;
    lastFields = extraction.fields;
    renderOcrList(extraction.ocr_list || []);
    renderPairs(extraction);
    saveSession();
    fieldsEl.innerHTML = "";

    const found = extraction.found_count || 0;
    const total = extraction.requested_count || lastFields.length;
    fieldsMetaEl.textContent =
      total === 0
        ? `OCR list: ${lastOcrList.length} عنصر`
        : `وُجد ${found} من ${total} · OCR list: ${lastOcrList.length} عنصر`;

    lastFields.forEach((field) => {
      const card = document.createElement("article");
      const state = field.found ? "is-valid" : "is-missing";
      const badge = field.found
        ? `✓ موجود · ${Math.round((field.confidence || 0) * 100)}%`
        : "غير موجود";
      card.className = `field-card ${state}`;
      card.innerHTML = `
        <div class="field-top">
          <p class="field-label">${field.label}</p>
          <span class="field-badge">${badge}</span>
        </div>
        <p class="field-value ${field.value ? "" : "empty"}">${field.value || "—"}</p>
        <p class="field-note">
          name_index = <b>${field.name_index ?? "—"}</b>
          · value_index = <b>${field.value_index ?? "—"}</b>
          ${field.zone ? ` · ${field.zone}` : ""}
          ${field.source_page ? ` · صفحة ${field.source_page}` : ""}
        </p>
      `;
      fieldsEl.appendChild(card);
    });
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
  }

  addFieldBtn.addEventListener("click", addFieldFromInput);
  fieldInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addFieldFromInput();
    }
  });

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
    lastFields = [];
    lastMapped = [];
    lastOcrList = [];
    lastPairs = [];
    pagesEl.innerHTML = "";
    fieldsEl.innerHTML = "";
    if (mappedFieldsEl) mappedFieldsEl.innerHTML = "";
    outputEl.textContent = "";
    if (ocrListEl) ocrListEl.textContent = "";
    if (pairsJsonEl) pairsJsonEl.textContent = "";
    metaEl.textContent = "";
    if (mappedMetaEl) {
      mappedMetaEl.textContent = "جاري تطبيق الـ Mapping المحفوظ…";
    }
    fieldsMetaEl.textContent =
      requestedFields.length > 0
        ? `جاري البحث عن: ${requestedFields.join(" · ")}`
        : "لا توجد حقول محددة — قراءة النص فقط";
    resultsEl.hidden = false;
    setBusy(true);
    setStatus("جاري تجهيز الملف…");

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("fields", JSON.stringify(requestedFields));
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
            if (event.extraction) renderFields(event.extraction);
            if (event.mapped) renderMapped(event.mapped);
            setStatus(
              `جاري قراءة الصفحة ${event.pages_done} من ${event.page_count}…`
            );
            metaEl.textContent = `${event.filename} · ${event.pages_done} / ${event.page_count} · ${event.line_count} سطر`;
          } else if (event.type === "done") {
            if (event.extraction) renderFields(event.extraction);
            if (event.mapped) renderMapped(event.mapped);
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

  copyFieldsBtn.addEventListener("click", async () => {
    const text = fieldsAsText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copyFieldsBtn.textContent = "تم النسخ";
      setTimeout(() => {
        copyFieldsBtn.textContent = "نسخ الحقول";
      }, 1400);
    } catch {
      showError("تعذر النسخ إلى الحافظة.");
    }
  });

  copyListBtn.addEventListener("click", async () => {
    const payload = JSON.stringify(listPayload(), null, 2);
    if (!lastOcrList.length && !lastPairs.length) return;
    try {
      await navigator.clipboard.writeText(payload);
      copyListBtn.textContent = "تم النسخ";
      setTimeout(() => {
        copyListBtn.textContent = "نسخ List JSON";
      }, 1400);
    } catch {
      showError("تعذر النسخ إلى الحافظة.");
    }
  });

  downloadBtn.addEventListener("click", () => {
    const body = [
      fieldsAsText(),
      "",
      "—— pairs JSON ——",
      JSON.stringify(listPayload(), null, 2),
      "",
      "—— النص ——",
      lastText,
    ]
      .filter(Boolean)
      .join("\n");
    if (!body.trim()) return;
    const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = lastName.replace(/\.txt$/i, "") + ".json.txt";
    a.click();
    URL.revokeObjectURL(url);
  });
})();
