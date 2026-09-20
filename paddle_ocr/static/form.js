(() => {
  const input = document.getElementById("form-input");
  const runBtn = document.getElementById("form-run");
  const fileLabel = document.getElementById("form-file-label");
  const statusEl = document.getElementById("form-status");
  const statusText = document.getElementById("form-status-text");
  const errorEl = document.getElementById("form-error");
  const resultsEl = document.getElementById("form-results");
  const metaEl = document.getElementById("form-meta");
  const titleBadge = document.getElementById("form-title-badge");
  const fieldsEl = document.getElementById("form-fields");
  const cropsEl = document.getElementById("form-crops");
  const rulesList = document.getElementById("form-rules-list");
  const annotatedImg = document.getElementById("form-annotated");
  const annotatedEmpty = document.getElementById("form-annotated-empty");
  const originalImg = document.getElementById("form-original");
  const originalEmpty = document.getElementById("form-original-empty");

  let selectedFile = null;

  function showError(msg) {
    errorEl.hidden = false;
    errorEl.textContent = msg;
  }

  function clearError() {
    errorEl.hidden = true;
    errorEl.textContent = "";
  }

  function setBusy(busy) {
    runBtn.disabled = busy || !selectedFile;
    input.disabled = busy;
    if (!busy) statusEl.hidden = true;
  }

  function setImg(img, empty, b64) {
    if (b64) {
      img.src = `data:image/jpeg;base64,${b64}`;
      img.hidden = false;
      empty.hidden = true;
    } else {
      img.removeAttribute("src");
      img.hidden = true;
      empty.hidden = false;
    }
  }

  function renderFields(fields) {
    fieldsEl.innerHTML = "";
    (fields || []).forEach((f, idx) => {
      const card = document.createElement("article");
      card.className = `field-card form-zone-card ${f.found ? "is-found" : "is-empty"}`;
      const val = f.found && f.value ? f.value : "— فارغ / مش متكتب";
      const ar = f.label_ar || "";
      const en = f.label_en || "";
      const title = f.label || (ar && en ? `${ar} / ${en}` : ar || en || f.key);
      card.innerHTML = `
        <div class="field-label">${idx + 1}. ${title}</div>
        <div class="zone-tag">${ar ? `عربي: ${ar}` : ""}${ar && en ? " · " : ""}${en ? `EN: ${en}` : ""}</div>
        <div class="field-value">${val}</div>
        <div class="zone-tag">${f.section || ""} · ${f.strategy || ""} · ${f.message || ""}</div>
      `;
      fieldsEl.appendChild(card);
    });
  }

  function renderCrops(fields, crops) {
    cropsEl.innerHTML = "";
    (fields || []).forEach((f) => {
      const b64 = (crops && crops[f.key]) || f.crop_b64;
      if (!b64) return;
      const wrap = document.createElement("article");
      wrap.className = "id-crop-card";
      const title = f.label || f.label_ar || f.key;
      wrap.innerHTML = `
        <h4>${title}</h4>
        <img alt="${title}" src="data:image/jpeg;base64,${b64}" />
        <p class="meta">${f.label_en || ""}</p>
        <p class="meta">${f.found && f.value ? f.value : "منطقة فارغة"}</p>
      `;
      cropsEl.appendChild(wrap);
    });
  }

  input.addEventListener("change", () => {
    selectedFile = input.files && input.files[0] ? input.files[0] : null;
    runBtn.disabled = !selectedFile;
    fileLabel.textContent = selectedFile
      ? `الملف: ${selectedFile.name}`
      : "الحد الأقصى صورة فورم واضح · مثال KYC البنك الزراعي";
    clearError();
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    clearError();
    resultsEl.hidden = true;
    setBusy(true);
    statusEl.hidden = false;
    statusText.textContent = "جاري فهم الفورم وتحديد مناطق الإجابات…";

    const fd = new FormData();
    fd.append("file", selectedFile);

    try {
      const res = await fetch("/api/form/zones", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        throw new Error((data && data.detail) || "فشل تحليل الفورم");
      }

      metaEl.textContent = data.message || "";
      titleBadge.hidden = false;
      titleBadge.textContent = data.form_title || data.form_type || "فورم";

      const imgs = data.images || {};
      setImg(annotatedImg, annotatedEmpty, imgs.annotated);
      setImg(originalImg, originalEmpty, imgs.original);
      renderFields(data.fields || []);
      renderCrops(data.fields || [], imgs.crops || {});

      rulesList.innerHTML = "";
      (data.rules || []).forEach((r) => {
        const li = document.createElement("li");
        li.textContent = r;
        rulesList.appendChild(li);
      });

      resultsEl.hidden = false;
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  });
})();
