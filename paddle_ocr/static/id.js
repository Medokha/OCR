(() => {
  const input = document.getElementById("id-input");
  const runBtn = document.getElementById("id-run");
  const fileLabel = document.getElementById("id-file-label");
  const handwritingOpt = document.getElementById("id-handwriting");
  const statusEl = document.getElementById("id-status");
  const statusText = document.getElementById("id-status-text");
  const errorEl = document.getElementById("id-error");
  const resultsEl = document.getElementById("id-results");
  const fieldsEl = document.getElementById("id-fields");
  const metaEl = document.getElementById("id-meta");
  const rawEl = document.getElementById("id-raw");
  const faceImg = document.getElementById("face-img");
  const faceEmpty = document.getElementById("face-empty");
  const cardImg = document.getElementById("card-img");
  const cardEmpty = document.getElementById("card-empty");
  const annotatedImg = document.getElementById("annotated-img");
  const annotatedEmpty = document.getElementById("annotated-empty");
  const cropsEl = document.getElementById("id-crops");
  const copyBtn = document.getElementById("id-copy");
  const rulesEl = document.getElementById("id-rules");

  let selectedFile = null;
  let lastPayload = null;

  const FIELD_LABELS = [
    ["full_name", "الاسم بالكامل"],
    ["national_id", "الرقم القومي"],
    ["birth_date", "تاريخ الميلاد"],
    ["gender", "النوع"],
    ["governorate", "محافظة الميلاد"],
    ["address", "العنوان / محل الإقامة"],
    ["job", "المهنة"],
    ["religion", "الديانة"],
    ["marital_status", "الحالة الاجتماعية"],
    ["husband_name", "اسم الزوج"],
    ["expiry_date", "سريان البطاقة حتى"],
    ["card_side", "وجه البطاقة"],
  ];

  const CROP_LABELS = {
    face: "الصورة الشخصية",
    full_name: "منطقة الاسم",
    address: "منطقة العنوان / محل الإقامة",
    national_id: "منطقة الرقم القومي",
    job: "منطقة المهنة (الظهر)",
    religion: "منطقة الديانة (الظهر)",
    marital_status: "منطقة الحالة الاجتماعية (الظهر)",
    gender: "منطقة الجنس",
    husband_name: "منطقة اسم الزوج (الظهر)",
    expiry_date: "منطقة تاريخ السريان (الظهر)",
  };

  const ALLOWED = new Set([
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
  ]);

  function extOf(name) {
    const i = String(name || "").lastIndexOf(".");
    return i >= 0 ? String(name).slice(i).toLowerCase() : "";
  }

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

  function setStatus(msg) {
    statusEl.hidden = false;
    statusText.textContent = msg;
  }

  function setImage(imgEl, emptyEl, src) {
    if (src) {
      imgEl.src = src;
      imgEl.hidden = false;
      emptyEl.hidden = true;
    } else {
      imgEl.removeAttribute("src");
      imgEl.hidden = true;
      emptyEl.hidden = false;
    }
  }

  function renderRules(rules, decoded) {
    if (!rulesEl) return;
    rulesEl.innerHTML = "";
    const items = [...(rules || [])];
    if (decoded && decoded.valid) {
      items.unshift(
        `معادلة الرقم: ميلاد ${decoded.birth_date || "—"} · ${decoded.gender || "—"} · ${decoded.governorate || "—"}`
      );
    }
    if (!items.length) {
      rulesEl.innerHTML = "<li>لم تُطبَّق قواعد كافية — حسّن وضوح الصورة.</li>";
      return;
    }
    items.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      rulesEl.appendChild(li);
    });
  }

  function renderCrops(images, fields) {
    if (!cropsEl) return;
    cropsEl.innerHTML = "";
    const crops = (images && images.crops) || {};
    const order = [
      "full_name",
      "address",
      "national_id",
      "job",
      "religion",
      "marital_status",
      "gender",
      "husband_name",
      "expiry_date",
    ];

    if (images && images.face) {
      const card = document.createElement("article");
      card.className = "id-crop-card";
      card.innerHTML = `
        <img src="${images.face}" alt="face" />
        <p class="crop-cap">${CROP_LABELS.face}</p>
        <p class="crop-val">مقصوصة من البطاقة</p>
      `;
      cropsEl.appendChild(card);
    }

    order.forEach((key) => {
      const src = crops[key];
      if (!src) return;
      const card = document.createElement("article");
      card.className = "id-crop-card";
      const val = (fields && fields[key]) || "—";
      card.innerHTML = `
        <img src="${src}" alt="${key}" />
        <p class="crop-cap">${CROP_LABELS[key] || key}</p>
        <p class="crop-val">${val}</p>
      `;
      cropsEl.appendChild(card);
    });

    if (!cropsEl.children.length) {
      cropsEl.innerHTML =
        '<p class="meta">لا توجد قصّات حقول بعد — تأكد من وضوح النصوص على البطاقة.</p>';
    }
  }

  function renderFields(fields, crops) {
    fieldsEl.innerHTML = "";
    FIELD_LABELS.forEach(([key, label], idx) => {
      const value = fields && fields[key];
      const cropSrc = crops && crops[key];
      const card = document.createElement("article");
      card.className = `id-field ${value ? "" : "is-empty"} ${cropSrc ? "has-crop" : ""}`;
      card.style.animationDelay = `${idx * 40}ms`;
      const valClass = key === "national_id" ? "id-field-value nid" : "id-field-value";
      card.innerHTML = `
        <p class="id-field-label">${label}</p>
        <p class="${valClass}">${value || "—"}</p>
        ${cropSrc ? `<img class="id-field-crop" src="${cropSrc}" alt="${key}-crop" />` : ""}
      `;
      fieldsEl.appendChild(card);
    });
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
    if (!ALLOWED.has(extOf(file.name))) {
      selectedFile = null;
      runBtn.disabled = true;
      showError("يُقبل صورة فقط (PNG / JPG / WEBP / BMP / TIFF).");
      return;
    }
    selectedFile = file;
    runBtn.disabled = false;
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    fileLabel.textContent = `${file.name} · ${mb} ميجابايت`;
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    clearError();
    resultsEl.hidden = true;
    setBusy(true);
    setStatus("جاري قص البطاقة وقراءة النصوص والصورة الشخصية…");

    const form = new FormData();
    form.append("file", selectedFile);
    form.append(
      "enhance_handwriting",
      handwritingOpt && handwritingOpt.checked ? "1" : "0"
    );

    try {
      const res = await fetch("/api/id/ocr", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        throw new Error(
          Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join(" · ")
            : detail || "فشل استخراج بيانات البطاقة."
        );
      }

      lastPayload = data;
      const images = data.images || {};
      setImage(faceImg, faceEmpty, images.face || null);
      setImage(cardImg, cardEmpty, images.card || images.original || null);
      setImage(
        annotatedImg,
        annotatedEmpty,
        images.annotated || images.card || null
      );
      renderFields(data.fields || {}, images.crops || {});
      renderCrops(images, data.fields || {});
      renderRules(data.rules || [], data.decoded || {});
      metaEl.textContent = data.message || "";
      rawEl.textContent = data.raw_text || "(فارغ)";
      resultsEl.hidden = false;
      setStatus("اكتمل الاستخراج");
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      if (/failed to fetch|networkerror|load failed/i.test(msg)) {
        showError("انقطع الاتصال بالسيرفر. تأكد أنه شغال ثم أعد المحاولة.");
      } else {
        showError(msg || "حدث خطأ غير متوقع.");
      }
    } finally {
      setBusy(false);
    }
  });

  copyBtn.addEventListener("click", async () => {
    if (!lastPayload) return;
    const slim = {
      fields: lastPayload.fields,
      decoded: lastPayload.decoded,
      side: lastPayload.side,
      filename: lastPayload.filename,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(slim, null, 2));
      copyBtn.textContent = "تم النسخ";
      setTimeout(() => {
        copyBtn.textContent = "نسخ JSON";
      }, 1200);
    } catch {
      showError("تعذر النسخ.");
    }
  });
})();
