(() => {
  const input = document.getElementById("id-input");
  const runBtn = document.getElementById("id-run");
  const fileLabel = document.getElementById("id-file-label");
  const handwritingOpt = document.getElementById("id-handwriting");
  const statusEl = document.getElementById("id-status");
  const statusText = document.getElementById("id-status-text");
  const errorEl = document.getElementById("id-error");
  const resultsEl = document.getElementById("id-results");
  const fieldsFrontEl = document.getElementById("id-fields-front");
  const fieldsBackEl = document.getElementById("id-fields-back");
  const formFront = document.getElementById("form-front");
  const formBack = document.getElementById("form-back");
  const sideBadge = document.getElementById("id-side-badge");
  const metaEl = document.getElementById("id-meta");
  const rawEl = document.getElementById("id-raw");
  const faceImg = document.getElementById("face-img");
  const faceEmpty = document.getElementById("face-empty");
  const cardImg = document.getElementById("card-img");
  const cardEmpty = document.getElementById("card-empty");
  const annotatedImg = document.getElementById("annotated-img");
  const annotatedEmpty = document.getElementById("annotated-empty");
  const cropsFrontEl = document.getElementById("id-crops-front");
  const cropsBackEl = document.getElementById("id-crops-back");
  const copyBtn = document.getElementById("id-copy");
  const rulesEl = document.getElementById("id-rules");

  let selectedFile = null;
  let lastPayload = null;

  // Separate forms — no shared field list between sides
  const FRONT_FIELDS = [
    ["full_name", "الاسم بالكامل"],
    ["national_id", "الرقم القومي"],
    ["birth_date", "تاريخ الميلاد"],
    ["gender", "النوع (من الرقم القومي)"],
    ["governorate", "محافظة الميلاد"],
    ["address", "العنوان / محل الإقامة"],
  ];

  const BACK_FIELDS = [
    ["job", "المهنة"],
    ["gender", "النوع"],
    ["religion", "الديانة"],
    ["marital_status", "الحالة الاجتماعية"],
    ["husband_name", "اسم الزوج"],
    ["expiry_date", "سريان البطاقة حتى"],
  ];

  const FRONT_CROPS = ["full_name", "address", "national_id"];
  const BACK_CROPS = [
    "job",
    "religion",
    "marital_status",
    "gender",
    "husband_name",
    "expiry_date",
  ];

  const CROP_LABELS = {
    face: "الصورة الشخصية",
    full_name: "منطقة الاسم",
    address: "منطقة العنوان / محل الإقامة",
    national_id: "منطقة الرقم القومي",
    job: "منطقة المهنة",
    religion: "منطقة الديانة",
    marital_status: "منطقة الحالة الاجتماعية",
    gender: "منطقة الجنس",
    husband_name: "منطقة اسم الزوج",
    expiry_date: "منطقة تاريخ السريان",
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

  function detectSide(data) {
    const side = String(data.side || "").toLowerCase();
    if (side === "front" || side === "back") return side;
    const fields = data.fields || {};
    const cardSide = String(fields.card_side || "");
    if (cardSide.includes("ظهر")) return "back";
    if (cardSide.includes("وجه")) return "front";
    // Heuristic: back-only fields filled without name/address
    if (fields.job || fields.expiry_date || fields.marital_status) {
      if (!fields.full_name && !fields.address) return "back";
    }
    return "front";
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

  function renderFieldGrid(container, fieldDefs, fields, crops) {
    container.innerHTML = "";
    fieldDefs.forEach(([key, label], idx) => {
      const value = fields && fields[key];
      const cropSrc = crops && crops[key];
      const card = document.createElement("article");
      card.className = `id-field ${value ? "" : "is-empty"} ${cropSrc ? "has-crop" : ""}`;
      card.style.animationDelay = `${idx * 40}ms`;
      const valClass =
        key === "national_id" ? "id-field-value nid" : "id-field-value";
      card.innerHTML = `
        <p class="id-field-label">${label}</p>
        <p class="${valClass}">${value || "—"}</p>
        ${
          cropSrc
            ? `<img class="id-field-crop" src="${cropSrc}" alt="${key}-crop" />`
            : ""
        }
      `;
      container.appendChild(card);
    });
  }

  function renderCropGrid(container, order, images, fields, { withFace = false } = {}) {
    if (!container) return;
    container.innerHTML = "";
    const crops = (images && images.crops) || {};

    if (withFace && images && images.face) {
      const card = document.createElement("article");
      card.className = "id-crop-card";
      card.innerHTML = `
        <img src="${images.face}" alt="face" />
        <p class="crop-cap">${CROP_LABELS.face}</p>
        <p class="crop-val">مقصوصة من الوجه</p>
      `;
      container.appendChild(card);
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
      container.appendChild(card);
    });

    if (!container.children.length) {
      container.innerHTML =
        '<p class="meta">لا توجد قصّات لهذه الجهة بعد.</p>';
    }
  }

  function renderBySide(data) {
    const side = detectSide(data);
    const fields = data.fields || {};
    const images = data.images || {};
    const crops = images.crops || {};

    // Only show the form matching the detected side — prevents front/back bleed
    formFront.hidden = side !== "front";
    formBack.hidden = side !== "back";
    formFront.classList.toggle("is-active", side === "front");
    formBack.classList.toggle("is-active", side === "back");

    if (sideBadge) {
      sideBadge.hidden = false;
      sideBadge.className =
        side === "back" ? "id-side-badge is-back" : "id-side-badge is-front";
      sideBadge.textContent =
        side === "back" ? "تم التعرف: ظهر البطاقة" : "تم التعرف: وجه البطاقة";
    }

    // Face photo only relevant on front
    const faceCard = document.querySelector(".face-card");
    if (faceCard) faceCard.hidden = side === "back";

    if (side === "front") {
      renderFieldGrid(fieldsFrontEl, FRONT_FIELDS, fields, crops);
      renderCropGrid(cropsFrontEl, FRONT_CROPS, images, fields, {
        withFace: true,
      });
      fieldsBackEl.innerHTML = "";
      cropsBackEl.innerHTML = "";
    } else {
      renderFieldGrid(fieldsBackEl, BACK_FIELDS, fields, crops);
      renderCropGrid(cropsBackEl, BACK_CROPS, images, fields, {
        withFace: false,
      });
      fieldsFrontEl.innerHTML = "";
      cropsFrontEl.innerHTML = "";
    }

    return side;
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
      const side = renderBySide(data);
      setImage(
        faceImg,
        faceEmpty,
        side === "front" ? images.face || null : null
      );
      setImage(cardImg, cardEmpty, images.card || images.original || null);
      setImage(
        annotatedImg,
        annotatedEmpty,
        images.annotated || images.card || null
      );
      renderRules(data.rules || [], data.decoded || {});
      metaEl.textContent = data.message || "";
      rawEl.textContent = data.raw_text || "(فارغ)";
      resultsEl.hidden = false;
      setStatus(
        side === "back"
          ? "اكتمل استخراج الظهر"
          : "اكتمل استخراج الوجه"
      );
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
    const side = detectSide(lastPayload);
    const fields = lastPayload.fields || {};
    const keys =
      side === "back"
        ? BACK_FIELDS.map(([k]) => k)
        : FRONT_FIELDS.map(([k]) => k);
    const slimFields = {};
    keys.forEach((k) => {
      if (fields[k] != null && fields[k] !== "") slimFields[k] = fields[k];
    });
    const slim = {
      side,
      fields: slimFields,
      decoded: side === "front" ? lastPayload.decoded : undefined,
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
