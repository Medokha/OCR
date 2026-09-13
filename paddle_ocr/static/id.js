(() => {
  const input = document.getElementById("id-input");
  const runFrontBtn = document.getElementById("id-run-front");
  const runBackBtn = document.getElementById("id-run-back");
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
  let lastSide = "front";

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
    const noFile = !selectedFile;
    runFrontBtn.disabled = busy || noFile;
    runBackBtn.disabled = busy || noFile;
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

  function renderBySide(data, side) {
    const fields = data.fields || {};
    const images = data.images || {};
    const crops = images.crops || {};
    lastSide = side;

    formFront.hidden = side !== "front";
    formBack.hidden = side !== "back";
    formFront.classList.toggle("is-active", side === "front");
    formBack.classList.toggle("is-active", side === "back");

    if (sideBadge) {
      sideBadge.hidden = false;
      sideBadge.className =
        side === "back" ? "id-side-badge is-back" : "id-side-badge is-front";
      sideBadge.textContent =
        side === "back" ? "نموذج الظهر" : "نموذج الوجه";
    }

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
  }

  async function runOcr(side) {
    if (!selectedFile) return;
    clearError();
    resultsEl.hidden = true;
    setBusy(true);
    setStatus(
      side === "back"
        ? "جاري قراءة ظهر البطاقة…"
        : "جاري قراءة وجه البطاقة…"
    );

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("side", side);
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
      const resolvedSide =
        data.side === "back" || data.side === "front" ? data.side : side;
      renderBySide(data, resolvedSide);
      setImage(
        faceImg,
        faceEmpty,
        resolvedSide === "front" ? images.face || null : null
      );
      setImage(cardImg, cardEmpty, images.card || images.original || null);
      setImage(
        annotatedImg,
        annotatedEmpty,
        images.annotated || images.card || null
      );
      renderRules(data.rules || [], data.decoded || {});
      const pipe = data.pipeline || "?";
      const cropM = data.crop_method || "?";
      const doneMsg =
        resolvedSide === "back" ? "اكتمل استخراج الظهر" : "اكتمل استخراج الوجه";
      setStatus(
        `${doneMsg} · ${data.message || ""} · محرك ${pipe} · قص ${cropM}`
      );
      metaEl.textContent = data.message || "";
      rawEl.textContent = data.raw_text || "(فارغ)";
      resultsEl.hidden = false;
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
  }

  input.addEventListener("change", () => {
    clearError();
    resultsEl.hidden = true;
    const file = input.files && input.files[0];
    if (!file) {
      selectedFile = null;
      setBusy(false);
      fileLabel.textContent = "لم يتم اختيار ملف بعد";
      return;
    }
    if (!ALLOWED.has(extOf(file.name))) {
      selectedFile = null;
      setBusy(false);
      runFrontBtn.disabled = true;
      runBackBtn.disabled = true;
      showError("يُقبل صورة فقط (PNG / JPG / WEBP / BMP / TIFF).");
      return;
    }
    selectedFile = file;
    runFrontBtn.disabled = false;
    runBackBtn.disabled = false;
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    fileLabel.textContent = `${file.name} · ${mb} ميجابايت · اختر: قراءة الوجه أو قراءة الظهر`;
  });

  runFrontBtn.addEventListener("click", () => runOcr("front"));
  runBackBtn.addEventListener("click", () => runOcr("back"));

  copyBtn.addEventListener("click", async () => {
    if (!lastPayload) return;
    const side = lastSide;
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
