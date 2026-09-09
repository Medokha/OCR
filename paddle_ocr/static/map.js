(() => {
  const STORAGE_KEY = "qiraa_ocr_session";

  const emptyEl = document.getElementById("empty");
  const mapperEl = document.getElementById("mapper");
  const statusEl = document.getElementById("map-status");
  const metaEl = document.getElementById("map-meta");
  const rowsEl = document.getElementById("map-rows");
  const jsonEl = document.getElementById("map-json");
  const sideListEl = document.getElementById("ocr-side-list");
  const addBtn = document.getElementById("add-map-btn");
  const seedBtn = document.getElementById("seed-auto-btn");
  const saveBtn = document.getElementById("save-map-btn");
  const copyBtn = document.getElementById("copy-map-btn");
  const clearBtn = document.getElementById("clear-map-btn");

  /** @type {{ index:number, text:string, page?:number }[]} */
  let ocrList = [];
  /** @type {{ name_index:number|null, value_indices:number[] }[]} */
  let rows = [];
  /** @type {{ name:string, value:string, name_index:number|null, value_index:number|null, found?:boolean }[]} */
  let autoPairs = [];

  function loadSession() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  function saveMappingLocal() {
    const session = loadSession() || {};
    session.manual_mapping = buildPayload();
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }

  async function saveMappingToServer() {
    const mapping = buildPayload();
    if (!mapping.length) {
      statusEl.textContent = "أضف صفوف Mapping قبل الحفظ.";
      return;
    }
    const incomplete = mapping.some((m) => m.name_index == null);
    if (incomplete) {
      statusEl.textContent = "اختر name_index لكل صف قبل الحفظ.";
      return;
    }
    saveBtn.disabled = true;
    statusEl.textContent = "جاري حفظ الـ Mapping…";
    try {
      const res = await fetch("/api/mapping", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mapping, source: "map_ui" }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "فشل الحفظ.");
      }
      saveMappingLocal();
      statusEl.textContent = `تم حفظ قالب ثابت · ${data.mapping_count ?? mapping.length} صف (اسم الحقل + إزاحة القيم) · هيشتغل على أي OCR لاحق.`;
      saveBtn.textContent = "تم الحفظ ✓";
      setTimeout(() => {
        saveBtn.textContent = "حفظ الـ Mapping";
      }, 1600);
    } catch (err) {
      statusEl.textContent = err.message || "تعذر حفظ الـ Mapping.";
    } finally {
      saveBtn.disabled = false;
    }
  }

  async function loadSavedMappingFromServer() {
    try {
      const res = await fetch("/api/mapping");
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }

  function normalizeText(text) {
    return String(text || "")
      .replace(/[\u064B-\u065F\u0670]/g, "")
      .replace(/[أإآٱ]/g, "ا")
      .replace(/ة/g, "ه")
      .replace(/ى/g, "ي")
      .replace(/\s+/g, " ")
      .trim()
      .toLocaleLowerCase("ar");
  }

  function findNameIndexByText(name) {
    if (!name) return null;
    const target = normalizeText(name);
    let best = null;
    let bestScore = 0;
    ocrList.forEach((item) => {
      const cand = normalizeText(item.text);
      let score = 0;
      if (cand === target) score = 100;
      else if (cand.startsWith(target) || target.startsWith(cand)) score = 80;
      else if (cand.includes(target) || target.includes(cand)) score = 60;
      if (score > bestScore) {
        bestScore = score;
        best = item.index;
      }
    });
    return bestScore >= 50 ? best : null;
  }

  /** Project fixed template (name + offsets) onto current OCR list. */
  function projectFixedRow(m) {
    const nameIndex =
      findNameIndexByText(m.name) ??
      (m.name_index != null ? Number(m.name_index) : null);
    let offsets = Array.isArray(m.value_offsets) ? [...m.value_offsets] : null;
    if ((!offsets || !offsets.length) && m.name_index != null && m.value_indices) {
      offsets = m.value_indices.map((vi) => Number(vi) - Number(m.name_index));
    }
    offsets = offsets || [];
    const valueIndices =
      nameIndex == null
        ? []
        : offsets
            .map((off) => nameIndex + Number(off))
            .filter((vi) => ocrList.some((x) => x.index === vi));
    return {
      name_index: nameIndex,
      value_indices: valueIndices,
      name: m.name || null,
    };
  }

  function itemLabel(idx) {
    const item = ocrList.find((x) => x.index === idx);
    if (!item) return `[${idx}] (غير موجود)`;
    return `[${idx}] ${item.text}`;
  }

  function buildPayload() {
    return rows.map((row) => {
      const nameItem = ocrList.find((x) => x.index === row.name_index);
      const nameIndex = row.name_index;
      const valueIndices = [...(row.value_indices || [])];
      const valueOffsets =
        nameIndex == null
          ? []
          : valueIndices.map((vi) => vi - nameIndex);
      const values = valueIndices
        .map((vi) => {
          const it = ocrList.find((x) => x.index === vi);
          return it
            ? {
                value_index: vi,
                offset: nameIndex == null ? null : vi - nameIndex,
                value: it.text,
                page: it.page,
              }
            : { value_index: vi, offset: null, value: null };
        })
        .filter((v) => v.value_index != null);

      return {
        // Fixed template used on ANY later OCR:
        name: nameItem ? nameItem.text : row.name || null,
        value_offsets: valueOffsets,
        // Teaching snapshot (reference only):
        name_index: nameIndex,
        page: nameItem ? nameItem.page : null,
        value_indices: valueIndices,
        values: values.map((v) => v.value),
        value_items: values,
        has_values: values.length > 0,
      };
    });
  }

  function refreshJson() {
    const payload = {
      ocr_list: ocrList,
      mapping: buildPayload(),
    };
    jsonEl.textContent = JSON.stringify(payload, null, 2);
    metaEl.textContent = `${rows.length} صف · ${ocrList.length} عنصر في OCR list`;
    saveMappingLocal();
  }

  function renderSideList() {
    sideListEl.innerHTML = "";
    ocrList.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ocr-item";
      btn.innerHTML = `<span class="ocr-idx">${item.index}</span><span class="ocr-txt">${item.text}</span>`;
      btn.title = "اضغط لنسخ الرقم";
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(String(item.index));
          btn.classList.add("copied");
          setTimeout(() => btn.classList.remove("copied"), 700);
        } catch {
          /* ignore */
        }
      });
      sideListEl.appendChild(btn);
    });
  }

  function nameOptions(selected) {
    return ocrList
      .map((item) => {
        const sel = item.index === selected ? "selected" : "";
        return `<option value="${item.index}" ${sel}>[${item.index}] ${escapeHtml(item.text)}</option>`;
      })
      .join("");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderRows() {
    rowsEl.innerHTML = "";
    if (!rows.length) {
      rowsEl.innerHTML =
        '<p class="meta">لا توجد صفوف بعد. اضغط «إضافة صف Mapping» أو «تعبئة من الاستخراج التلقائي».</p>';
      refreshJson();
      return;
    }

    rows.forEach((row, rowIdx) => {
      const card = document.createElement("article");
      card.className = "map-row";

      const valuesSet = new Set(row.value_indices || []);
      const checks = ocrList
        .map((item) => {
          // Don't allow selecting the name index as a value by default display,
          // but allow if user wants (same block) — still show all.
          const checked = valuesSet.has(item.index) ? "checked" : "";
          return `<label class="val-check"><input type="checkbox" data-row="${rowIdx}" data-vi="${item.index}" ${checked}/><span>[${item.index}] ${escapeHtml(item.text)}</span></label>`;
        })
        .join("");

      card.innerHTML = `
        <div class="map-row-top">
          <div>
            <label class="map-label">اسم الحقل (name_index)</label>
            <select class="map-select" data-row="${rowIdx}" data-role="name">
              <option value="">— اختر من OCR list —</option>
              ${nameOptions(row.name_index)}
            </select>
          </div>
          <button type="button" class="btn-ghost map-del" data-del="${rowIdx}">حذف الصف</button>
        </div>
        <div class="map-values">
          <label class="map-label">القيم (value_indices) — يمكن اختيار صفر أو أكثر</label>
          <div class="val-grid">${checks}</div>
          <p class="meta">المختار: ${(row.value_indices || []).length} قيمة${
            (row.value_indices || []).length && row.name_index != null
              ? " · offsets: [" +
                (row.value_indices || [])
                  .map((i) => i - row.name_index)
                  .join(", ") +
                "] → " +
                (row.value_indices || []).map((i) => itemLabel(i)).join(" · ")
              : (row.value_indices || []).length
                ? " → " + (row.value_indices || []).map((i) => itemLabel(i)).join(" · ")
                : " (بدون قيمة)"
          }</p>
        </div>
      `;
      rowsEl.appendChild(card);
    });

    rowsEl.querySelectorAll("select[data-role='name']").forEach((sel) => {
      sel.addEventListener("change", (e) => {
        const idx = Number(e.target.getAttribute("data-row"));
        const val = e.target.value;
        rows[idx].name_index = val === "" ? null : Number(val);
        refreshJson();
        renderRows();
      });
    });

    rowsEl.querySelectorAll("input[type='checkbox'][data-vi]").forEach((cb) => {
      cb.addEventListener("change", (e) => {
        const idx = Number(e.target.getAttribute("data-row"));
        const vi = Number(e.target.getAttribute("data-vi"));
        const set = new Set(rows[idx].value_indices || []);
        if (e.target.checked) set.add(vi);
        else set.delete(vi);
        rows[idx].value_indices = [...set].sort((a, b) => a - b);
        refreshJson();
        // Update summary text without full re-render to keep scroll — full re-render is ok for clarity
        renderRows();
      });
    });

    rowsEl.querySelectorAll("[data-del]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.getAttribute("data-del"));
        rows.splice(idx, 1);
        renderRows();
      });
    });

    refreshJson();
  }

  function seedFromAuto() {
    if (!autoPairs.length) {
      statusEl.textContent = "لا يوجد استخراج تلقائي محفوظ للتعبئة.";
      return;
    }
    const byName = new Map();
    autoPairs.forEach((p) => {
      if (p.name_index == null) return;
      if (!byName.has(p.name_index)) {
        byName.set(p.name_index, {
          name_index: p.name_index,
          value_indices: [],
        });
      }
      const row = byName.get(p.name_index);
      if (p.value_index != null && !row.value_indices.includes(p.value_index)) {
        row.value_indices.push(p.value_index);
      }
    });
    rows = [...byName.values()];
    if (!rows.length) {
      statusEl.textContent = "الاستخراج التلقائي لا يحتوي name_index صالح.";
      return;
    }
    statusEl.textContent = `تم تعبئة ${rows.length} صف من الاستخراج التلقائي. عدّل كما تشاء.`;
    renderRows();
  }

  addBtn.addEventListener("click", () => {
    rows.push({ name_index: null, value_indices: [] });
    renderRows();
  });

  seedBtn.addEventListener("click", seedFromAuto);

  if (saveBtn) {
    saveBtn.addEventListener("click", () => {
      saveMappingToServer();
    });
  }

  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(
        JSON.stringify({ ocr_list: ocrList, mapping: buildPayload() }, null, 2)
      );
      copyBtn.textContent = "تم النسخ";
      setTimeout(() => {
        copyBtn.textContent = "نسخ JSON";
      }, 1200);
    } catch {
      statusEl.textContent = "تعذر النسخ.";
    }
  });

  clearBtn.addEventListener("click", () => {
    rows = [];
    renderRows();
    statusEl.textContent = "تم مسح صفوف الـ mapping.";
  });

  // init
  (async () => {
    const session = loadSession();
    const saved = await loadSavedMappingFromServer();

    if (!session || !Array.isArray(session.ocr_list) || !session.ocr_list.length) {
      emptyEl.hidden = false;
      mapperEl.hidden = true;
      if (saved && (saved.mapping || []).length) {
        statusEl.textContent = `لا توجد بيانات OCR في الجلسة · يوجد Mapping محفوظ (${saved.mapping.length} صف) على السيرفر.`;
      } else {
        statusEl.textContent = "لا توجد بيانات OCR في الجلسة.";
      }
      return;
    }

    emptyEl.hidden = true;
    mapperEl.hidden = false;
    ocrList = session.ocr_list;
    autoPairs = session.pairs || [];

    // Prefer fixed server template projected onto THIS OCR list.
    if (saved && Array.isArray(saved.mapping) && saved.mapping.length) {
      rows = saved.mapping.map(projectFixedRow);
      statusEl.textContent = `قالب ثابت محمّل (${rows.length} صف) ومُسقَط على OCR الحالي (${ocrList.length} عنصر). احفظ بعد التعديل إن لزم.`;
    } else if (Array.isArray(session.manual_mapping) && session.manual_mapping.length) {
      rows = session.manual_mapping.map(projectFixedRow);
      statusEl.textContent = `تم تحميل ${ocrList.length} عنصر OCR و ${rows.length} صف من الجلسة.`;
    } else {
      statusEl.textContent = `تم تحميل ${ocrList.length} عنصر من آخر قراءة. أضف صفوف أو عبّئ تلقائياً ثم احفظ كقالب ثابت.`;
    }
    renderSideList();
    renderRows();
  })();
})();
