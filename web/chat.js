/*
 * Brightnest chat: one script for the customer website widget and the internal chat page.
 *
 * Website widget (a chat bubble in the corner), one line on any page:
 *     <script src="https://assistant.example.com/web/chat.js" data-widget></script>
 *   Optional: data-api="https://assistant.example.com" (default: where this script came from),
 *             data-lang="id" (interface language; default: the browser's language).
 *   The site must be listed in the server's WIDGET_ORIGINS setting if it is on another domain.
 *
 * Full page (used by internal.html):
 *     BrightnestChat.mount(element, { assistant: "internal", getToken: () => token, onSignedOut: fn })
 *
 * Safety: every answer is put on the page with textContent, never innerHTML. The small formatter below
 * builds elements for **bold**, bullet lists and [1] citations, so the model's text can never become HTML.
 * The widget lives in a Shadow DOM, so the host site's CSS cannot break it and it cannot break the site.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var MAX_HISTORY = 6;

  var TEXT = {
    en: {
      title: { external: "Brightnest Customer Help", internal: "Brightnest Employee Helper" },
      hello: {
        external: "Hi! Ask me about products, orders, store hours, returns, shipping or warranty.",
        internal: "Hi! Ask me about leave, holidays, remote work, reimbursements, benefits or IT help."
      },
      placeholder: "Type your question...",
      send: "Send",
      thinking: "Looking it up...",
      sources: "Sources",
      updated: "updated",
      tool: "live data",
      contact: "Need a person?",
      helpful: "Was this helpful?",
      thanks: "Thanks for the feedback!",
      open: "Chat with us",
      close: "Close chat",
      errors: {
        429: "You're sending messages quickly. Please wait a moment and try again.",
        503: "The assistant is busy right now. Please try again in a minute.",
        401: "Your sign-in has expired. Please sign in again.",
        422: "That message is too long. Please shorten it.",
        other: "Something went wrong. Please try again."
      }
    },
    id: {
      title: { external: "Bantuan Pelanggan Brightnest", internal: "Asisten Karyawan Brightnest" },
      hello: {
        external: "Halo! Tanyakan tentang produk, pesanan, jam buka toko, pengembalian, pengiriman, atau garansi.",
        internal: "Halo! Tanyakan tentang cuti, hari libur, kerja jarak jauh, reimbursement, tunjangan, atau bantuan IT."
      },
      placeholder: "Tulis pertanyaan Anda...",
      send: "Kirim",
      thinking: "Sedang mencari...",
      sources: "Sumber",
      updated: "diperbarui",
      tool: "data langsung",
      contact: "Perlu bantuan petugas?",
      helpful: "Apakah ini membantu?",
      thanks: "Terima kasih atas masukannya!",
      open: "Chat dengan kami",
      close: "Tutup chat",
      errors: {
        429: "Pesan Anda terlalu cepat. Mohon tunggu sebentar lalu coba lagi.",
        503: "Asisten sedang sibuk. Silakan coba lagi dalam satu menit.",
        401: "Sesi Anda telah berakhir. Silakan masuk kembali.",
        422: "Pesan terlalu panjang. Mohon dipersingkat.",
        other: "Terjadi kesalahan. Silakan coba lagi."
      }
    }
  };

  var CSS = [
    ":host{all:initial;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#1f2a24;",
    "  --brand:#2f6b4f;--brand-dark:#24543e;--soft:#eef4f0;--line:#dfe6e1;--muted:#5d6b63}",
    "*{box-sizing:border-box}",
    ".chat{display:flex;flex-direction:column;height:100%;background:#fff;border:1px solid var(--line);",
    "  border-radius:14px;overflow:hidden}",
    ".head{background:var(--brand);color:#fff;padding:12px 16px;font-weight:600;font-size:15px;",
    "  display:flex;align-items:center;justify-content:space-between;gap:8px}",
    ".head button{background:none;border:0;color:#fff;font-size:20px;cursor:pointer;line-height:1}",
    ".log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;background:#fafcfb}",
    ".msg{max-width:88%;padding:10px 12px;border-radius:12px;font-size:14px;line-height:1.45;",
    "  white-space:normal;overflow-wrap:anywhere}",
    ".user{align-self:flex-end;background:var(--brand);color:#fff;border-bottom-right-radius:4px}",
    ".bot{align-self:flex-start;background:#fff;border:1px solid var(--line);border-bottom-left-radius:4px}",
    ".bot p{margin:0 0 6px}.bot p:last-child{margin-bottom:0}.bot ul{margin:4px 0 6px;padding-left:20px}",
    ".cite{display:inline-block;border:0;background:var(--soft);color:var(--brand-dark);font-size:11px;",
    "  font-weight:600;border-radius:6px;padding:1px 5px;margin:0 1px;cursor:pointer;vertical-align:1px}",
    ".cite:hover,.cite:focus{background:var(--brand);color:#fff;outline:none}",
    ".sources{margin-top:8px;border-top:1px dashed var(--line);padding-top:6px;font-size:12px;color:var(--muted)}",
    ".sources summary{cursor:pointer}",
    ".src{margin:6px 0 0;padding:6px 8px;border-radius:8px;background:#f6f8f7}",
    ".src.flash{outline:2px solid var(--brand)}",
    ".src b{color:#1f2a24}.src div{white-space:pre-wrap;margin-top:4px;color:#33403a;max-height:160px;overflow:auto}",
    ".handoff{margin-top:8px;padding:8px;border-radius:8px;background:#fff7e6;border:1px solid #f1dcae;font-size:13px}",
    ".rate{margin-top:6px;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px}",
    ".rate button{border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer;padding:1px 7px;font-size:13px}",
    ".rate button:hover{border-color:var(--brand)}",
    ".note{align-self:center;font-size:12px;color:var(--muted)}",
    ".error{align-self:center;font-size:13px;color:#9b2c2c;background:#fdecec;padding:6px 10px;border-radius:8px}",
    "form{display:flex;gap:8px;padding:10px;border-top:1px solid var(--line);background:#fff}",
    "textarea{flex:1;resize:none;border:1px solid var(--line);border-radius:10px;padding:9px 10px;",
    "  font:inherit;font-size:14px;height:42px;max-height:120px}",
    "textarea:focus{outline:2px solid var(--brand);border-color:transparent}",
    "form button{background:var(--brand);color:#fff;border:0;border-radius:10px;padding:0 16px;",
    "  font:inherit;font-weight:600;cursor:pointer}",
    "form button:disabled{opacity:.5;cursor:default}",
    ".bubble{position:fixed;right:20px;bottom:20px;width:60px;height:60px;border-radius:50%;border:0;",
    "  background:var(--brand);color:#fff;font-size:26px;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.2)}",
    ".panel{position:fixed;right:20px;bottom:92px;width:380px;height:560px;max-height:calc(100vh - 110px);",
    "  box-shadow:0 12px 40px rgba(0,0,0,.18);border-radius:14px}",
    ".panel[hidden]{display:none}",
    "@media (max-width:480px){.panel{right:8px;left:8px;width:auto;bottom:84px;height:calc(100vh - 100px)}}"
  ].join("\n");

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // Inline text: **bold** and [1] / [1][2] / [1, 2] citations. Everything else stays plain text.
  function inline(parent, text, onCite) {
    var re = /\*\*(.+?)\*\*|\[(\d+(?:\s*,\s*\d+)*)\]/g, last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      if (m[1] !== undefined) {
        parent.appendChild(el("strong", "", m[1]));
      } else {
        m[2].split(",").forEach(function (n) {
          n = parseInt(n, 10);
          var b = el("button", "cite", String(n));
          b.type = "button";
          b.addEventListener("click", function () { onCite(n); });
          parent.appendChild(b);
        });
      }
      last = re.lastIndex;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  // Paragraphs and "* " / "- " bullet lists.
  function render(parent, text, onCite) {
    var list = null, para = null;
    text.split("\n").forEach(function (line) {
      var bullet = /^\s*[*-]\s+(.*)$/.exec(line);
      if (bullet) {
        para = null;
        if (!list) { list = el("ul"); parent.appendChild(list); }
        var li = el("li"); inline(li, bullet[1], onCite); list.appendChild(li);
      } else if (!line.trim()) {
        list = null; para = null;
      } else {
        list = null;
        if (!para) { para = el("p"); parent.appendChild(para); } else { para.appendChild(el("br")); }
        inline(para, line, onCite);
      }
    });
  }

  function pickLang(wanted) {
    var l = (wanted || navigator.language || "en").slice(0, 2).toLowerCase();
    return TEXT[l] ? l : "en";
  }

  function apiBase() {
    if (script && script.dataset.api) return script.dataset.api.replace(/\/$/, "");
    if (script && script.src) return new URL(script.src).origin;
    return location.origin;
  }

  /*
   * Build a chat inside `host`. Options:
   *   assistant   "external" (default) or "internal"
   *   api         server address (default: where chat.js was loaded from)
   *   lang        interface language "en" / "id" (answers always follow the language of the question)
   *   getToken    internal only: returns the sign-in token
   *   onSignedOut internal only: called when the server says the token is no longer valid
   *   onClose     show a close button that calls this
   */
  function mount(host, opts) {
    opts = opts || {};
    var assistant = opts.assistant === "internal" ? "internal" : "external";
    var t = TEXT[pickLang(opts.lang)];
    var api = (opts.api || apiBase()).replace(/\/$/, "");
    var history = [];
    var busy = false;

    var root = host.shadowRoot || host.attachShadow({ mode: "open" });
    root.replaceChildren();
    var style = el("style"); style.textContent = CSS; root.appendChild(style);

    var box = el("div", "chat");
    box.setAttribute("role", "region");
    box.setAttribute("aria-label", t.title[assistant]);
    var head = el("div", "head"); head.appendChild(el("span", "", t.title[assistant]));
    if (opts.onClose) {
      var x = el("button", "", "×"); x.type = "button"; x.setAttribute("aria-label", t.close);
      x.addEventListener("click", opts.onClose); head.appendChild(x);
    }
    var log = el("div", "log"); log.setAttribute("aria-live", "polite");
    var form = el("form");
    var input = el("textarea"); input.placeholder = t.placeholder; input.maxLength = 1000;
    input.setAttribute("aria-label", t.placeholder);
    var send = el("button", "", t.send); send.type = "submit";
    form.appendChild(input); form.appendChild(send);
    box.appendChild(head); box.appendChild(log); box.appendChild(form);
    root.appendChild(box);

    log.appendChild(el("div", "msg bot", t.hello[assistant]));

    function scroll() { log.scrollTop = log.scrollHeight; }

    function addAnswer(data) {
      var bubble = el("div", "msg bot");
      var sourceBoxes = {};
      var details = null;

      function showSource(n) {
        if (!details || !sourceBoxes[n]) return;
        details.open = true;
        var s = sourceBoxes[n];
        s.classList.add("flash");
        s.scrollIntoView({ block: "nearest", behavior: "smooth" });
        setTimeout(function () { s.classList.remove("flash"); }, 1200);
      }

      var body = el("div");
      render(body, data.answer, showSource);
      bubble.appendChild(body);

      var cited = (data.sources || []).filter(function (s) { return s.cited; });
      if (cited.length) {
        details = el("details", "sources");
        details.appendChild(el("summary", "", t.sources + " (" + cited.length + ")"));
        cited.forEach(function (s) {
          var item = el("div", "src");
          var label = "[" + s.number + "] " + s.title + (s.section && s.section !== s.title ? " › " + s.section : "");
          item.appendChild(el("b", "", label));
          item.appendChild(document.createTextNode(
            " · " + (s.collection === "tool" ? t.tool : t.updated + " " + s.last_updated)));
          item.appendChild(el("div", "", s.text));
          sourceBoxes[s.number] = item;
          details.appendChild(item);
        });
        bubble.appendChild(details);
      }

      if (data.handoff) {
        var h = el("div", "handoff");
        h.appendChild(el("strong", "", t.contact + " "));
        h.appendChild(document.createTextNode(data.handoff));
        bubble.appendChild(h);
      }

      if (data.answer_id) {
        var rate = el("div", "rate"); rate.appendChild(el("span", "", t.helpful));
        [[1, "👍"], [-1, "👎"]].forEach(function (pair) {
          var b = el("button", "", pair[1]); b.type = "button";
          b.setAttribute("aria-label", pair[0] > 0 ? "helpful" : "not helpful");
          b.addEventListener("click", function () {
            fetch(api + "/feedback", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ answer_id: data.answer_id, rating: pair[0] })
            }).catch(function () {});
            rate.textContent = t.thanks;
          });
          rate.appendChild(b);
        });
        bubble.appendChild(rate);
      }
      log.appendChild(bubble);
      scroll();
    }

    function addNote(cls, text) { var n = el("div", cls, text); log.appendChild(n); scroll(); return n; }

    function ask(question) {
      busy = true; send.disabled = true;
      log.appendChild(el("div", "msg user", question)); scroll();
      var waiting = addNote("note", t.thinking);
      var headers = { "Content-Type": "application/json" };
      if (assistant === "internal" && opts.getToken) headers.Authorization = "Bearer " + opts.getToken();

      fetch(api + "/chat/" + assistant, {
        method: "POST", headers: headers,
        body: JSON.stringify({ question: question, history: history.slice(-MAX_HISTORY) })
      }).then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          waiting.remove();
          if (!res.ok) {
            addNote("error", t.errors[res.status] || t.errors.other);
            if (res.status === 401 && opts.onSignedOut) opts.onSignedOut();
            return;
          }
          addAnswer(data);
          // Blocked turns are shown but not sent back as history, so they cannot steer later answers.
          if (data.status !== "blocked_input" && data.status !== "blocked_output") {
            history.push({ role: "user", content: question.slice(0, 2000) },
                         { role: "assistant", content: data.answer.slice(0, 2000) });
          }
        });
      }).catch(function () {
        waiting.remove();
        addNote("error", t.errors.other);
      }).then(function () {
        busy = false; send.disabled = false; input.focus();
      });
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var q = input.value.trim();
      if (!q || busy) return;
      input.value = "";
      ask(q);
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });

    return { focus: function () { input.focus(); } };
  }

  // The floating website widget: a bubble that opens a chat panel.
  function widget(opts) {
    var t = TEXT[pickLang(opts.lang)];
    var host = el("div");
    host.id = "brightnest-chat";
    document.body.appendChild(host);
    var root = host.attachShadow({ mode: "open" });
    var style = el("style"); style.textContent = CSS; root.appendChild(style);
    var bubble = el("button", "bubble", "💬");
    bubble.type = "button";
    bubble.setAttribute("aria-label", t.open);
    var panelHost = el("div", "panel");
    panelHost.hidden = true;
    root.appendChild(panelHost); root.appendChild(bubble);
    var chat = null;
    function toggle() {
      panelHost.hidden = !panelHost.hidden;
      bubble.setAttribute("aria-expanded", String(!panelHost.hidden));
      if (!panelHost.hidden) {
        if (!chat) chat = mount(panelHost, Object.assign({}, opts, { onClose: toggle }));
        chat.focus();
      }
    }
    bubble.addEventListener("click", toggle);
  }

  window.BrightnestChat = { mount: mount, widget: widget };

  if (script && script.hasAttribute("data-widget")) {
    var start = function () { widget({ assistant: "external", lang: script.dataset.lang, api: apiBase() }); };
    if (document.body) start(); else document.addEventListener("DOMContentLoaded", start);
  }
})();
