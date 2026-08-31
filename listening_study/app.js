/* Blinded listening study — static client.
   Loads a per-participant frozen assignment manifest and records comparative
   judgments. No model identity is present in the manifest or on screen.
   Collects NO personal data. Submits a minimal JSON payload; always offers an
   offline fallback so responses cannot be lost. */
(function () {
  "use strict";
  var CFG = window.STUDY_CONFIG || {};
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  var state = {
    participant: null, manifest: null, trials: [], i: 0,
    responses: [], startedAt: null, submitted: false,
    submissionUUID: null, activeQ: "relevance"
  };

  function uuid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "uuid-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }
  function show(id) { $$(".screen").forEach(function (s) { s.hidden = true; }); $(id).hidden = false; }
  function qs(name) {
    var m = new RegExp("[?&]" + name + "=([^&]+)").exec(location.search);
    return m ? decodeURIComponent(m[1]) : null;
  }

  /* ---------- load ---------- */
  function boot() {
    var p = qs("p");
    state.submissionUUID = uuid();
    if (!p || !/^P0[1-9]$|^P1[0-9]$/.test(p)) {
      return fail("No valid participant code in the link (expected ...?p=P01). " +
        "Please use the personal link you were given.");
    }
    state.participant = p;
    $("#consent-participant").textContent = "Participant: " + p;
    var url = "public_manifests/" + p + ".json";
    fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("manifest HTTP " + r.status);
      return r.json();
    }).then(function (m) {
      state.manifest = m;
      state.trials = m.trials || [];
      if (!state.trials.length) throw new Error("empty trial list");
      wireConsent();
      show("#screen-consent");
    }).catch(function (e) {
      fail("Could not load your study assignment (" + e.message + "). " +
        "Please check your link or contact the organiser.");
    });
  }
  function fail(msg) { $("#load-error").textContent = msg; show("#screen-error"); }

  /* ---------- consent ---------- */
  function wireConsent() {
    var ack = $("#ack"), start = $("#btn-start");
    ack.addEventListener("change", function () { start.disabled = !ack.checked; });
    start.addEventListener("click", function () {
      if (!ack.checked) return;
      show("#screen-instructions");
      wireInstructions();
    });
  }

  /* ---------- instructions / level check ---------- */
  var levelAudio = null;
  function wireInstructions() {
    var lp = $("#btn-levelplay");
    lp.addEventListener("click", function () {
      var f = state.manifest.level_check_audio;
      if (!f) { $("#levelcheck-note").textContent = "(no level-check clip configured)"; return; }
      if (!levelAudio) levelAudio = new Audio(f);
      levelAudio.currentTime = 0; levelAudio.play();
      $("#levelcheck-note").textContent = "Playing… set a comfortable level.";
    });
    $("#btn-begin").addEventListener("click", function () {
      state.startedAt = Date.now();
      renderTrial();
      show("#screen-trial");
    });
  }

  /* ---------- trials ---------- */
  var players = {}, playCounts = {}, current = null;

  function renderTrial() {
    var t = state.trials[state.i];
    current = {
      trial_id: t.trial_id, type: t.type, shownAt: Date.now(),
      relevance: null, quality: null,
      playsA: 0, playsB: 0, firstResponseAt: null
    };
    $("#progress-bar").style.width = ((state.i) / state.trials.length * 100) + "%";
    $("#progress-text").textContent = "Trial " + (state.i + 1) + " of " + state.trials.length;
    $("#prompt-text").textContent = t.prompt_text || "";

    // reset audio + counts
    stopAll();
    players = {
      A: new Audio(t.audio_A),
      B: new Audio(t.audio_B)
    };
    playCounts = { A: 0, B: 0 };
    $$(".replay-note").forEach(function (n) { n.textContent = ""; });
    $$(".play").forEach(function (b) { b.classList.remove("playing"); b.disabled = false; });

    // reset answers
    $$(".scale button").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
    $("#btn-next").disabled = true;
    setActiveQuestion("relevance");
  }

  function stopAll() {
    ["A", "B"].forEach(function (s) {
      if (players[s]) { try { players[s].pause(); } catch (e) {} }
    });
    $$(".play").forEach(function (b) { b.classList.remove("playing"); });
  }

  function playSide(side) {
    if (playCounts[side] >= 2) return; // original + one replay
    stopAll();
    var a = players[side];
    if (!a) return;
    a.currentTime = 0;
    a.play();
    playCounts[side]++;
    current["plays" + side] = playCounts[side];
    var btn = $('.play[data-side="' + side + '"]');
    btn.classList.add("playing");
    var note = $('.replay-note[data-side="' + side + '"]');
    note.textContent = playCounts[side] >= 2 ? "no replays left" : "1 replay left";
    if (playCounts[side] >= 2) btn.disabled = true;
    a.onended = function () { btn.classList.remove("playing"); };
  }

  function setActiveQuestion(q) {
    state.activeQ = q;
    $("#q-relevance").classList.toggle("active", q === "relevance");
    $("#q-quality").classList.toggle("active", q === "quality");
  }

  function answer(q, v) {
    current[q] = v;
    if (!current.firstResponseAt) current.firstResponseAt = Date.now();
    var wrap = $('.scale[data-q="' + q + '"]');
    $$("button", wrap).forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-v") === String(v) ? "true" : "false");
    });
    if (q === "relevance") setActiveQuestion("quality");
    $("#btn-next").disabled = (current.relevance === null || current.quality === null);
  }

  function next() {
    if (current.relevance === null || current.quality === null) return;
    stopAll();
    current.respondedAt = Date.now();
    current.dwell_ms = current.respondedAt - current.shownAt;
    state.responses.push({
      trial_id: current.trial_id, type: current.type,
      relevance: current.relevance, quality: current.quality,
      plays_A: current.playsA, plays_B: current.playsB,
      shown_ts: current.shownAt, responded_ts: current.respondedAt,
      dwell_ms: current.dwell_ms
    });
    state.i++;
    if (state.i >= state.trials.length) {
      $("#progress-bar").style.width = "100%";
      show("#screen-done");
      wireSubmit();
    } else {
      renderTrial();
    }
  }

  /* ---------- keyboard ---------- */
  document.addEventListener("keydown", function (e) {
    if ($("#screen-trial").hidden) return;
    var k = e.key.toLowerCase();
    if (k === "a") { playSide("A"); e.preventDefault(); }
    else if (k === "b") { playSide("B"); e.preventDefault(); }
    else if (["1", "2", "3", "4", "5"].indexOf(k) >= 0) {
      answer(state.activeQ, parseInt(k, 10) - 3); e.preventDefault();
    } else if (k === "enter") { if (!$("#btn-next").disabled) next(); }
  });

  function wireTrialButtons() {
    $$(".play").forEach(function (b) {
      b.addEventListener("click", function () { playSide(b.getAttribute("data-side")); });
    });
    $$(".scale").forEach(function (wrap) {
      var q = wrap.getAttribute("data-q");
      $$("button", wrap).forEach(function (b) {
        b.addEventListener("click", function () { answer(q, parseInt(b.getAttribute("data-v"), 10)); });
      });
    });
    $("#btn-next").addEventListener("click", next);
  }

  /* ---------- submit ---------- */
  function buildPayload() {
    return {
      study_version: state.manifest.study_version,
      protocol_hash: state.manifest.protocol_hash,
      participant_code: state.participant,
      assignment_hash: state.manifest.assignment_hash,
      submission_uuid: state.submissionUUID,
      client_started_ts: state.startedAt,
      client_completed_ts: Date.now(),
      total_ms: Date.now() - state.startedAt,
      responses: state.responses
    };
  }

  function wireSubmit() {
    var payload = null;
    $("#btn-submit").addEventListener("click", function () {
      if (state.submitted) return;
      payload = buildPayload();
      submit(payload);
    });
    $("#btn-download").addEventListener("click", function () {
      var p = payload || buildPayload();
      var blob = new Blob([JSON.stringify(p, null, 2)], { type: "application/json" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "listening_" + state.participant + "_" + state.submissionUUID + ".json";
      document.body.appendChild(a); a.click(); a.remove();
    });
    $("#btn-copy").addEventListener("click", function () {
      var p = payload || buildPayload();
      var txt = JSON.stringify(p);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(function () {
          $("#submit-status").textContent = "Copied to clipboard.";
        });
      } else {
        var ta = document.createElement("textarea"); ta.value = txt;
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        ta.remove();
        $("#submit-status").textContent = "Copied to clipboard.";
      }
    });
  }

  function submit(payload) {
    var status = $("#submit-status");
    var ep = CFG.RESULTS_ENDPOINT;
    $("#btn-submit").disabled = true;
    status.textContent = "Submitting…";
    if (!ep) { onSubmitFail("no endpoint configured"); return; }
    // text/plain avoids CORS preflight for Apps Script; try to read {ok:true}
    fetch(ep, {
      method: "POST", mode: "cors",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify(payload)
    }).then(function (r) {
      return r.text().then(function (txt) {
        var ok = false;
        try { ok = JSON.parse(txt).ok === true; } catch (e) { ok = r.ok; }
        if (ok) onSubmitOk(); else onSubmitFail("server did not confirm");
      });
    }).catch(function (e) { onSubmitFail(e.message); });
  }

  function onSubmitOk() {
    state.submitted = true;
    $("#submit-status").textContent = "Submitted successfully. Thank you — you may close this tab.";
    $("#fallback").hidden = true;
    $("#btn-submit").hidden = true;
  }
  function onSubmitFail(reason) {
    $("#btn-submit").disabled = false;
    $("#submit-status").textContent = "Could not confirm automatic submission (" + reason + ").";
    $("#fallback").hidden = false;
  }

  wireTrialButtons();
  boot();
})();
