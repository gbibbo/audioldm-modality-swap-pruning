/**
 * Minimal results receiver for the blinded listening study.
 *
 * Deploy as: Apps Script > Deploy > New deployment > Web app
 *   - Execute as: Me
 *   - Who has access: Anyone
 * Copy the /exec URL into listening_study/config.js RESULTS_ENDPOINT at deploy time.
 *
 * Configuration lives in Script Properties (Project Settings > Script properties),
 * NOT in this file and NOT in the public client bundle:
 *   RECIPIENTS       comma-separated email address(es) to receive results
 *   EXPECT_STUDY     expected study_version string (optional; reject if mismatch)
 *   EXPECT_PROTOCOL  expected protocol_hash string (optional; reject if mismatch)
 *
 * Privacy: this script intentionally does NOT read or store IP address,
 * user-agent, geolocation, or any personal metadata. It only forwards the
 * anonymous responses JSON that the client sends.
 */
function doPost(e) {
  try {
    var props = PropertiesService.getScriptProperties();
    var recipients = props.getProperty('RECIPIENTS');
    if (!recipients) return _json({ ok: false, error: 'receiver not configured' });

    var body = (e && e.postData && e.postData.contents) ? e.postData.contents : '';
    var data;
    try { data = JSON.parse(body); } catch (err) { return _json({ ok: false, error: 'bad json' }); }

    var expStudy = props.getProperty('EXPECT_STUDY');
    var expProto = props.getProperty('EXPECT_PROTOCOL');
    if (expStudy && data.study_version !== expStudy) return _json({ ok: false, error: 'study mismatch' });
    if (expProto && data.protocol_hash !== expProto) return _json({ ok: false, error: 'protocol mismatch' });

    var pc = String(data.participant_code || 'P??').replace(/[^A-Za-z0-9_]/g, '');
    var uuid = String(data.submission_uuid || '').replace(/[^A-Za-z0-9_-]/g, '');
    var subject = 'Listening study result: ' + pc + ' (' + (data.study_version || '?') + ')';
    var n = (data.responses && data.responses.length) || 0;
    var summary = 'participant=' + pc + '  responses=' + n +
      '  total_ms=' + (data.total_ms || '?') + '  uuid=' + uuid + '\n\n';
    var attachment = Utilities.newBlob(JSON.stringify(data, null, 2),
      'application/json', 'listening_' + pc + '_' + uuid + '.json');

    MailApp.sendEmail({
      to: recipients,
      subject: subject,
      body: summary + 'Full anonymous responses attached as JSON.',
      attachments: [attachment]
    });
    return _json({ ok: true, received: n });
  } catch (err) {
    return _json({ ok: false, error: String(err) });
  }
}

function doGet() {
  return _json({ ok: true, service: 'listening-study-receiver', method: 'POST only' });
}

function _json(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
