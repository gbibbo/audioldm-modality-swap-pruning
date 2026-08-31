/* Deployment configuration for the listening study client.
 *
 * RESULTS_ENDPOINT: HTTPS POST target that receives the results JSON and emails
 * it onward. Leave EMPTY in the committed repo; set it at deploy time to the
 * Google Apps Script web-app URL (see receiver/google_apps_script/). The email
 * recipient address is configured server-side in the Apps Script, NEVER here.
 *
 * If RESULTS_ENDPOINT is empty or the POST fails, the client shows the offline
 * fallback (download JSON / copy to clipboard) so no response is ever lost.
 */
window.STUDY_CONFIG = {
  RESULTS_ENDPOINT: ""   // e.g. "https://script.google.com/macros/s/XXXX/exec"
};
