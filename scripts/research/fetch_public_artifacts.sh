#!/usr/bin/env bash
# M0 public-artifact fetch. Public, CPU/network only. No GPU, no training.
#
# Sources (all public, all recorded in docs/m0_baseline_reproduction/):
#   Zenodo 10.5281/zenodo.7884686   AudioLDM official checkpoints (CC-BY-4.0 record;
#                                   note upstream README states the pretrained AudioLDM
#                                   checkpoints are CC-BY-NC-4.0 -- no commercial use)
#   Zenodo 10.5281/zenodo.14342967  AudioLDM aux checkpoints + preprocessed AudioCaps
#   Zenodo 10.5281/zenodo.21376822  Arshdeep Singh pruned AudioLDM-M-Full checkpoints
#
# Every file is md5-verified against the value published in the Zenodo record.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p data/checkpoints data/dataset artifacts/m0_baseline_reproduction

fetch() {  # fetch <url> <dest> <expected_md5>
  local url="$1" dest="$2" want="$3"
  if [ -f "$dest" ]; then
    local have; have="$(md5sum "$dest" | cut -d' ' -f1)"
    if [ "$have" = "$want" ]; then echo "OK (cached)   $dest"; return 0; fi
    echo "REDOWNLOAD    $dest (md5 $have != $want)"
  fi
  echo "FETCH         $dest"
  curl -fL --retry 5 --retry-delay 10 --retry-all-errors -C - "$url" -o "$dest" \
    --progress-bar || { echo "FAIL download $dest"; return 1; }
  local have; have="$(md5sum "$dest" | cut -d' ' -f1)"
  if [ "$have" = "$want" ]; then echo "OK md5        $dest"; else
    echo "FAIL md5      $dest got=$have want=$want"; return 1; fi
}

Z7=https://zenodo.org/records/7884686/files
Z14=https://zenodo.org/records/14342967/files
Z21=https://zenodo.org/records/21376822/files

rc=0
fetch "$Z21/sorted_indexes_dict.pkl?download=1" artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl a4cd11ff83438ee0f9aa5fe0917f39e3 || rc=1
fetch "$Z7/audioldm-m-full.ckpt?download=1"     data/checkpoints/audioldm-m-full.ckpt            46bad9f176651404b3cf1484942749b9 || rc=1
fetch "$Z21/Unet_model-m.ckpt?download=1"       data/checkpoints/Unet_model-m.ckpt               e44eaa7cbd5a358111d496d1cd246a33 || rc=1
fetch "$Z21/l1_audioldm-m-full_p1.ckpt?download=1" data/checkpoints/l1_audioldm-m-full_p1.ckpt   2666e6fc108a9c4fc0d19bbf26832905 || rc=1
fetch "$Z14/checkpoints.tar?download=1"         data/checkpoints.tar                             d9898f93372582119fa19c6464f59cdc || rc=1
fetch "$Z14/dataset.tar?download=1"             data/dataset.tar                                 1c4e6642754c38f7041efdfeabe6e32d || rc=1

echo "=== fetch rc=$rc ==="
exit $rc
