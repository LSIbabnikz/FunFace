
#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_config="${script_dir}/configs/inference_default.yaml"

for arg in "$@"; do
    if [[ "${arg}" == "-c" || "${arg}" == "--config" ]]; then
        exec python3 "${script_dir}/inference.py" "$@"
    fi
done

exec python3 "${script_dir}/inference.py" -c "${default_config}" "$@"
