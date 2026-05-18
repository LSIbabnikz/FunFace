
#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_config="${script_dir}/configs/train_default_webface4m.yaml"

for arg in "$@"; do
    if [[ "${arg}" == "-c" || "${arg}" == "--config" ]]; then
        exec accelerate launch "${script_dir}/train.py" "$@"
    fi
done

exec accelerate launch "--mixed_precision" "fp16" "--multi_gpu" "${script_dir}/train.py" -c "${default_config}" "$@" 
