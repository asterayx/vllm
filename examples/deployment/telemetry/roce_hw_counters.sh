#!/bin/sh
# Write /sys/class/infiniband/<dev>/ports/<port>/hw_counters/* as Prometheus
# textfile metrics every ${INTERVAL:-10} seconds.
out=/textfile/roce_hw_counters.prom
while true; do
    tmp="$out.$$"
    {
        echo "# HELP roce_hw_counter Mellanox RDMA hw_counters (monotonic)."
        echo "# TYPE roce_hw_counter counter"
        for f in /ib/*/ports/*/hw_counters/*; do
            [ -f "$f" ] || continue
            v=$(cat "$f" 2>/dev/null) || continue
            case "$v" in '' | *[!0-9]*) continue ;; esac
            port_dir=${f%/hw_counters/*}
            dev=${port_dir%/ports/*}
            echo "roce_hw_counter{device=\"${dev##*/}\",port=\"${port_dir##*/}\",counter=\"${f##*/}\"} $v"
        done
    } > "$tmp" && mv "$tmp" "$out"
    sleep "${INTERVAL:-10}"
done
