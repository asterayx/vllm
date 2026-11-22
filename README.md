# RTX PRO 5000 vLLM v0.34.0 validation artifacts

Tested source commit: [`d77cff6e2823637d2eab077caa53d78cb6dfce7c`](https://github.com/asterayx/vllm/commit/d77cff6e2823637d2eab077caa53d78cb6dfce7c).

This independent branch stores review evidence. It is not a vLLM source or deployment branch.

- [Complete C1 performance and profiling report](REPORT.md)
- [Results and raw profiling archive](results-and-profiling.tar.gz?raw=true): both raw Nsight reports, Python sampling, native benchmark JSON, parsed request outputs, analysis evidence, commands, scripts and final environment verification.
- [Merge validation archive](integration-validation.tar.gz?raw=true): regression commands/logs, pre-commit evidence, small-model comparison. Initial failures and passing reruns are retained for traceability.
- [PR checklist findings](PR-CHECKLIST.md)
- [SHA-256 checksums](SHA256SUMS)

After downloading the four files listed in `SHA256SUMS`, run:

```sh
sha256sum -c SHA256SUMS
```

Qwen3.8-Flash-Next-NVFP4, xzss, TP4, C1, 480 successful measured responses. Both variants include 64-step Nsight traces and 30-second Python samples. Packet captures remain on xzss; parsed outputs are in the archive. See the report for historical comparison limitations and output-review candidates.
