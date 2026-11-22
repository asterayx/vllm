# PR checklist: v0.34.0 integration into rtxpro5000

- Scope/design: merge official release 62d4bc3c7 into fork 644379a54; preserve both parents. The 5851-file diff includes the upstream release, not 5851 newly authored custom files. Review the custom delta against v0.34.0 and the integration notes. No source changes were made while publishing this PR.
- Duplicate check: no open PR for this source branch or v0.34.0 integration in asterayx/vllm at publication.
- Compatibility: retain MRV2; port file-backed/MXFP4 Engram to shared upstream implementation; preserve SM120, draft persistence, PP and PCIe customizations. Full acceptance across other models/backends remains unverified.
- Validation: targeted regression groups and focused reruns are documented in the committed integration notes; exact commands/logs in integration-validation.tar.gz. Qwen3-0.6B matches 72 token IDs across 3 prompts against official release. Qwen3.8 TP4 C1: 480/480 measured responses completed, plus warmups; MTP has nonzero acceptance. Both variants include 64-step Nsight and 30-second py-spy evidence.
- Quality: existing integration pre-commit final/final-fixes logs retained. git diff --check against official v0.34.0 passes; C/CUDA source delta against the release is empty. Whole old-fork-to-merge whitespace warnings are from imported release files; do not claim whole-tree whitespace lint passes.
- Documentation: examples/deployment/rtxpro5000_v034_merge.md documents migration and isolated validation; this PR adds complete C1 results as attachments and inline report.
- Limitations: historical, non-interleaved comparison; changed Torch/FlashInfer/runtime; MTP TPOT intervals overlap. Long-output repetition candidates are retained. Performance success is not full-model accuracy acceptance. No comprehensive correctness suite or full upstream CI was run for this publishing task.
- Human review: this is an AI-assisted integration PR for the user's fork; no claim that a human has reviewed every changed line. It is not merged or deployed by this task.
- Contents: repository PR template retained; evidence, reproduction commands, artifact hashes, environment isolation and cleanup included.
