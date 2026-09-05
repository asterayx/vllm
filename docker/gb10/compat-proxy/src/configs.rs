// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Client configs for Codex, Grok Build, and OpenCode.

pub struct ConfigContext<'a> {
    pub public_base: &'a str,
    pub model: &'a str,
    pub display_name: &'a str,
    pub context_window: u32,
}

impl ConfigContext<'_> {
    fn api_base(&self) -> String {
        format!("{}/v1", self.public_base.trim_end_matches('/'))
    }
}

pub fn codex_toml(ctx: &ConfigContext<'_>) -> String {
    let api = ctx.api_base();
    format!(
        r#"# Codex CLI → Spark vLLM (via spark-compat-proxy)
# Install:
#   mkdir -p ~/.codex
#   curl -fsS {pub}/configs/codex.toml -o ~/.codex/config.toml
#   export VLLM_API_KEY=dummy
#
# Codex only accepts wire_api = "responses" (chat was removed Feb 2026).

model = "{model}"
model_provider = "vllm"

[model_providers.vllm]
name = "vLLM Spark"
env_key = "VLLM_API_KEY"
base_url = "{api}"
wire_api = "responses"
# Long DSv4 generations; default 300s is too short.
stream_idle_timeout_ms = 600000
request_max_retries = 4
"#,
        model = ctx.model,
        api = api,
        pub = ctx.public_base.trim_end_matches('/'),
    )
}

pub fn grok_toml(ctx: &ConfigContext<'_>) -> String {
    let api = ctx.api_base();
    format!(
        r#"# Grok Build → Spark vLLM (via spark-compat-proxy)
# Merge the [model."{model}"] block into ~/.grok/config.toml
#   mkdir -p ~/.grok
#   curl -fsS {pub}/configs/grok.toml >> ~/.grok/config.toml
#   export VLLM_API_KEY=dummy
#   grok --model {model}
#
# Do not replace your existing [models] default unless you want Spark
# for every new session. Uncomment below to do that.
#
# [models]
# default = "{model}"

[model."{model}"]
model = "{model}"
base_url = "{api}"
name = "{display}"
description = "{display} on 2x DGX Spark"
env_key = "VLLM_API_KEY"
api_backend = "responses"
context_window = {context_window}
"#,
        model = ctx.model,
        api = api,
        display = ctx.display_name,
        context_window = ctx.context_window,
        pub = ctx.public_base.trim_end_matches('/'),
    )
}

pub fn opencode_json(ctx: &ConfigContext<'_>) -> String {
    let api = ctx.api_base();
    // Provider id must not contain '.': OpenCode drops options if it does.
    // `api` is the reliable base URL; `options.baseURL` is the documented
    // form (some OpenCode builds only honor one of the two).
    let mut models = serde_json::Map::new();
    models.insert(
        ctx.model.to_string(),
        serde_json::json!({
            "name": ctx.display_name,
            "reasoning": true,
            "limit": { "context": ctx.context_window }
        }),
    );
    let value = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "model": format!("spark/{}", ctx.model),
        "provider": {
            "spark": {
                "npm": "@ai-sdk/openai-compatible",
                "name": ctx.display_name,
                "api": api,
                "options": {
                    "baseURL": api,
                    "apiKey": "dummy"
                },
                "models": models
            }
        }
    });
    let mut out = serde_json::to_string_pretty(&value).expect("opencode json");
    out.push('\n');
    out
}

pub fn index_html(ctx: &ConfigContext<'_>) -> String {
    let api = ctx.api_base();
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Spark compat proxy</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }}
    code, pre {{ background: #f4f4f4; padding: 0.15em 0.35em; }}
    pre {{ padding: 0.8rem 1rem; overflow: auto; }}
    a {{ color: #0b57d0; }}
  </style>
</head>
<body>
  <h1>Spark OpenAI compat proxy</h1>
  <p>Upstream rewrite: <code>reasoning</code> → <code>reasoning_content</code>.
     API base: <code>{api}</code> · model <code>{model}</code></p>
  <p><a href="/healthz">/healthz</a></p>
  <h2>Download client configs</h2>
  <ul>
    <li><a href="/configs/codex.toml" download="config.toml">Codex</a> → <code>~/.codex/config.toml</code></li>
    <li><a href="/configs/grok.toml" download="config.toml">Grok Build</a> → <code>~/.grok/config.toml</code></li>
    <li><a href="/configs/opencode.json" download="opencode.json">OpenCode</a> → <code>~/.config/opencode/opencode.json</code></li>
  </ul>
  <h2>Quick start</h2>
  <pre>export VLLM_API_KEY=dummy
curl -fsS {pub}/healthz
# Codex
curl -fsS {pub}/configs/codex.toml -o ~/.codex/config.toml
# Grok Build
curl -fsS {pub}/configs/grok.toml -o ~/.grok/config.toml
# OpenCode
mkdir -p ~/.config/opencode
curl -fsS {pub}/configs/opencode.json -o ~/.config/opencode/opencode.json</pre>
</body>
</html>
"#,
        api = api,
        model = ctx.model,
        pub = ctx.public_base.trim_end_matches('/'),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ConfigContext<'static> {
        ConfigContext {
            public_base: "http://192.168.8.134:30000",
            model: "deepseek-v4-flash-vision-exp",
            display_name: "DeepSeek-V4-Flash-Vision-Exp",
            context_window: 524288,
        }
    }

    #[test]
    fn codex_has_base_and_responses() {
        let t = codex_toml(&ctx());
        assert!(t.contains("base_url = \"http://192.168.8.134:30000/v1\""));
        assert!(t.contains("wire_api = \"responses\""));
        assert!(t.contains("deepseek-v4-flash-vision-exp"));
    }

    #[test]
    fn grok_has_model_block() {
        let t = grok_toml(&ctx());
        assert!(t.contains("[model.\"deepseek-v4-flash-vision-exp\"]"));
        assert!(t.contains("api_backend = \"responses\""));
        assert!(t.contains("base_url = \"http://192.168.8.134:30000/v1\""));
    }

    #[test]
    fn opencode_provider_has_no_dot() {
        let t = opencode_json(&ctx());
        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
        assert_eq!(v["model"], "spark/deepseek-v4-flash-vision-exp");
        assert_eq!(
            v["provider"]["spark"]["options"]["baseURL"],
            "http://192.168.8.134:30000/v1"
        );
        assert_eq!(
            v["provider"]["spark"]["api"],
            "http://192.168.8.134:30000/v1"
        );
        assert!(t.contains("\"spark\""));
        assert!(!t.contains("spark.vllm"));
    }

    #[test]
    fn client_configs_use_the_served_model_and_context_limit() {
        let ctx = ConfigContext {
            public_base: "https://token.asterayx.com",
            model: "qwen38-nvfp4",
            display_name: "Qwen3.8 Flash NVFP4",
            context_window: 16384,
        };
        let grok = grok_toml(&ctx);
        assert!(grok.contains("[model.\"qwen38-nvfp4\"]"));
        assert!(grok.contains("context_window = 16384"));
        assert!(!grok.contains("DeepSeek"));
        let opencode: serde_json::Value = serde_json::from_str(&opencode_json(&ctx)).unwrap();
        assert_eq!(
            opencode["provider"]["spark"]["models"][ctx.model]["limit"]["context"],
            16384
        );
    }
}
