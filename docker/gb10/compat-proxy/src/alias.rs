// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Copy v0.28 `reasoning` onto Codex/Grok `reasoning_content`.

use serde_json::Value;

/// Recursively copy `reasoning` → `reasoning_content` when the latter is absent.
pub fn alias(value: &mut Value) -> bool {
    let mut changed = false;
    match value {
        Value::Object(map) => {
            for child in map.values_mut() {
                changed |= alias(child);
            }
            if map.contains_key("reasoning") && !map.contains_key("reasoning_content") {
                if let Some(reasoning) = map.get("reasoning").cloned() {
                    map.insert("reasoning_content".to_string(), reasoning);
                    changed = true;
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                changed |= alias(item);
            }
        }
        _ => {}
    }
    changed
}

/// Rewrite one SSE line. Non-`data:` lines are unchanged.
pub fn rewrite_sse_line(line: &str) -> String {
    let Some(payload) = line.strip_prefix("data: ") else {
        return line.to_string();
    };
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return line.to_string();
    }
    match serde_json::from_str::<Value>(payload) {
        Ok(mut obj) => {
            if alias(&mut obj) {
                format!("data: {obj}")
            } else {
                line.to_string()
            }
        }
        Err(_) => line.to_string(),
    }
}

/// Alias a complete JSON body. Returns the original bytes if parse fails.
pub fn alias_json_bytes(data: &[u8]) -> Vec<u8> {
    match serde_json::from_slice::<Value>(data) {
        Ok(mut obj) => {
            if alias(&mut obj) {
                serde_json::to_vec(&obj).unwrap_or_else(|_| data.to_vec())
            } else {
                data.to_vec()
            }
        }
        Err(_) => data.to_vec(),
    }
}

/// Split a chunk stream into SSE lines, rewriting each `data:` payload.
pub fn rewrite_sse_chunk(pending: &mut Vec<u8>, chunk: &[u8]) -> String {
    let scan_start = pending.len();
    pending.extend_from_slice(chunk);
    let mut out = String::new();
    let mut line_start = 0;
    for (offset, &byte) in pending[scan_start..].iter().enumerate() {
        if byte != b'\n' {
            continue;
        }
        let line_end = scan_start + offset;
        let line = String::from_utf8_lossy(&pending[line_start..line_end]);
        out.push_str(&rewrite_sse_line(line.strip_suffix('\r').unwrap_or(&line)));
        out.push('\n');
        line_start = line_end + 1;
    }
    if line_start > 0 {
        pending.drain(..line_start);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn copies_reasoning_when_missing() {
        let mut v = json!({"choices":[{"delta":{"reasoning":"think"}}]});
        alias(&mut v);
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "think");
        assert_eq!(v["choices"][0]["delta"]["reasoning"], "think");
    }

    #[test]
    fn does_not_overwrite_existing() {
        let mut v = json!({"reasoning":"new","reasoning_content":"keep"});
        alias(&mut v);
        assert_eq!(v["reasoning_content"], "keep");
    }

    #[test]
    fn unchanged_payload_preserves_formatting() {
        let body = br#"{ "choices": [{"delta": {"content": "hello"}}] }"#;
        assert_eq!(alias_json_bytes(body), body);
        let line = format!("data: {}", std::str::from_utf8(body).unwrap());
        assert_eq!(rewrite_sse_line(&line), line);
    }

    #[test]
    fn long_fragmented_lines_match_complete_stream() {
        let input = format!(
            "data: {{\"reasoning\":\"{}\"}}\n\ndata: [DONE]\n",
            "图".repeat(4096)
        );
        let expected = rewrite_sse_chunk(&mut Vec::new(), input.as_bytes());
        for size in [1, 7, 1024, input.len()] {
            let mut pending = Vec::new();
            let mut output = String::new();
            for chunk in input.as_bytes().chunks(size) {
                output.push_str(&rewrite_sse_chunk(&mut pending, chunk));
            }
            assert_eq!(output, expected);
            assert!(pending.is_empty());
        }
    }

    #[test]
    fn rewrite_sse_keeps_done() {
        assert_eq!(rewrite_sse_line("data: [DONE]"), "data: [DONE]");
        assert_eq!(rewrite_sse_line(": ping"), ": ping");
        let out = rewrite_sse_line(r#"data: {"reasoning":"x"}"#);
        let payload = out.strip_prefix("data: ").unwrap();
        let v: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(v["reasoning_content"], "x");
    }

    #[test]
    fn rewrite_sse_chunk_handles_split_lines() {
        let mut pending = Vec::new();
        let a = rewrite_sse_chunk(&mut pending, b"data: {\"reasoning\":");
        assert!(a.is_empty());
        let b = rewrite_sse_chunk(&mut pending, b"\"y\"}\n");
        assert!(b.contains("reasoning_content"));
        assert!(pending.is_empty());
    }

    #[test]
    fn rewrite_sse_preserves_utf8_at_every_chunk_boundary() {
        let input = "data: {\"reasoning\":\"图片中有一只猫🐈\"}\r\n\r\ndata: [DONE]\n\n";
        let expected = rewrite_sse_chunk(&mut Vec::new(), input.as_bytes());
        assert!(expected.contains("图片中有一只猫🐈"));
        assert!(expected.contains("reasoning_content"));
        for split in 0..=input.len() {
            let mut pending = Vec::new();
            let mut output = rewrite_sse_chunk(&mut pending, &input.as_bytes()[..split]);
            output.push_str(&rewrite_sse_chunk(&mut pending, &input.as_bytes()[split..]));
            assert_eq!(output, expected, "split at byte {split}");
            assert!(pending.is_empty());
        }
    }
}
