//! Copy v0.28 `reasoning` onto Codex/Grok `reasoning_content`.

use serde_json::Value;

/// Recursively copy `reasoning` → `reasoning_content` when the latter is absent.
pub fn alias(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.contains_key("reasoning") && !map.contains_key("reasoning_content") {
                if let Some(reasoning) = map.get("reasoning").cloned() {
                    map.insert("reasoning_content".to_string(), reasoning);
                }
            }
            for child in map.values_mut() {
                alias(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                alias(item);
            }
        }
        _ => {}
    }
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
            alias(&mut obj);
            format!("data: {obj}")
        }
        Err(_) => line.to_string(),
    }
}

/// Alias a complete JSON body. Returns the original bytes if parse fails.
pub fn alias_json_bytes(data: &[u8]) -> Vec<u8> {
    match serde_json::from_slice::<Value>(data) {
        Ok(mut obj) => {
            alias(&mut obj);
            serde_json::to_vec(&obj).unwrap_or_else(|_| data.to_vec())
        }
        Err(_) => data.to_vec(),
    }
}

/// Split a chunk stream into SSE lines, rewriting each `data:` payload.
pub fn rewrite_sse_chunk(pending: &mut String, chunk: &str) -> String {
    pending.push_str(chunk);
    let mut out = String::with_capacity(pending.len() + 32);
    while let Some(idx) = pending.find('\n') {
        let mut line = pending.drain(..=idx).collect::<String>();
        if line.ends_with('\n') {
            line.pop();
        }
        if line.ends_with('\r') {
            line.pop();
        }
        out.push_str(&rewrite_sse_line(&line));
        out.push('\n');
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
        let mut pending = String::new();
        let a = rewrite_sse_chunk(&mut pending, "data: {\"reasoning\":");
        assert!(a.is_empty());
        let b = rewrite_sse_chunk(&mut pending, "\"y\"}\n");
        assert!(b.contains("reasoning_content"));
        assert!(pending.is_empty());
    }
}
