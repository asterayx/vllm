//! Production OpenAI-compat proxy for Spark vLLM.

mod alias;
mod configs;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use alias::{alias_json_bytes, rewrite_sse_chunk, rewrite_sse_line};
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use clap::{Parser, Subcommand};
use configs::ConfigContext;
use futures_util::StreamExt;
use reqwest::Client;
use tracing::{info, warn};

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

#[derive(Clone)]
struct AppState {
    client: Client,
    upstream: String,
    public_base: String,
    model: String,
    display_name: String,
}

impl AppState {
    fn ctx(&self) -> ConfigContext<'_> {
        ConfigContext {
            public_base: &self.public_base,
            model: &self.model,
            display_name: &self.display_name,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "spark-compat-proxy", about = "Spark vLLM OpenAI compat proxy")]
struct Args {
    /// Listen address (the Codex/Grok/OpenCode port).
    #[arg(
        long,
        default_value = "0.0.0.0:30000",
        env = "SPARK_COMPAT_LISTEN",
        global = true
    )]
    listen: String,
    /// vLLM OpenAI server.
    #[arg(
        long,
        default_value = "http://127.0.0.1:30001",
        env = "SPARK_COMPAT_UPSTREAM",
        global = true
    )]
    upstream: String,
    /// URL clients should use (configs + download page).
    #[arg(long, env = "SPARK_COMPAT_PUBLIC_BASE", global = true)]
    public_base: Option<String>,
    #[arg(
        long,
        default_value = "deepseek-v4-flash-vision-exp",
        env = "SPARK_COMPAT_MODEL",
        global = true
    )]
    model: String,
    #[arg(
        long,
        default_value = "DeepSeek-V4-Flash-Vision-Exp",
        env = "SPARK_COMPAT_DISPLAY_NAME",
        global = true
    )]
    display_name: String,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Write Codex / Grok / OpenCode configs to a directory and exit.
    WriteConfigs {
        #[arg(long, default_value = ".")]
        out: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let public_base = args.public_base.unwrap_or_else(|| {
        default_public_base(&args.listen)
    });

    if let Some(Command::WriteConfigs { out }) = args.command {
        write_config_files(
            &out,
            &ConfigContext {
                public_base: &public_base,
                model: &args.model,
                display_name: &args.display_name,
            },
        )?;
        return Ok(());
    }

    let client = Client::builder()
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(5))
        // No total timeout: DSv4 streams can run past 10 minutes.
        // Stall if the upstream goes silent between SSE chunks.
        .read_timeout(Duration::from_secs(600))
        .build()?;

    let state = AppState {
        client,
        upstream: args.upstream.trim_end_matches('/').to_string(),
        public_base,
        model: args.model,
        display_name: args.display_name,
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/configs", get(configs_index))
        .route("/configs/", get(configs_index))
        .route("/configs/codex.toml", get(download_codex))
        .route("/configs/grok.toml", get(download_grok))
        .route("/configs/opencode.json", get(download_opencode))
        .fallback(proxy)
        .with_state(state.clone());

    let addr: SocketAddr = args.listen.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(
        listen = %addr,
        upstream = %state.upstream,
        public_base = %state.public_base,
        "spark-compat-proxy listening (reasoning => reasoning_content)"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn default_public_base(listen: &str) -> String {
    let port = listen.rsplit(':').next().unwrap_or("30000");
    format!("http://127.0.0.1:{port}")
}

fn write_config_files(
    out: &std::path::Path,
    ctx: &ConfigContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("codex.toml"), configs::codex_toml(ctx))?;
    std::fs::write(out.join("grok.toml"), configs::grok_toml(ctx))?;
    std::fs::write(out.join("opencode.json"), configs::opencode_json(ctx))?;
    std::fs::write(out.join("index.html"), configs::index_html(ctx))?;
    info!(dir = %out.display(), "wrote Codex / Grok / OpenCode configs");
    Ok(())
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn configs_index(State(st): State<AppState>) -> impl IntoResponse {
    Html(configs::index_html(&st.ctx()))
}

async fn download_codex(State(st): State<AppState>) -> impl IntoResponse {
    attachment("codex.toml", "text/toml; charset=utf-8", configs::codex_toml(&st.ctx()))
}

async fn download_grok(State(st): State<AppState>) -> impl IntoResponse {
    attachment("grok.toml", "text/toml; charset=utf-8", configs::grok_toml(&st.ctx()))
}

async fn download_opencode(State(st): State<AppState>) -> impl IntoResponse {
    attachment(
        "opencode.json",
        "application/json; charset=utf-8",
        configs::opencode_json(&st.ctx()),
    )
}

fn attachment(filename: &str, content_type: &str, body: String) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(content_type).unwrap_or(HeaderValue::from_static("text/plain")),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
            .unwrap_or(HeaderValue::from_static("attachment")),
    );
    (headers, body).into_response()
}

async fn proxy(State(st): State<AppState>, req: Request) -> Response {
    match proxy_inner(&st, req).await {
        Ok(resp) => resp,
        Err(err) => {
            warn!(error = %err, "upstream proxy failed");
            (
                StatusCode::BAD_GATEWAY,
                [("content-type", "application/json")],
                serde_json::json!({"error": {"message": err}}).to_string(),
            )
                .into_response()
        }
    }
}

async fn proxy_inner(st: &AppState, req: Request) -> Result<Response, String> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let path_and_query = uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or(uri.path());
    let url = format!("{}{path_and_query}", st.upstream);

    let body = axum::body::to_bytes(req.into_body(), 8 * 1024 * 1024)
        .await
        .map_err(|e| e.to_string())?;

    let mut builder = st
        .client
        .request(to_reqwest_method(&method), &url)
        .body(body);
    for (name, value) in headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()),
            reqwest::header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(n, v);
        }
    }

    let upstream = builder.send().await.map_err(|e| e.to_string())?;
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let upstream_headers = upstream.headers().clone();
    let content_type = upstream_headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let mut out_headers = HeaderMap::new();
    for (name, value) in upstream_headers.iter() {
        if is_hop_by_hop_str(name.as_str()) {
            continue;
        }
        if name == reqwest::header::CONTENT_LENGTH {
            continue;
        }
        if let Ok(ax_name) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(ax_val) = HeaderValue::from_bytes(value.as_bytes()) {
                out_headers.append(ax_name, ax_val);
            }
        }
    }
    out_headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));

    let sse = content_type.contains("text/event-stream");
    tracing::debug!(
        %method,
        path = path_and_query,
        %status,
        sse,
        "proxied"
    );

    if sse {
        let byte_stream = upstream.bytes_stream();
        let mapped = sse_rewrite_stream(byte_stream);
        let mut resp = Response::new(Body::from_stream(mapped));
        *resp.status_mut() = status;
        *resp.headers_mut() = out_headers;
        return Ok(resp);
    }

    let raw = upstream.bytes().await.map_err(|e| e.to_string())?;
    let rewritten = alias_json_bytes(&raw);
    out_headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&rewritten.len().to_string()).unwrap(),
    );
    let mut resp = Response::new(Body::from(rewritten));
    *resp.status_mut() = status;
    *resp.headers_mut() = out_headers;
    Ok(resp)
}

fn sse_rewrite_stream<S, E>(
    stream: S,
) -> impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>>
where
    S: futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    use bytes::Bytes;

    let mut stream = Box::pin(stream);
    let mut pending = String::new();
    let mut finished = false;
    futures_util::stream::poll_fn(move |cx| {
        if finished {
            return std::task::Poll::Ready(None);
        }
        match stream.poll_next_unpin(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                let text = String::from_utf8_lossy(&chunk);
                std::task::Poll::Ready(Some(Ok(Bytes::from(rewrite_sse_chunk(
                    &mut pending, &text,
                )))))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                finished = true;
                std::task::Poll::Ready(Some(Err(std::io::Error::other(e.to_string()))))
            }
            std::task::Poll::Ready(None) => {
                finished = true;
                if pending.is_empty() {
                    std::task::Poll::Ready(None)
                } else {
                    let line = std::mem::take(&mut pending);
                    let line = line.trim_end_matches(['\r', '\n']);
                    std::task::Poll::Ready(Some(Ok(Bytes::from(format!(
                        "{}\n",
                        rewrite_sse_line(line)
                    )))))
                }
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    })
    .filter(|item| {
        futures_util::future::ready(match item {
            Ok(b) => !b.is_empty(),
            Err(_) => true,
        })
    })
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    is_hop_by_hop_str(name.as_str())
}

fn is_hop_by_hop_str(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| name.eq_ignore_ascii_case(h))
}

fn to_reqwest_method(method: &Method) -> reqwest::Method {
    reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        let _ = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    info!("spark-compat-proxy shutting down");
}
