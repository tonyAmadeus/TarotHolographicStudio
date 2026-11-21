#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::fs::OpenOptions;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::State;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Serialize)]
struct BackendInfo {
    url: String,
    backend_dir: String,
    error: Option<String>,
}

struct BackendState {
    info: BackendInfo,
    _child: Arc<Mutex<Option<Child>>>,
}

fn find_bundled_backend() -> Option<(PathBuf, PathBuf)> {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let candidates = [
                exe_dir.join("backend").join("tarot_backend.exe"),
                exe_dir.join("tarot_backend.exe"),
            ];
            for bin in candidates {
                if bin.is_file() {
                    return Some((exe_dir.to_path_buf(), bin));
                }
            }
        }
    }
    None
}

fn spawn_bundled_backend(
    app_root: &Path,
    backend_exe: &Path,
    port: u16,
) -> Result<Child, String> {
    let log_path = std::env::temp_dir().join("tarot_holographic_backend.log");
    let stderr_io = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());

    let mut cmd = Command::new(backend_exe);
    cmd.arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .current_dir(app_root)
        .env("TAROT_APP_ROOT", app_root)
        .env("PYTHONUTF8", "1")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr_io);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    cmd.spawn()
        .map_err(|e| format!("启动内置渲染引擎 `{}` 失败: {}", backend_exe.display(), e))
}

fn find_backend_script() -> Option<(PathBuf, PathBuf)> {
    // 1. Explicit override via environment variable
    if let Ok(env_dir) = std::env::var("TAROT_BACKEND_DIR") {
        let dir = PathBuf::from(env_dir);
        let candidate = dir.join("backend_server.py");
        if candidate.is_file() {
            return Some((dir, candidate));
        }
    }

    // 2. Walk up from executable path (works for standalone .exe and target/release)
    if let Ok(exe_path) = std::env::current_exe() {
        let mut cur = exe_path.parent().map(Path::to_path_buf);
        for _ in 0..7 {
            if let Some(dir) = cur {
                let candidate = dir.join("backend_server.py");
                if candidate.is_file() {
                    return Some((dir, candidate));
                }
                cur = dir.parent().map(Path::to_path_buf);
            } else {
                break;
            }
        }
    }

    // 3. Walk up from current working directory (works for `tauri dev`)
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur: Option<PathBuf> = Some(cwd);
        for _ in 0..5 {
            if let Some(dir) = cur {
                let candidate = dir.join("backend_server.py");
                if candidate.is_file() {
                    return Some((dir, candidate));
                }
                cur = dir.parent().map(Path::to_path_buf);
            } else {
                break;
            }
        }
    }

    None
}

fn pick_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .unwrap_or(18765)
}

fn spawn_python_backend(
    backend_dir: &Path,
    script_path: &Path,
    port: u16,
) -> Result<Child, String> {
    let log_path = std::env::temp_dir().join("tarot_holographic_backend.log");
    let make_stderr = || -> Stdio {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null())
    };

    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
    if let Ok(custom_py) = std::env::var("TAROT_PYTHON") {
        if !custom_py.trim().is_empty() {
            candidates.push((custom_py, Vec::new()));
        }
    }
    candidates.push(("python".to_string(), Vec::new()));
    candidates.push(("py".to_string(), vec!["-3".to_string()]));

    let mut last_err = String::from("未能找到 Python 3 解释器");
    for (bin, extra_args) in candidates {
        let mut cmd = Command::new(&bin);
        for arg in extra_args {
            cmd.arg(arg);
        }
        cmd.arg(script_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .current_dir(backend_dir)
            .env("PYTHONUTF8", "1")
            .env("NO_PROXY", "127.0.0.1,localhost")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(make_stderr());

        #[cfg(target_os = "windows")]
        cmd.creation_flags(CREATE_NO_WINDOW);

        match cmd.spawn() {
            Ok(child) => return Ok(child),
            Err(e) => {
                last_err = format!("启动 `{}` 失败: {}", bin, e);
            }
        }
    }

    Err(last_err)
}

#[tauri::command]
fn get_backend_info(state: State<'_, BackendState>) -> BackendInfo {
    state.info.clone()
}

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("路径为空".to_string());
    }
    let p = Path::new(trimmed);
    let mut cmd = Command::new("explorer");
    if p.is_file() {
        cmd.arg(format!("/select,{}", trimmed));
    } else {
        cmd.arg(trimmed);
    }
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    cmd.spawn().map_err(|e| format!("打开资源管理器失败: {}", e))?;
    Ok(())
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("路径为空".to_string());
    }
    let mut cmd = Command::new("explorer");
    cmd.arg(trimmed);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    cmd.spawn().map_err(|e| format!("打开文件失败: {}", e))?;
    Ok(())
}

fn kill_backend(child_handle: &Arc<Mutex<Option<Child>>>) {
    if let Ok(mut guard) = child_handle.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn main() {
    let port = pick_free_port();
    let url = format!("http://127.0.0.1:{}", port);

    let (backend_dir_str, child_opt, err_opt) = if let Some((app_root, bundled_exe)) = find_bundled_backend() {
        let dir_str = app_root.to_string_lossy().to_string();
        match spawn_bundled_backend(&app_root, &bundled_exe, port) {
            Ok(child) => (dir_str, Some(child), None),
            Err(e) => (dir_str, None, Some(e)),
        }
    } else {
        match find_backend_script() {
            Some((dir, script)) => {
                let dir_str = dir.to_string_lossy().to_string();
                match spawn_python_backend(&dir, &script, port) {
                    Ok(child) => (dir_str, Some(child), None),
                    Err(e) => (dir_str, None, Some(e)),
                }
            }
            None => (
                String::new(),
                None,
                Some("未找到内置渲染后端（backend\\tarot_backend.exe）或 backend_server.py，请完整解压整个程序文件夹后再运行。".to_string()),
            ),
        }
    };

    let child_arc = Arc::new(Mutex::new(child_opt));
    let child_for_exit = Arc::clone(&child_arc);

    let state = BackendState {
        info: BackendInfo {
            url,
            backend_dir: backend_dir_str,
            error: err_opt,
        },
        _child: child_arc,
    };

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            get_backend_info,
            reveal_path,
            open_path
        ])
        .build(tauri::generate_context!())
        .expect("error while building Tarot Holographic Studio");

    app.run(move |_app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            kill_backend(&child_for_exit);
        }
    });
}
