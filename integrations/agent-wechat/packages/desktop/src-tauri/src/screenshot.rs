//! 由系统提供区域选取。截图临时文件只供当前桌面窗口读取。
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionCapture {
    png_base64: String,
}

/// 打开系统区域选取工具。取消时返回空值，截图数据留在本机桌面进程内。
#[tauri::command]
pub async fn desktop_capture_region(
    window: tauri::WebviewWindow,
) -> Result<Option<RegionCapture>, String> {
    if window.label() != "main" {
        return Err("screen_capture_unavailable".into());
    }
    #[cfg(target_os = "macos")]
    {
        macos::capture(&window).await.map_err(str::to_owned)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("screen_capture_unavailable".into())
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::RegionCapture;
    use base64::{engine::general_purpose::STANDARD, Engine};
    use std::{io::ErrorKind, path::Path, process::Stdio, time::Duration};
    use tokio::io::AsyncReadExt;
    use tokio::sync::{oneshot, Mutex};

    // 截图跨 IPC 时会膨胀为 Base64。限制原始 PNG，避免大图耗尽桌面内存。
    const MAX_PNG_BYTES: u64 = 24 * 1024 * 1024;
    const PNG_HEADER: &[u8] = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
    static CAPTURE: Mutex<()> = Mutex::const_new(());

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
        fn CGRequestScreenCaptureAccess() -> bool;
    }

    /// 在用户主动截图后检查屏幕录制权限，并让系统在需要时显示授权对话框。
    async fn capture_access(window: &tauri::WebviewWindow) -> Result<(), &'static str> {
        let (permission, result) = oneshot::channel();
        window
            .run_on_main_thread(move || {
                // 系统函数没有参数，且最低 macOS 版本已包含这两个 API。
                let allowed =
                    unsafe { CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess() };
                let _ = permission.send(allowed);
            })
            .map_err(|_| "screen_capture_failed")?;
        if result.await.map_err(|_| "screen_capture_failed")? {
            Ok(())
        } else {
            Err("screen_capture_permission_denied")
        }
    }

    /// 调用固定的系统程序。在单次选择期间保留互斥锁和私有临时目录。
    pub async fn capture(
        window: &tauri::WebviewWindow,
    ) -> Result<Option<RegionCapture>, &'static str> {
        let _capture = CAPTURE
            .try_lock()
            .map_err(|_| "screen_capture_in_progress")?;
        capture_access(window).await?;
        let temporary = wechat_platform::private_tempdir("agent-wechat-capture-")
            .map_err(|_| "screen_capture_failed")?;
        let path = temporary.path().join("capture.png");
        let mut command = tokio::process::Command::new("/usr/sbin/screencapture");
        command
            .args(["-i", "-s", "-x", "-t", "png"])
            .arg(&path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|_| "screen_capture_failed")?;
        let status = match tokio::time::timeout(Duration::from_secs(120), child.wait()).await {
            Ok(result) => result.map_err(|_| "screen_capture_failed")?,
            Err(_) => {
                child.kill().await.map_err(|_| "screen_capture_failed")?;
                return Ok(None);
            }
        };
        if status.code().is_none() {
            return Err("screen_capture_failed");
        }
        let png = read_capture(&path).await?;
        if !status.success() && png.is_some() {
            return Err("screen_capture_failed");
        }
        Ok(png.map(|png| RegionCapture {
            png_base64: STANDARD.encode(png),
        }))
    }

    /// 退出选择时没有输出文件。读取时检查文件类型和大小，并验证 PNG 文件头。
    async fn read_capture(path: &Path) -> Result<Option<Vec<u8>>, &'static str> {
        let metadata = match tokio::fs::symlink_metadata(path).await {
            Ok(metadata) if metadata.is_file() && !wechat_platform::is_link(&metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            _ => return Err("screen_capture_failed"),
        };
        if metadata.len() > MAX_PNG_BYTES {
            return Err("screen_capture_too_large");
        }
        let file = tokio::fs::File::open(path)
            .await
            .map_err(|_| "screen_capture_failed")?;
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_PNG_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| "screen_capture_failed")?;
        if bytes.len() as u64 > MAX_PNG_BYTES {
            return Err("screen_capture_too_large");
        }
        if bytes.len() < 33 || !bytes.starts_with(PNG_HEADER) {
            return Err("screen_capture_failed");
        }
        Ok(Some(bytes))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn capture_result_is_bounded_and_removed_with_its_private_directory() {
            let temporary = wechat_platform::private_tempdir("agent-wechat-capture-test-").unwrap();
            let path = temporary.path().join("capture.png");
            assert!(wechat_platform::private_path(temporary.path()).unwrap());
            assert_eq!(read_capture(&path).await, Ok(None));
            std::fs::write(&path, b"not a PNG").unwrap();
            assert_eq!(read_capture(&path).await, Err("screen_capture_failed"));
            let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(MAX_PNG_BYTES + 1).unwrap();
            drop(file);
            assert_eq!(read_capture(&path).await, Err("screen_capture_too_large"));
            let png = STANDARD.decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH5sAAAAASUVORK5CYII=").unwrap();
            std::fs::write(&path, &png).unwrap();
            assert_eq!(read_capture(&path).await, Ok(Some(png)));
            drop(temporary);
            assert!(!path.exists());
        }
    }
}
