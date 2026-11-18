//! Short admission barrier, request-owned snapshots, bounded concurrent queries.
use crate::Core;
use anyhow::{Context, Result};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use wechat_profile::{OwnedReaderCache, ReadOptions};

/// Per-request durations contain no account, SQL, content or credential data.
#[derive(Clone, Debug)]
pub struct ReadTimings {
    pub queue: Duration,
    pub account: Duration,
    pub worker: Duration,
    pub snapshot: Duration,
    pub query: Duration,
    pub parallel: bool,
}
impl ReadTimings {
    pub fn server_timing(&self) -> String {
        format!("read_queue;dur={:.2}, account_wait;dur={:.2}, worker_queue;dur={:.2}, snapshot;dur={:.2}, query;dur={:.2}, read_mode;desc=\"{}\"", self.queue.as_secs_f64()*1000.0, self.account.as_secs_f64()*1000.0, self.worker.as_secs_f64()*1000.0, self.snapshot.as_secs_f64()*1000.0, self.query.as_secs_f64()*1000.0, if self.parallel { "parallel" } else { "serialized" })
    }
}

pub(crate) struct ReadPool {
    permits: Arc<tokio::sync::Semaphore>,
    idle: Mutex<Vec<OwnedReaderCache>>,
}
impl ReadPool {
    pub(crate) fn active(&self) -> bool {
        self.permits.available_permits() != 4
    }
}
impl Default for ReadPool {
    fn default() -> Self {
        Self {
            permits: Arc::new(tokio::sync::Semaphore::new(4)),
            idle: Mutex::new(Vec::new()),
        }
    }
}
struct Slot {
    pool: Arc<ReadPool>,
    cache: Option<OwnedReaderCache>,
}
impl Drop for Slot {
    fn drop(&mut self) {
        if let (Some(cache), Ok(mut idle)) = (self.cache.take(), self.pool.idle.lock()) {
            idle.push(cache);
        }
    }
}

/// 读取完成或取消后回收闲置的账号池，避免跨账号的请求积压保留过多连接。
struct TrimReadPools(Arc<Core>);
impl Drop for TrimReadPools {
    fn drop(&mut self) {
        if let Ok(mut pools) = self.0.read_pools.lock() {
            while pools.len() > 4 {
                let idle = pools
                    .iter()
                    .find(|(_, pool)| Arc::strong_count(pool) == 1 && !pool.active())
                    .map(|(account, _)| account.clone());
                match idle {
                    Some(account) => {
                        pools.remove(&account);
                    }
                    None => break,
                }
            }
        }
    }
}

impl Core {
    /// The mutation actor still owns all writes. Only snapshot establishment
    /// excludes it; queries/normalization execute on independent connections.
    pub async fn read<T: Send + 'static>(
        self: &Arc<Self>,
        account: &str,
        query: impl FnOnce(&Core) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        Ok(self.read_timed(account, query).await?.0)
    }

    /// 单独统计读槽排队、账号锁和数据库读取。请求取消时释放尚在等待的读槽。
    pub async fn read_timed<T: Send + 'static>(
        self: &Arc<Self>,
        account: &str,
        query: impl FnOnce(&Core) -> Result<T> + Send + 'static,
    ) -> Result<(T, ReadTimings)> {
        let start = Instant::now();
        self.store.account(account)?;
        let cleanup = TrimReadPools(self.clone());
        let pool = {
            let mut pools = self
                .read_pools
                .lock()
                .map_err(|_| anyhow::anyhow!("read pool unavailable"))?;
            if !pools.contains_key(account) && pools.len() >= 4 {
                let idle = pools
                    .iter()
                    .find(|(_, pool)| Arc::strong_count(pool) == 1 && !pool.active())
                    .map(|(account, _)| account.clone());
                if let Some(idle) = idle {
                    pools.remove(&idle);
                }
            }
            pools
                .entry(account.into())
                .or_insert_with(|| Arc::new(ReadPool::default()))
                .clone()
        };
        let permit = pool.permits.clone().acquire_owned().await?;
        let global = self.read_limit.clone().acquire_owned().await?;
        let admitted = Instant::now();
        // 在请求内等待账号锁；切换聊天后取消请求，不会留下阻塞工作线程继续排队。
        let guard = self.lock_account(account).await?;
        let locked = Instant::now();
        let account = account.to_owned();
        tokio::task::spawn_blocking(move || {
            let working = Instant::now();
            let mut timing = ReadTimings {
                queue: admitted - start,
                account: locked - admitted,
                worker: working - locked,
                snapshot: Duration::ZERO,
                query: Duration::ZERO,
                parallel: false,
            };
            // 回收账号池前先释放读槽和缓存引用。
            let cleanup = cleanup;
            let _permit = permit;
            let _global = global;
            let core = &cleanup.0;
            let mut slot = Slot { pool, cache: None };
            slot.cache = Some(
                slot.pool
                    .idle
                    .lock()
                    .map_err(|_| anyhow::anyhow!("read pool unavailable"))?
                    .pop()
                    .unwrap_or_default(),
            );
            // Offline/non-WAL layouts retain the existing serialized adapter.
            if !core.owns_profile(&account)? {
                let result = query(core);
                timing.query = working.elapsed();
                return result.map(|value| (value, timing));
            }
            let Ok((location, keys)) = core.profile_location_keys(&account) else {
                // Unbound operational media can be read before a profile exists.
                // Keep the old serialized path; profile queries still perform
                // their own key/path admission and return the original error.
                let result = query(core);
                timing.query = working.elapsed();
                return result.map(|value| (value, timing));
            };
            let (reader, parallel) = slot.cache.as_mut().unwrap().read_request_view(
                location,
                keys,
                ReadOptions {
                    stopped_or_consistent_snapshot: true,
                    ..Default::default()
                },
            )?;
            let store = Arc::new(core.store.read_snapshot(&account)?);
            let view = Core {
                store,
                shared: core.shared.clone(),
                snapshot_reader: Some((account, Arc::new(reader))),
                live_store: Some(core.store.clone()),
            };
            let serial_guard = if parallel {
                drop(guard);
                None
            } else {
                Some(guard)
            };
            timing.snapshot = working.elapsed();
            timing.parallel = parallel;
            let querying = Instant::now();
            let result = query(&view);
            timing.query = querying.elapsed();
            drop(view); // End transactions before returning this cache to the pool.
            drop(serial_guard);
            result.map(|value| (value, timing))
        })
        .await
        .context("read worker stopped")?
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::Duration;

    pub(crate) fn fixture() -> (tempfile::TempDir, Arc<Core>) {
        let tmp = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let profile = tmp.path().join("account");
        let password = [42u8; 32];
        let (_, keys, _) = wechat_profile::initialize_profile(&profile, &password).unwrap();
        for relative in keys.databases.keys() {
            let db = rusqlite::Connection::open(profile.join("db_storage").join(relative)).unwrap();
            assert_eq!(
                unsafe { rusqlite::ffi::sqlite3_key(db.handle(), password.as_ptr().cast(), 32) },
                0
            );
            db.execute_batch("PRAGMA cipher_compatibility=4;PRAGMA cipher_page_size=4096;PRAGMA kdf_iter=256000;PRAGMA journal_mode=WAL;").unwrap();
        }
        let store = Arc::new(crate::Store::open(&tmp.path().join("data")).unwrap());
        store.attach("one", &profile, true).unwrap();
        let path = tmp.path().join("keys.json");
        keys.write_new(&path).unwrap();
        store.set_keys_file("one", &path).unwrap();
        let core = Core::new(store, None).unwrap();
        core.store.put_runtime("one", "value", b"old").unwrap();
        (tmp, core)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocked_queries_overlap_and_writer_advances_without_changing_their_boundary() {
        let (_tmp, core) = fixture();
        let mut tasks = Vec::new();
        let mut releases = Vec::new();
        for _ in 0..2 {
            let (started, ready) = tokio::sync::oneshot::channel();
            let (release, resume) = std::sync::mpsc::channel();
            releases.push(release);
            let core = core.clone();
            tasks.push(tokio::spawn(async move {
                core.read("one", move |view| {
                    let boundary = view.store.event_boundary("one")?;
                    started.send(()).unwrap();
                    resume.recv_timeout(Duration::from_secs(10))?;
                    assert!(view.store.is_read_snapshot());
                    assert_eq!(
                        view.store.runtime("one", "value")?.as_deref(),
                        Some(b"old".as_slice())
                    );
                    assert_eq!(view.store.event_boundary("one")?, boundary);
                    Ok(())
                })
                .await
            }));
            tokio::time::timeout(Duration::from_secs(5), ready)
                .await
                .unwrap()
                .unwrap();
        }
        let writer = tokio::time::timeout(Duration::from_secs(2), core.lock_account("one"))
            .await
            .unwrap()
            .unwrap();
        core.store.put_runtime("one", "value", b"new").unwrap();
        core.store
            .append_event("one", "test", &serde_json::json!({}))
            .unwrap();
        drop(writer);
        assert!(core
            .remove_account("one")
            .await
            .unwrap_err()
            .to_string()
            .contains("operation_in_progress"));
        for release in releases {
            release.send(()).unwrap();
        }
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        core.read("one", |view| {
            assert_eq!(
                view.store.runtime("one", "value")?.as_deref(),
                Some(b"new".as_slice())
            );
            assert_eq!(view.store.event_boundary("one")?, "1");
            Ok(())
        })
        .await
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_queries_release_their_snapshots_and_pool_slots() {
        let (_tmp, core) = fixture();
        assert!(core
            .read::<()>("one", |_| anyhow::bail!("synthetic query error"))
            .await
            .is_err());
        for _ in 0..6 {
            core.read("one", |view| {
                view.catalog_page("one", "threads", None, &Default::default())
            })
            .await
            .unwrap();
        }
        assert!(!core.read_pools.lock().unwrap()["one"].active());
        let profile = core.store.account("one").unwrap().profile_path;
        core.remove_account("one").await.unwrap();
        assert!(!core.read_pools.lock().unwrap().contains_key("one"));
        assert!(profile.exists());
    }

    #[tokio::test]
    async fn cancelled_request_releases_read_slots_while_the_account_is_locked() {
        use std::{future::Future, task::Poll};

        let (_tmp, core) = fixture();
        let writer = core.lock_account("one").await.unwrap();
        let available = core.read_limit.available_permits();
        let (ran, mut ran_query) = tokio::sync::oneshot::channel();
        let mut abandoned = Box::pin(core.read("one", move |_| {
            ran.send(()).unwrap();
            Ok(())
        }));
        assert!(
            std::future::poll_fn(|cx| Poll::Ready(abandoned.as_mut().poll(cx).is_pending())).await
        );
        assert_eq!(core.read_limit.available_permits(), available - 1);
        assert!(core.read_pools.lock().unwrap()["one"].active());

        drop(abandoned);
        assert_eq!(core.read_limit.available_permits(), available);
        assert!(!core.read_pools.lock().unwrap()["one"].active());
        assert!(matches!(
            ran_query.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed)
        ));
        drop(writer);

        let result = core
            .read("one", |view| view.store.runtime("one", "value"))
            .await
            .unwrap();
        assert_eq!(result.as_deref(), Some(b"old".as_slice()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exhausted_read_pool_queues_before_the_writer_and_reuses_a_released_slot() {
        let (_tmp, core) = fixture();
        let mut tasks = Vec::new();
        let mut releases = Vec::new();
        for _ in 0..4 {
            let (started, ready) = tokio::sync::oneshot::channel();
            let (release, resume) = std::sync::mpsc::channel();
            releases.push(release);
            let local = core.clone();
            tasks.push(tokio::spawn(async move {
                local
                    .read("one", move |_| {
                        started.send(()).unwrap();
                        resume.recv_timeout(Duration::from_secs(10))?;
                        Ok(())
                    })
                    .await
            }));
            tokio::time::timeout(Duration::from_secs(5), ready)
                .await
                .unwrap()
                .unwrap();
        }
        assert_eq!(
            core.read_pools.lock().unwrap()["one"]
                .permits
                .available_permits(),
            0
        );
        let (started, mut ready) = tokio::sync::oneshot::channel();
        let local = core.clone();
        let fifth = tokio::spawn(async move {
            local
                .read("one", move |_| {
                    started.send(()).unwrap();
                    Ok(())
                })
                .await
        });
        assert!(tokio::time::timeout(Duration::from_millis(100), &mut ready)
            .await
            .is_err());
        let writer = tokio::time::timeout(Duration::from_secs(2), core.lock_account("one"))
            .await
            .unwrap()
            .unwrap();
        drop(writer);
        releases.remove(0).send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), ready)
            .await
            .unwrap()
            .unwrap();
        fifth.await.unwrap().unwrap();
        for release in releases {
            release.send(()).unwrap();
        }
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        assert!(!core.read_pools.lock().unwrap()["one"].active());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rollback_profiles_remain_serialized_without_changing_their_journal_mode() {
        let (_tmp, core) = fixture();
        let (location, keys) = core.profile_location_keys("one").unwrap();
        for relative in keys.databases.keys() {
            let db = rusqlite::Connection::open(location.root().join("db_storage").join(relative))
                .unwrap();
            assert_eq!(
                unsafe { rusqlite::ffi::sqlite3_key(db.handle(), [42u8; 32].as_ptr().cast(), 32) },
                0
            );
            db.execute_batch("PRAGMA cipher_compatibility=4;PRAGMA cipher_page_size=4096;PRAGMA kdf_iter=256000;PRAGMA journal_mode=DELETE;").unwrap();
        }
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, resume) = std::sync::mpsc::channel();
        let local = core.clone();
        let query = tokio::spawn(async move {
            local
                .read_timed("one", move |view| {
                    started.send(()).unwrap();
                    resume.recv_timeout(Duration::from_secs(10))?;
                    view.catalog_page("one", "threads", None, &Default::default())
                })
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), ready)
            .await
            .unwrap()
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), core.lock_account("one"))
                .await
                .is_err()
        );
        release.send(()).unwrap();
        let (_, timing) = query.await.unwrap().unwrap();
        assert!(!timing.parallel);
        assert!(timing
            .server_timing()
            .contains("read_mode;desc=\"serialized\""));
        for relative in keys.databases.keys() {
            let db = rusqlite::Connection::open(location.root().join("db_storage").join(relative))
                .unwrap();
            assert_eq!(
                unsafe { rusqlite::ffi::sqlite3_key(db.handle(), [42u8; 32].as_ptr().cast(), 32) },
                0
            );
            db.execute_batch("PRAGMA cipher_compatibility=4;PRAGMA cipher_page_size=4096;PRAGMA kdf_iter=256000;").unwrap();
            assert_eq!(
                db.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))
                    .unwrap(),
                "delete"
            );
        }
    }
}
