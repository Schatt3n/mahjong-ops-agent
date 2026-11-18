//! Online views retain membership/order independently of mutable official files.
use crate::Core;
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};
use wechat_profile::{Contact, ListQuery, MessageQuery, ProfileReader, SourceMessageId, Thread};

#[derive(Clone, Default)]
pub struct CatalogQuery {
    pub limit: Option<usize>,
    pub cursor: Option<String>,
    pub search: Option<String>,
    pub thread_kind: Option<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    version: u32,
    view: String,
    query: String,
    last: String,
    expires: i64,
}
#[derive(Serialize, Deserialize)]
struct View {
    ids: Vec<String>,
    metadata: Value,
    // Account-wide message continuations retain each peer even when mutable
    // contact/name tables no longer identify it. Older scoped views omit this.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    message_threads: BTreeMap<String, String>,
}
/// 保存一个分页视图的位置，供同一请求的行读取和响应组装共用。
struct PagePosition {
    id: String,
    view: View,
    boundary: String,
    expires: i64,
    start: usize,
}
const MAX_ITEMS: usize = 100_000;
// Large history exports need a bounded window longer than an interactive read.
const VIEW_SECONDS: i64 = 30 * 60;

/// 会话列表和单条会话共用同一份返回字段与已读版本计算。
fn thread_value(account: &str, thread: Thread, generation: &str, version: &str) -> Result<Value> {
    let id = thread.username;
    let kind = if id.ends_with("@chatroom") {
        "group"
    } else if id == "filehelper" || id.starts_with("weixin") {
        "system"
    } else {
        "direct"
    };
    Ok(json!({
        "id": id,
        "peerId": id,
        "accountId": account,
        "kind": kind,
        "displayName": thread.label,
        "unreadCount": thread.unread_count,
        "readStateVersion": Core::read_state_hash(&id, generation, version)?,
        "history": {"coverage":"partial","localCount":null},
        "version": generation,
    }))
}

/// 详情和分页使用同一份联系人名称与备注。
fn contact_value(contact: Contact, generation: &str) -> Value {
    json!({"id":contact.username,"username":contact.username,"displayName":contact.label,"nickname":contact.nickname,"remark":contact.remark,"alias":contact.alias,"relation":"unknown","version":generation})
}

impl Core {
    pub fn group_member(&self, account: &str, group: &str, username: &str) -> Result<Value> {
        wechat_protocol::ordinary_identity(group)?;
        wechat_protocol::ordinary_identity(username)?;
        if !group.ends_with("@chatroom") {
            bail!("invalid_group_identity");
        }
        let reader = self.profile_reader_owned(account)?;
        let own = self.profile_username(account)?;
        let mut member = self
            .catalog_rows(account, "members", Some(group), &reader, own.as_deref())?
            .into_iter()
            .find(|m| m["username"] == username);
        if let Some(member) = member.as_mut() {
            self.classify_peers(std::slice::from_mut(member), &reader)?;
        }
        // A cached roster can be incomplete. Absence is never an authoritative
        // server assertion that this person is not a member.
        Ok(
            json!({"groupId":group,"username":username,"state":if member.is_some(){"present"}else{"unconfirmed"},"member":member,"coverage":"cached"}),
        )
    }
    /// 校验游标并读取它所引用的已保存视图。
    fn pagination_continuation(
        &self,
        account: &str,
        query: &str,
        encoded: &str,
    ) -> Result<PagePosition> {
        if encoded.len() > 8192 {
            bail!("invalid_cursor");
        }
        let decode = || -> Result<Cursor> {
            let bytes = URL_SAFE_NO_PAD.decode(encoded)?;
            let plain = self
                .cursor_vault
                .open_record(account, "pagination-cursor", &bytes)?;
            Ok(serde_json::from_slice(&plain)?)
        };
        let cursor = decode().map_err(|_| anyhow::anyhow!("invalid_cursor"))?;
        if cursor.version != 1 || cursor.query != query {
            bail!("invalid_cursor");
        }
        if cursor.expires <= wechat_store::unix_seconds() {
            bail!("pagination_cursor_expired");
        }
        let retained = self.store.pagination_view(account, &cursor.view)?;
        if retained.query_hash != query || retained.expires_at != cursor.expires {
            bail!("invalid_cursor");
        }
        let plain = self.vault.open_record(
            account,
            &format!("pagination-view:{}", cursor.view),
            &retained.payload,
        )?;
        let view: View = serde_json::from_slice(&plain)?;
        let start = view
            .ids
            .iter()
            .position(|id| id == &cursor.last)
            .context("invalid_cursor")?
            + 1;
        Ok(PagePosition {
            id: cursor.view,
            view,
            boundary: retained.event_cursor,
            expires: cursor.expires,
            start,
        })
    }

    /// 读取指定对象及其补充字段。会话详情直接查主键，使用与列表相同的快照和版本。
    pub fn catalog_item(&self, account: &str, kind: &str, id: &str) -> Result<Value> {
        if !matches!(kind, "threads" | "contacts" | "groups") || id.is_empty() || id.len() > 512 {
            bail!("invalid_request");
        }
        let reader = self.profile_reader_owned(account)?;
        let own = self.profile_username(account)?;
        if kind == "threads" {
            let thread = reader.thread(id)?.context("no rows")?;
            let version = self.store.read_version(account, id)?;
            let mut item = thread_value(account, thread, reader.generation(), &version)?;
            self.thread_summaries(
                account,
                std::slice::from_mut(&mut item),
                &reader,
                own.as_deref(),
            )?;
            return Ok(item);
        }
        let mut item = if kind == "contacts" {
            let contact = reader
                .contacts_by_ids(&[id.to_owned()])?
                .into_iter()
                .find(|contact| !contact.is_group)
                .context("no rows")?;
            contact_value(contact, reader.generation())
        } else {
            self.catalog_rows(account, kind, None, &reader, own.as_deref())?
                .into_iter()
                .find(|row| row["id"] == id)
                .context("no rows")?
        };
        self.classify_peers(std::slice::from_mut(&mut item), &reader)?;
        Ok(item)
    }
    /// Caller holds the account writer or a request-owned profile/control snapshot.
    pub fn catalog_page(
        &self,
        account: &str,
        kind: &str,
        subject: Option<&str>,
        q: &CatalogQuery,
    ) -> Result<Value> {
        if !matches!(
            kind,
            "threads" | "contacts" | "groups" | "members" | "messages"
        ) || q.search.as_ref().is_some_and(|s| s.len() > 256)
            || q.thread_kind
                .as_ref()
                .is_some_and(|k| !matches!(k.as_str(), "direct" | "group" | "system"))
        {
            bail!("invalid_request");
        }
        let query = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                kind,
                subject,
                &q.search,
                &q.thread_kind
            ))?)
        );
        let limit = q.limit.unwrap_or(50);
        if !(1..=200).contains(&limit) {
            bail!("invalid_request");
        }
        let history_position = if kind == "messages" {
            q.cursor
                .as_deref()
                .map(|cursor| self.pagination_continuation(account, &query, cursor))
                .transpose()?
        } else {
            None
        };
        let reader = self.profile_reader_owned(account)?;
        let own = self.profile_username(account)?;
        if let Some(position) = history_position {
            let rows = self.history_continuation_rows(
                account,
                subject,
                limit,
                &reader,
                own.as_deref(),
                &position,
            )?;
            return self.render_page(account, &query, limit, rows, position);
        }
        if kind == "messages" && q.search.is_none() && q.thread_kind.is_none() {
            let (rows, view) =
                self.first_history_rows(account, subject, limit, &reader, own.as_deref())?;
            let position = self.save_first_view(account, &query, limit, view)?;
            return self.render_page(account, &query, limit, rows, position);
        }
        if kind == "contacts" {
            if let Some(cursor) = &q.cursor {
                return self.contact_continuation(account, &query, limit, cursor, &reader);
            }
        }
        let rows = self.catalog_rows(account, kind, subject, &reader, own.as_deref())?;
        let metadata = json!({"sourceGeneration":reader.generation(),"coverage":reader.coverage(),"order":rows.iter().map(|r|r["id"].clone()).collect::<Vec<_>>()});
        let mut page = self.paginate_values(account, &query, q, rows, metadata)?;
        if kind == "threads" {
            self.thread_summaries(
                account,
                page["items"].as_array_mut().context("invalid page")?,
                &reader,
                own.as_deref(),
            )?;
        } else if matches!(kind, "contacts" | "groups" | "members") {
            self.classify_peers(
                page["items"].as_array_mut().context("invalid page")?,
                &reader,
            )?;
        }
        Ok(page)
    }

    /// 续页沿用首屏确定的成员及顺序，只读取本页联系人当前的资料。
    fn contact_continuation(
        &self,
        account: &str,
        query: &str,
        limit: usize,
        cursor: &str,
        reader: &ProfileReader,
    ) -> Result<Value> {
        let position = self.pagination_continuation(account, query, cursor)?;
        let end = (position.start + limit).min(position.view.ids.len());
        let rows = reader
            .contacts_by_ids(&position.view.ids[position.start..end])?
            .into_iter()
            .filter(|contact| !contact.is_group)
            .map(|contact| {
                (
                    contact.username.clone(),
                    contact_value(contact, reader.generation()),
                )
            })
            .collect();
        let mut page = self.render_page(account, query, limit, rows, position)?;
        self.classify_peers(
            page["items"].as_array_mut().context("invalid page")?,
            reader,
        )?;
        Ok(page)
    }

    /// 续页只解码已保存视图中的本页消息，并在账号或会话不匹配时拒绝读取。
    fn history_continuation_rows(
        &self,
        account: &str,
        subject: Option<&str>,
        limit: usize,
        reader: &ProfileReader,
        own: Option<&str>,
        position: &PagePosition,
    ) -> Result<BTreeMap<String, Value>> {
        // A continuation resolves retained IDs against the current generation.
        // New arrivals and renames cannot change the original view membership.
        let ids = position.view.ids.iter().skip(position.start).take(limit);
        let mut peers: BTreeMap<SourceMessageId, &str> = BTreeMap::new();
        let mut sources = Vec::with_capacity(limit);
        let mut tables = BTreeMap::new();
        for id in ids {
            let peer = subject
                .or_else(|| position.view.message_threads.get(id).map(String::as_str))
                .context("invalid_cursor")?;
            let table = tables
                .entry(peer)
                .or_insert_with(|| format!("Msg_{:x}", md5::Md5::digest(peer.as_bytes())));
            let source = crate::source_message_id(id)?;
            if !source.table.eq_ignore_ascii_case(table) {
                bail!("invalid_cursor");
            }
            // A removed database/table/row is a tombstone. A present but
            // unreadable or incompatible source remains an explicit error.
            let Some(database) = reader
                .inventory()
                .iter()
                .find(|d| d.relative_path == source.database)
            else {
                continue;
            };
            if database.state == wechat_profile::DatabaseState::Readable
                && !reader.schemas().iter().any(|s| {
                    s.database == source.database && s.tables.iter().any(|t| t.name == source.table)
                })
            {
                continue;
            }
            peers.insert(source.clone(), peer);
            sources.push(source);
        }
        let mut messages = reader.messages_by_ids(&sources, own)?;
        for message in &mut messages {
            // The retained query and table hash establish the peer even
            // after contact/name metadata changes after page one.
            message.peer = Some((*peers.get(&message.id).context("invalid_cursor")?).to_owned());
        }
        self.history_values(account, messages, own, reader)
    }

    /// 首屏直接保存已排序的历史 ID，只有首屏需要解码正文和补充撤回信息。
    fn first_history_rows(
        &self,
        account: &str,
        subject: Option<&str>,
        limit: usize,
        reader: &ProfileReader,
        own: Option<&str>,
    ) -> Result<(BTreeMap<String, Value>, View)> {
        let references = reader.message_references(subject, MAX_ITEMS).map_err(|e| {
            if matches!(e, wechat_profile::ProfileError::ResourceLimit) {
                anyhow::anyhow!("pagination_capacity_exceeded")
            } else {
                e.into()
            }
        })?;
        let first: BTreeMap<&SourceMessageId, &Option<String>> = references
            .iter()
            .take(limit)
            .map(|reference| (&reference.id, &reference.peer))
            .collect();
        let sources: Vec<_> = references
            .iter()
            .take(limit)
            .map(|r| r.id.clone())
            .collect();
        let mut messages = reader.messages_by_ids(&sources, own)?;
        for message in &mut messages {
            message.peer = (*first
                .get(&message.id)
                .context("history row missing from snapshot")?)
            .clone();
        }
        let rows = self.history_values(account, messages, own, reader)?;
        if rows.len() != sources.len() {
            bail!("history row missing from snapshot");
        }
        let mut view = View {
            ids: Vec::with_capacity(references.len()),
            metadata: json!({"sourceGeneration":reader.generation(),"coverage":reader.coverage()}),
            message_threads: BTreeMap::new(),
        };
        let mut seen = BTreeSet::new();
        for (offset, reference) in references.iter().enumerate() {
            if !seen.insert(&reference.id) {
                continue;
            }
            let id = crate::message_id(&reference.id)?;
            if subject.is_none() {
                let peer = if offset < limit {
                    rows.get(&id).and_then(|row| row["threadId"].as_str())
                } else {
                    reference.peer.as_deref()
                };
                if let Some(peer) = peer {
                    view.message_threads.insert(id.clone(), peer.to_owned());
                }
            }
            view.ids.push(id);
        }
        Ok((rows, view))
    }

    /// 保留单页正文的大小上限。未返回的历史只存 ID，交由视图大小上限约束。
    fn history_values(
        &self,
        account: &str,
        messages: Vec<wechat_profile::Message>,
        own: Option<&str>,
        reader: &ProfileReader,
    ) -> Result<BTreeMap<String, Value>> {
        let mut encoded_bytes = 0usize;
        self.normalize_recalled_batch(account, messages, own, reader)?
            .map(|message| {
                let value = message?;
                encoded_bytes = encoded_bytes.saturating_add(serde_json::to_vec(&value)?.len());
                if encoded_bytes > 32 * 1024 * 1024 {
                    bail!("pagination_capacity_exceeded");
                }
                Ok((
                    value["id"]
                        .as_str()
                        .context("invalid message ID")?
                        .to_owned(),
                    value,
                ))
            })
            .collect()
    }
    /// 构建搜索或目录视图，并共用历史查询的游标与响应组装。
    fn paginate_values(
        &self,
        account: &str,
        query: &str,
        q: &CatalogQuery,
        rows: Vec<Value>,
        metadata: Value,
    ) -> Result<Value> {
        let limit = q.limit.unwrap_or(50);
        if !(1..=200).contains(&limit) {
            bail!("invalid_request");
        }
        let current: BTreeMap<String, Value> = rows
            .into_iter()
            .map(|item| {
                Ok((
                    item["id"]
                        .as_str()
                        .context("resource identity missing")?
                        .to_owned(),
                    item,
                ))
            })
            .collect::<Result<_>>()?;
        let position = if let Some(encoded) = &q.cursor {
            self.pagination_continuation(account, query, encoded)?
        } else {
            // Preserve the source ordering explicitly; the current resource map
            // is used only for field resolution, never as the ordering contract.
            let order = metadata["order"]
                .as_array()
                .context("resource ordering missing")?;
            let mut ids = Vec::new();
            let mut seen = BTreeSet::new();
            for id in order {
                let id = id.as_str().context("resource identity missing")?;
                let Some(item) = current.get(id) else {
                    bail!("resource identity missing");
                };
                let search = q.search.as_ref().map(|s| s.to_lowercase());
                let matches_search = search.as_ref().is_none_or(|s| {
                    ["id", "displayName", "nickname", "remark", "alias", "body"]
                        .iter()
                        .any(|key| {
                            item[key]
                                .as_str()
                                .is_some_and(|v| v.to_lowercase().contains(s))
                        })
                });
                let matches_kind = q
                    .thread_kind
                    .as_ref()
                    .is_none_or(|k| item["kind"].as_str() == Some(k.as_str()));
                if matches_search && matches_kind && seen.insert(id.to_owned()) {
                    ids.push(id.to_owned());
                }
                if ids.len() > MAX_ITEMS {
                    bail!("pagination_capacity_exceeded");
                }
            }
            let mut clean_metadata = metadata;
            clean_metadata
                .as_object_mut()
                .context("invalid page metadata")?
                .remove("order");
            let view = View {
                message_threads: ids
                    .iter()
                    .filter_map(|id| {
                        current
                            .get(id)?
                            .get("threadId")?
                            .as_str()
                            .map(|peer| (id.clone(), peer.to_owned()))
                    })
                    .collect(),
                ids,
                metadata: clean_metadata,
            };
            self.save_first_view(account, query, limit, view)?
        };
        self.render_page(account, query, limit, current, position)
    }

    /// 从读取快照获取事件边界，续页所需的 ID 视图写入现有分页存储。
    fn save_first_view(
        &self,
        account: &str,
        query: &str,
        limit: usize,
        view: View,
    ) -> Result<PagePosition> {
        let id = uuid::Uuid::new_v4().to_string();
        let boundary = self.store.event_boundary(account)?;
        let expires = wechat_store::unix_seconds() + VIEW_SECONDS;
        if view.ids.len() > limit {
            let payload = self.vault.seal(
                account,
                &format!("pagination-view:{id}"),
                &serde_json::to_vec(&view)?,
            )?;
            self.live_store
                .as_ref()
                .unwrap_or(&self.store)
                .save_pagination_view(account, &id, query, &boundary, expires, &payload)?;
        }
        Ok(PagePosition {
            id,
            view,
            boundary,
            expires,
            start: 0,
        })
    }

    /// 按原视图的顺序组装已读取的行。删除的旧记录保留占位，避免翻页跳过或重复。
    fn render_page(
        &self,
        account: &str,
        query: &str,
        limit: usize,
        mut current: BTreeMap<String, Value>,
        position: PagePosition,
    ) -> Result<Value> {
        let PagePosition {
            id,
            view,
            boundary,
            expires,
            start,
        } = position;
        let end = start.saturating_add(limit).min(view.ids.len());
        let items: Vec<_> = view.ids[start..end]
            .iter()
            .map(|id| {
                current.remove(id).unwrap_or_else(
                    || json!({"id":id,"deleted":true,"version":metadata_version(&view.metadata)}),
                )
            })
            .collect();
        let next = if end < view.ids.len() {
            let cursor = Cursor {
                version: 1,
                view: id,
                query: query.into(),
                last: view.ids[end - 1].clone(),
                expires,
            };
            Some(URL_SAFE_NO_PAD.encode(self.cursor_vault.seal(
                account,
                "pagination-cursor",
                &serde_json::to_vec(&cursor)?,
            )?))
        } else {
            None
        };
        let mut result = view.metadata;
        let object = result.as_object_mut().context("invalid page metadata")?;
        object.insert("items".into(), Value::Array(items));
        object.insert("nextCursor".into(), json!(next));
        object.insert("asOfEventCursor".into(), json!(boundary));
        object.insert("viewExpiresAt".into(), json!(expires));
        Ok(result)
    }
    /// 读取目录和需要匹配正文的搜索结果。
    fn catalog_rows(
        &self,
        account: &str,
        kind: &str,
        subject: Option<&str>,
        reader: &ProfileReader,
        own: Option<&str>,
    ) -> Result<Vec<Value>> {
        let mut result = Vec::new();
        let mut cursor = None;
        let mut encoded_bytes = 0usize;
        let mut thread_order = BTreeMap::new();
        loop {
            let q = ListQuery {
                limit: Some(200),
                cursor: cursor.clone(),
                search: None,
            };
            let (items, next): (Vec<Value>, Option<String>) = match kind {
                "threads" => {
                    let page = reader.threads(&q)?;
                    let ids: Vec<_> = page
                        .items
                        .iter()
                        .map(|thread| thread.username.clone())
                        .collect();
                    let versions = self.store.read_versions(account, &ids)?;
                    let pinned = reader.thread_pinned(&ids).unwrap_or_default();
                    for thread in &page.items {
                        thread_order.insert(
                            thread.username.clone(),
                            (
                                pinned
                                    .get(&thread.username)
                                    .copied()
                                    .flatten()
                                    .unwrap_or(false),
                                thread
                                    .last_timestamp
                                    .unwrap_or(0)
                                    .max(thread.sort_timestamp.unwrap_or(0)),
                            ),
                        );
                    }
                    let items = page
                        .items
                        .into_iter()
                        .map(|thread| {
                            let version = versions
                                .get(&thread.username)
                                .map(String::as_str)
                                .unwrap_or("0");
                            thread_value(account, thread, reader.generation(), version)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    (items, page.next_cursor)
                }
                "contacts" => {
                    let page = reader.contacts(&q)?;
                    (
                        page.items
                            .into_iter()
                            .filter(|c| !c.is_group)
                            .map(|c| contact_value(c, reader.generation()))
                            .collect(),
                        page.next_cursor,
                    )
                }
                "groups" => {
                    let page = reader.groups(&q)?;
                    (page.items.into_iter().map(|g|json!({"id":g.username,"roomId":g.username,"displayName":g.label,"ownerId":g.owner,"announcement":g.announcement,"version":reader.generation()})).collect(),page.next_cursor)
                }
                "members" => {
                    let page = reader.group_members(subject.context("room required")?, &q)?;
                    (page.items.into_iter().map(|m|json!({"id":m.username,"username":m.username,"displayName":m.label,"roomDisplayName":m.room_display_name,"mentionName":crate::mention_send::mention_name(&m),"version":reader.generation()})).collect(),page.next_cursor)
                }
                "messages" => {
                    let page = reader.messages(&MessageQuery {
                        chat: subject.map(str::to_owned),
                        own_username: own.map(str::to_owned),
                        limit: Some(200),
                        cursor: cursor.clone(),
                        ..Default::default()
                    })?;
                    (
                        self.normalize_recalled_batch(account, page.items, own, reader)?
                            .collect::<Result<Vec<_>>>()?,
                        page.next_cursor,
                    )
                }
                _ => bail!("invalid_request"),
            };
            for item in items {
                encoded_bytes = encoded_bytes.saturating_add(serde_json::to_vec(&item)?.len());
                if encoded_bytes > 32 * 1024 * 1024 {
                    bail!("pagination_capacity_exceeded");
                }
                result.push(item);
            }
            if result.len() > MAX_ITEMS {
                bail!("pagination_capacity_exceeded");
            }
            match next {
                Some(next) if cursor.as_ref() != Some(&next) => cursor = Some(next),
                Some(_) => bail!("invalid profile continuation"),
                None => break,
            }
        }
        if kind == "threads" {
            // Read every canonical session page before selecting the public
            // page: an older pinned thread can occur after the reader's first
            // 200 rows. Preview enrichment still runs only for returned rows.
            result.sort_by_cached_key(|row| {
                let id = row["id"].as_str().unwrap_or_default();
                let (pinned, timestamp) = thread_order.get(id).copied().unwrap_or_default();
                (Reverse(pinned), Reverse(timestamp), id.to_owned())
            });
        }
        Ok(result)
    }
}
fn metadata_version(metadata: &Value) -> Value {
    metadata
        .get("sourceGeneration")
        .cloned()
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    fn rows(ids: &[(&str, &str)]) -> Vec<Value> {
        ids.iter()
            .map(|(id, name)| json!({"id":id,"displayName":name}))
            .collect()
    }
    fn metadata(ids: &[&str]) -> Value {
        json!({"sourceGeneration":"test","coverage":"partial","order":ids})
    }
    #[test]
    fn paging_retains_membership_order_updates_and_tombstones_across_restart() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("store");
        let core = Core::new(Arc::new(wechat_store::Store::open(&path).unwrap()), None).unwrap();
        core.create_account("a").unwrap();
        let first = core
            .paginate_values(
                "a",
                "query",
                &CatalogQuery {
                    limit: Some(1),
                    search: Some("match".into()),
                    ..Default::default()
                },
                rows(&[("a", "match a"), ("b", "match b"), ("c", "match c")]),
                metadata(&["a", "b", "c"]),
            )
            .unwrap();
        let cursor = first["nextCursor"].as_str().unwrap().to_owned();
        drop(core);
        let core = Core::new(Arc::new(wechat_store::Store::open(&path).unwrap()), None).unwrap();
        let second = core
            .paginate_values(
                "a",
                "query",
                &CatalogQuery {
                    limit: Some(2),
                    cursor: Some(cursor.clone()),
                    search: Some("match".into()),
                    ..Default::default()
                },
                rows(&[("d", "match late old timestamp"), ("b", "renamed")]),
                metadata(&["d", "b"]),
            )
            .unwrap();
        assert_eq!(second["items"][0]["id"], "b");
        assert_eq!(second["items"][0]["displayName"], "renamed");
        assert_eq!(second["items"][1]["id"], "c");
        assert_eq!(second["items"][1]["deleted"], true);
        assert!(second["nextCursor"].is_null());
        assert!(core
            .paginate_values(
                "a",
                "other-query",
                &CatalogQuery {
                    cursor: Some(cursor),
                    ..Default::default()
                },
                rows(&[]),
                metadata(&[])
            )
            .unwrap_err()
            .to_string()
            .contains("invalid_cursor"));
    }
    #[test]
    fn thread_order_uses_saved_pins_and_timestamps_before_paging_and_retains_old_views() {
        let temp = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let profile = temp.path().join("fixture");
        let password = [42u8; 32];
        let (_, keys, _) = wechat_profile::initialize_profile(&profile, &password).unwrap();
        let open = |relative: &str| {
            let db = rusqlite::Connection::open(profile.join("db_storage").join(relative)).unwrap();
            assert_eq!(
                unsafe { rusqlite::ffi::sqlite3_key(db.handle(), password.as_ptr().cast(), 32) },
                0
            );
            db.execute_batch("PRAGMA cipher_compatibility=4;PRAGMA cipher_page_size=4096;PRAGMA kdf_iter=256000;").unwrap();
            db
        };
        let contact = open("contact/contact.db");
        let session = open("session/session.db");
        for index in 0..205 {
            let peer = format!("peer{index:03}");
            contact
                .execute(
                    "INSERT INTO contact(username,local_type,flag,delete_flag) VALUES(?1,1,?2,0)",
                    rusqlite::params![peer, if index >= 203 { 0x801 } else { 1 }],
                )
                .unwrap();
            let sort = if index == 204 { 1 } else { 1000 - index };
            let last = match index {
                203 => 1002,
                202 => 2000,
                _ => sort,
            };
            session.execute(
                "INSERT INTO SessionTable(username,sort_timestamp,last_timestamp) VALUES(?1,?2,?3)",
                rusqlite::params![peer, sort, last],
            ).unwrap();
        }
        drop(contact);
        drop(session);
        let store = Arc::new(wechat_store::Store::open(&temp.path().join("data")).unwrap());
        store.attach("a", &profile, true).unwrap();
        let key_path = temp.path().join("keys.json");
        keys.write_new(&key_path).unwrap();
        store.set_keys_file("a", &key_path).unwrap();
        let core = Core::new(store, None).unwrap();
        let contacts = CatalogQuery {
            limit: Some(1),
            ..Default::default()
        };
        let first_contact = core.catalog_page("a", "contacts", None, &contacts).unwrap();
        let contact_cursor = first_contact["nextCursor"].as_str().unwrap().to_owned();
        let contact_db = open("contact/contact.db");
        contact_db
            .execute(
                "UPDATE contact SET remark='New remark' WHERE username='peer001'",
                [],
            )
            .unwrap();
        contact_db
            .execute("DELETE FROM contact WHERE username='peer002'", [])
            .unwrap();
        drop(contact_db);
        let next_contacts = core
            .catalog_page(
                "a",
                "contacts",
                None,
                &CatalogQuery {
                    limit: Some(2),
                    cursor: Some(contact_cursor),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(next_contacts["items"][0]["id"], "peer001");
        assert_eq!(next_contacts["items"][0]["displayName"], "New remark");
        assert_eq!(next_contacts["items"][1]["id"], "peer002");
        assert_eq!(next_contacts["items"][1]["deleted"], true);
        let first = core
            .catalog_page(
                "a",
                "threads",
                None,
                &CatalogQuery {
                    limit: Some(2),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(first["items"][0]["id"], "peer203");
        assert_eq!(first["items"][1]["id"], "peer204");
        let cursor = first["nextCursor"].as_str().unwrap().to_owned();
        let old_order = first["order"].clone();
        // Unpinning changes the canonical sort timestamp. New messages can
        // have a newer last timestamp than their saved sort timestamp.
        let contact = open("contact/contact.db");
        contact
            .execute("UPDATE contact SET flag=1 WHERE username='peer203'", [])
            .unwrap();
        drop(contact);
        let session = open("session/session.db");
        session
            .execute(
                "UPDATE SessionTable SET sort_timestamp=2500 WHERE username='peer203'",
                [],
            )
            .unwrap();
        session.execute("UPDATE SessionTable SET last_timestamp=3000 WHERE username IN ('peer200','peer201')", []).unwrap();
        drop(session);
        let fresh = core
            .catalog_page(
                "a",
                "threads",
                None,
                &CatalogQuery {
                    limit: Some(5),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            fresh["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["peer204", "peer200", "peer201", "peer203", "peer202"]
        );
        assert!(
            fresh["items"]
                .as_array()
                .unwrap()
                .iter()
                .all(|item| item.get("sortTimestamp").is_none()
                    && item.get("lastTimestamp").is_none())
        );
        let later = core
            .catalog_page(
                "a",
                "threads",
                None,
                &CatalogQuery {
                    limit: Some(2),
                    cursor: Some(cursor),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(later["order"], old_order);
        assert_eq!(later["items"][0]["id"], "peer202");
        assert_eq!(later["items"][1]["id"], "peer000");
    }

    #[test]
    fn account_message_search_retains_cross_thread_membership_and_peers() {
        use md5::Digest;
        let temp = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let profile = temp.path().join("fixture");
        let password = [42u8; 32];
        let (_, keys, _) = wechat_profile::initialize_profile(&profile, &password).unwrap();
        let database = profile.join("db_storage/message/message_0.db");
        let db = rusqlite::Connection::open(&database).unwrap();
        assert_eq!(
            unsafe { rusqlite::ffi::sqlite3_key(db.handle(), password.as_ptr().cast(), 32) },
            0
        );
        db.execute_batch(
            "PRAGMA cipher_compatibility=4;PRAGMA cipher_page_size=4096;PRAGMA kdf_iter=256000;",
        )
        .unwrap();
        for (index, peer) in [(1, "first-peer"), (2, "second@chatroom")] {
            let table = format!("Msg_{:x}", md5::Md5::digest(peer.as_bytes()));
            db.execute(
                "INSERT INTO Name2Id(rowid,user_name,is_session)VALUES(?1,?2,1)",
                rusqlite::params![index, peer],
            )
            .unwrap();
            db.execute_batch(&format!("CREATE TABLE {table}(local_id INTEGER PRIMARY KEY,server_id INTEGER,local_type INTEGER,create_time INTEGER,real_sender_id INTEGER,message_content TEXT,source TEXT,WCDB_CT_message_content INTEGER,WCDB_CT_source INTEGER);")).unwrap();
            db.execute(
                &format!("INSERT INTO {table} VALUES(1,?1,1,?1,1,'match original','',0,0)"),
                [index * 2],
            )
            .unwrap();
        }
        let first_table = format!("Msg_{:x}", md5::Md5::digest(b"first-peer"));
        db.execute_batch(&format!(
            "INSERT INTO {first_table} VALUES(2,3,1,3,1,'match middle','',0,0);"
        ))
        .unwrap();
        let store = Arc::new(wechat_store::Store::open(&temp.path().join("data")).unwrap());
        store.attach("a", &profile, true).unwrap();
        let key_path = temp.path().join("keys.json");
        keys.write_new(&key_path).unwrap();
        store.set_keys_file("a", &key_path).unwrap();
        let core = Core::new(store, None).unwrap();
        core.create_account("b").unwrap();
        let first = core
            .catalog_page(
                "a",
                "messages",
                None,
                &CatalogQuery {
                    limit: Some(1),
                    search: Some("match".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(first["items"][0]["threadId"], "second@chatroom");
        assert!(first.get("message_threads").is_none());
        let cursor = first["nextCursor"].as_str().unwrap().to_owned();
        // Retained peers survive name-map removal and a search-matching edit.
        db.execute_batch(&format!("UPDATE {first_table} SET message_content='edited' WHERE local_id=2;DELETE FROM {first_table} WHERE local_id=1;DELETE FROM Name2Id;INSERT INTO {first_table} VALUES(3,99,NULL,99,1,'late malformed row','',0,0);")).unwrap();
        let next = core
            .catalog_page(
                "a",
                "messages",
                None,
                &CatalogQuery {
                    limit: Some(2),
                    search: Some("match".into()),
                    cursor: Some(cursor.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(next["items"][0]["threadId"], "first-peer");
        assert_eq!(next["items"][0]["body"], "edited");
        assert_eq!(next["items"][1]["deleted"], true);
        assert!(next["nextCursor"].is_null());
        for (account, subject, search) in [
            ("a", Some("first-peer"), "match"),
            ("a", None, "changed"),
            ("b", None, "match"),
        ] {
            assert!(core
                .catalog_page(
                    account,
                    "messages",
                    subject,
                    &CatalogQuery {
                        cursor: Some(cursor.clone()),
                        search: Some(search.into()),
                        ..Default::default()
                    }
                )
                .unwrap_err()
                .to_string()
                .contains("invalid_cursor"));
        }
    }

    #[test]
    fn message_continuation_only_resolves_retained_page_rows() {
        use md5::Digest;
        let temp = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let profile = temp.path().join("fixture");
        let password = [42u8; 32];
        let (_, keys, _) = wechat_profile::initialize_profile(&profile, &password).unwrap();
        let table = format!("Msg_{:x}", md5::Md5::digest(b"peer"));
        let database = profile.join("db_storage/message/message_0.db");
        let mut db = rusqlite::Connection::open(&database).unwrap();
        assert_eq!(
            unsafe { rusqlite::ffi::sqlite3_key(db.handle(), password.as_ptr().cast(), 32) },
            0
        );
        db.execute_batch(&format!("PRAGMA cipher_compatibility=4;PRAGMA cipher_page_size=4096;PRAGMA kdf_iter=256000;INSERT INTO Name2Id(user_name,is_session)VALUES('peer',1);CREATE TABLE {table}(local_id INTEGER PRIMARY KEY,server_id INTEGER,local_type INTEGER,create_time INTEGER,real_sender_id INTEGER,message_content TEXT,source TEXT,WCDB_CT_message_content INTEGER,WCDB_CT_source INTEGER);")).unwrap();
        let tx = db.transaction().unwrap();
        for local in 1..=2048 {
            tx.execute(
                &format!("INSERT INTO {table} VALUES(?1,?1,1,?1,1,'original','',0,0)"),
                [local],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let store =
            std::sync::Arc::new(wechat_store::Store::open(&temp.path().join("data")).unwrap());
        store.attach("a", &profile, true).unwrap();
        let key_path = temp.path().join("keys.json");
        keys.write_new(&key_path).unwrap();
        store.set_keys_file("a", &key_path).unwrap();
        let core = Core::new(store, None).unwrap();
        let first = core
            .catalog_page(
                "a",
                "messages",
                Some("peer"),
                &CatalogQuery {
                    limit: Some(1),
                    ..Default::default()
                },
            )
            .unwrap();
        let cursor = first["nextCursor"].as_str().unwrap().to_owned();
        let query = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&("messages", Some("peer"), None::<String>, None::<String>))
                    .unwrap()
            )
        );
        let position = core.pagination_continuation("a", &query, &cursor).unwrap();
        assert_eq!(position.view.ids.len(), 2048);
        assert_eq!(position.start, 1);
        let retained = core.store.pagination_view("a", &position.id).unwrap();
        let raw = core
            .vault
            .open_record(
                "a",
                &format!("pagination-view:{}", position.id),
                &retained.payload,
            )
            .unwrap();
        let decoded: Value = serde_json::from_slice(&raw).unwrap();
        assert!(decoded.get("message_threads").is_none());
        let mut previous_view = decoded;
        previous_view["message_threads"] = serde_json::to_value(
            position
                .view
                .ids
                .iter()
                .map(|id| (id, "peer"))
                .collect::<BTreeMap<_, _>>(),
        )
        .unwrap();
        let previous_size = serde_json::to_vec(&previous_view).unwrap().len();
        assert!(raw.len() < previous_size / 2);
        eprintln!(
            "Scoped history view with 2048 message IDs: {} bytes (previous shape {previous_size})",
            raw.len()
        );
        // Edits and deletions in the original membership must be reflected;
        // unrelated late rows must never be decoded for this continuation.
        db.execute_batch(&format!("UPDATE {table} SET message_content='edited' WHERE local_id=2047;DELETE FROM {table} WHERE local_id=2046;INSERT INTO {table} VALUES(3000,3000,NULL,0,1,'late malformed row','',0,0);")).unwrap();
        let next = core
            .catalog_page(
                "a",
                "messages",
                Some("peer"),
                &CatalogQuery {
                    limit: Some(2),
                    cursor: Some(cursor.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(next["items"][0]["body"], "edited");
        assert_eq!(next["items"][0]["threadId"], "peer");
        assert_eq!(next["items"][1]["deleted"], true);
        let missing = crate::message_id(&wechat_profile::SourceMessageId {
            database: "message/message_0.db".into(),
            table: table.clone(),
            local_id: "2046".into(),
        })
        .unwrap();
        assert_eq!(next["items"][1]["id"], missing);
        assert_eq!(next["items"].as_array().unwrap().len(), 2);
        assert!(core
            .catalog_page(
                "a",
                "messages",
                Some("another_peer"),
                &CatalogQuery {
                    cursor: Some(cursor),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("invalid_cursor"));
        db.execute("DELETE FROM Name2Id WHERE user_name='peer'", [])
            .unwrap();
        let later = core
            .catalog_page(
                "a",
                "messages",
                Some("peer"),
                &CatalogQuery {
                    limit: Some(1),
                    cursor: Some(next["nextCursor"].as_str().unwrap().into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(later["items"][0]["threadId"], "peer");
        let mut unavailable = keys.clone();
        let relative = "message/message_0.db";
        let salt = unavailable.databases[relative].salt;
        unavailable.databases.insert(
            relative.into(),
            wechat_profile::DatabaseKey::new(relative.into(), salt, [7; 32]).unwrap(),
        );
        let unavailable_path = temp.path().join("unavailable-keys.json");
        unavailable.write_new(&unavailable_path).unwrap();
        core.store.set_keys_file("a", &unavailable_path).unwrap();
        let error = core
            .catalog_page(
                "a",
                "messages",
                Some("peer"),
                &CatalogQuery {
                    cursor: Some(next["nextCursor"].as_str().unwrap().into()),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<wechat_profile::ProfileError>(),
            Some(wechat_profile::ProfileError::KeyMismatch(_))
        ));
    }

    #[test]
    fn cursor_rejects_tampering_and_cross_account_use() {
        let temp = tempfile::tempdir().unwrap();
        let core = Core::new(
            Arc::new(wechat_store::Store::open(&temp.path().join("store")).unwrap()),
            None,
        )
        .unwrap();
        core.create_account("a").unwrap();
        core.create_account("b").unwrap();
        let page = core
            .paginate_values(
                "a",
                "query",
                &CatalogQuery {
                    limit: Some(1),
                    ..Default::default()
                },
                rows(&[("a", "a"), ("b", "b")]),
                metadata(&["a", "b"]),
            )
            .unwrap();
        let cursor = page["nextCursor"].as_str().unwrap().to_owned();
        assert!(core
            .paginate_values(
                "b",
                "query",
                &CatalogQuery {
                    cursor: Some(cursor.clone()),
                    ..Default::default()
                },
                rows(&[]),
                metadata(&[])
            )
            .is_err());
        let mut bytes = URL_SAFE_NO_PAD.decode(cursor).unwrap();
        bytes[5] ^= 1;
        assert!(core
            .paginate_values(
                "a",
                "query",
                &CatalogQuery {
                    cursor: Some(URL_SAFE_NO_PAD.encode(bytes)),
                    ..Default::default()
                },
                rows(&[]),
                metadata(&[])
            )
            .is_err());
    }
}
