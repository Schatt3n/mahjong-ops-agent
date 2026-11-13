-- 只索引未完成的同步批次和媒体回写项，避免后台检查扫描已完成历史。
CREATE INDEX apply_intents_unprojected ON apply_intents(stream) WHERE state!='projected';
CREATE INDEX media_assets_pending_writeback ON media_assets(id)
WHERE json_extract(metadata_json,'$.state')='available'
  AND coalesce(json_extract(metadata_json,'$.officialWriteback'),'')!='complete';
PRAGMA user_version=14;
