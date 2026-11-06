"""装配有限话题上下文并持久记录分类；不创建业务目标、牌局或任何客户回复。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from .models import GroupMessage
from .projections import public_group_game_summary
from .semantic_topic_router import SemanticTopicClassifier
from .topic_cache import load_group_topic_candidates
from .topic_models import PUBLIC_TOPIC_FIELDS, validate_public_topic_summary


class TopicRoutingService:
    """以 Store 为事实源连接语义分类器；同源重投、版本竞争和缓存故障均显式处理。"""

    def __init__(self, store, classifier, cache=None, *, idle_seconds: int = 1800):
        """缓存仅用于公开卡片；个人目标与原文只从 Store 按本人身份读取。"""
        self.store = store
        self.classifier = classifier
        self.cache = cache
        self.idle_seconds = idle_seconds

    def classify(self, message: GroupMessage, *, trace_id: str) -> dict:
        """返回不可变来源回执；模型失败不伪装闲聊，CAS 最多重新装配并分类一次。"""
        identity = self.store.get_channel_identity(message.channel,message.sender_external_id)
        customer_id = identity.customer_id if identity is not None else f'{message.channel}:{message.sender_external_id}'
        conflicts = []
        for attempt in range(2):
            recorded = self.store.get_group_topic_message(message.channel,message.room_id,customer_id,message.message_id)
            if recorded is not None:
                return self._replay(message,recorded)
            candidates,own_context,selection_audit = self._load_context(message,customer_id)
            board,board_source = self._load_board(message)
            selection_audit['board_source'] = board_source
            classification = self.classifier.classify(message=message,candidates=candidates,own_context=own_context,
                board=board,
                trace_id=trace_id if attempt==0 else f'{trace_id}_topic_cas_retry')
            classification = deepcopy(classification)
            classification['routing_audit'] = {**selection_audit,'cas_conflicts':deepcopy(conflicts)}
            topic_id = classification.get('topic_id') if classification['topic_action']=='existing' else None
            selected = next((row for row in candidates if row['topic_id']==topic_id),None)
            if topic_id is not None and selected is None:
                raise ValueError('topic selection is not in loaded candidates')
            create_topic = classification['topic_action']=='new' and classification['intent'] not in {'chitchat','uncertain'}
            try:
                receipt = self.store.record_group_topic_message(
                    channel=message.channel,room_id=message.room_id,customer_id=customer_id,
                    message_id=message.message_id,text=message.text,sent_at=message.sent_at,
                    intent=classification['intent'],topic_id=topic_id,
                    public_summary=classification['public_summary'],classification=classification,
                    expected_version=selected['version'] if selected is not None else None,
                    trace_id=trace_id,idle_seconds=self.idle_seconds,create_topic=create_topic)
            except ValueError as exc:
                # 仅已知版本冲突可重做分类；身份、来源冲突、合同和数据库错误不能吞掉。
                if str(exc) != 'topic version conflict' or attempt:
                    raise
                conflicts.append({'topic_id':topic_id,'expected_version':selected['version'],
                                  'classification':classification})
                continue
            if self.cache is not None and receipt['topic'] is not None:
                try:
                    self.cache.put(receipt['topic'])
                except Exception:
                    # 已落库记录不受可重建缓存影响；下一次读取仍会从事实索引补齐。
                    pass
            return receipt
        raise RuntimeError('unreachable topic routing retry state')

    def _replay(self, message: GroupMessage, recorded: dict) -> dict:
        """只复用完全相同的来源；同消息号换原文或时间必须拒绝，不能掩盖事件冲突。"""
        if recorded['text'] != message.text or datetime.fromisoformat(recorded['sent_at']) != message.sent_at:
            raise ValueError('topic message id conflict')
        topic = self.store.get_group_topic(recorded['topic_id']) if recorded['topic_id'] else None
        return {'topic':topic,'message':deepcopy(recorded),'deduplicated':True}

    def _load_board(self, message: GroupMessage) -> tuple[list,dict]:
        """编号取已发布不可变版本，现状取权威 Game；未知引用绝不套用最新版编号。"""
        snapshot = None
        if message.quoted_message_id:
            status = 'quoted'
            snapshot = self.store.get_board_snapshot_by_message_id(message.room_id,message.quoted_message_id)
            if snapshot is None:
                reference = self.store.resolve_message_reference(
                    conversation_id=f'wechaty:room:{message.room_id}',message_id=message.quoted_message_id)
                if reference is not None and reference.business_ref_type=='group_board_snapshot':
                    candidate = self.store.get_board_snapshot(reference.business_ref_id)
                    if candidate is not None and candidate.room_id==message.room_id:
                        snapshot = candidate
            if snapshot is None:
                return [],{'reference_status':'unresolved_quote','snapshot_id':None}
        else:
            status = 'latest'
            snapshot = self.store.get_latest_board_snapshot(message.room_id)
        if snapshot is None:
            return [],{'reference_status':'unavailable','snapshot_id':None}
        rows = []
        for item in snapshot.items:
            row = {'game_id':item.game_id,'display_no':item.item_no,
                   'snapshot_id':snapshot.snapshot_id,'board_message_id':snapshot.external_message_id,
                   'reference_status':status}
            try:
                game = self.store.require_game(item.game_id)
            except (ValueError,KeyError):
                row['status'] = 'missing'
                rows.append(row)
                continue
            public = public_group_game_summary(game)
            requirement = public['public_requirement']
            stake = requirement.get('stake',requirement.get('base_stake',requirement.get('stake_label')))
            if type(stake) in (int,float):
                stake = str(stake)
            row.update(game_type=requirement.get('requested_game',requirement.get('game_type')),
                stakes=stake,smoking=requirement.get('smoke_preference'),time=public['planned_start_at'],
                slots_filled=public['claimed_seats'],slots_total=public['seats_total'],status=public['status'])
            # 再做强类型投影，Game requirement 中即使混有内部字段也不能进入公共分类上下文。
            rows.append(SemanticTopicClassifier.project_board_item(row))
        return rows,{'reference_status':status,'snapshot_id':snapshot.snapshot_id,
                     'board_message_id':snapshot.external_message_id}

    def _load_context(self, message: GroupMessage, customer_id: str) -> tuple[list,dict,dict]:
        """先保留精确引用与本人冷目标，再补公共热话题；从不读取共享群原文。"""
        cache_errors = []
        def load(*, own: bool):
            """缓存失败后重读事实；数据库失败仍向上传播，不转成空候选。"""
            args = {'room_id':message.room_id,'customer_id':customer_id if own else None,
                    'at':message.sent_at,'limit':5,'include_cold':own}
            try:
                return load_group_topic_candidates(self.store,self.cache,**args)
            except Exception as exc:
                if self.cache is None:
                    raise
                cache_errors.append(type(exc).__name__)
                return self.store.list_group_topics(**args)
        own_topics = load(own=True)
        active = load(own=False)
        quoted = self.store.get_group_topic_for_message_reference(message.channel,message.room_id,message.quoted_message_id) if message.quoted_message_id else None
        rows = ([quoted] if quoted else []) + own_topics + active
        by_id = {}
        for row in rows:
            if row['room_id']==message.room_id and row['topic_id'] not in by_id:
                by_id[row['topic_id']] = deepcopy(row)
        max_candidates = max(1,min(12,self.classifier.max_candidates))
        selected = list(by_id.values())[:max_candidates]
        own_messages = []
        tasks = []
        for row in selected:
            row['quoted_reference'] = bool(quoted and row['topic_id']==quoted['topic_id'])
            messages = self.store.list_group_topic_customer_messages(row['topic_id'],customer_id,limit=6)
            # 防御 Store 实现错误：本人条件和原文都必须再次核对稳定客户身份。
            messages = [item for item in messages if item['customer_id']==customer_id]
            own_messages.extend(messages)
            bindings = self.store.list_group_topic_customer_bindings(row['topic_id'],customer_id)
            topic_tasks = self._project_tasks(message,row['topic_id'],customer_id,bindings,messages)
            tasks.extend(topic_tasks)
            if len(topic_tasks)==1:
                row['own_task_context_id'] = topic_tasks[0]['task_context_id']
                row['own_conversation_id'] = topic_tasks[0]['conversation_id']
            own_game_ids = {game_id for binding in bindings if binding['customer_id']==customer_id
                            for game_id in binding.get('game_ids',())}
            row['own_game_ids'] = sorted(own_game_ids)[:8]
        own_messages.sort(key=lambda row:(row['sent_at'],row['message_id']))
        # 同条来源可被多条任务关联；只携带一份本人原文并明确检索的省略数量。
        unique_messages = {row['source_key']:row for row in own_messages}
        omitted = max(0,len(by_id)-len(selected))
        own_context = {'customer_id':customer_id,'tasks':tasks,'recent_messages':list(unique_messages.values()),
                       'candidate_count_omitted':omitted}
        audit = {'candidate_count_omitted':omitted,'own_candidate_count':len(own_topics),
                 'active_candidate_count':len(active),'quoted_topic_id':quoted['topic_id'] if quoted else None,
                 'cache_errors':cache_errors}
        return selected,own_context,audit

    def _project_tasks(self, message: GroupMessage, topic_id: str, customer_id: str, bindings: list, messages: list) -> list:
        """同话题多目标全部给模型区分；只投影任务身份、结构条件及本人来源，不选最近者。"""
        tasks = {}
        for binding in bindings:
            task_id = binding.get('task_context_id')
            if not task_id or binding['customer_id'] != customer_id:
                continue
            if task_id in tasks:
                continue
            context = self.store.get_task_context(task_id)
            if context is None or context.customer_id != customer_id or context.conversation_id != binding['conversation_id']:
                continue
            row = {'customer_id':customer_id,'topic_id':topic_id,'task_context_id':task_id,
                   'conversation_id':context.conversation_id,'status':context.status,'facts':{}}
            task_bindings = [item for item in bindings if item.get('task_context_id')==task_id
                             and item['customer_id']==customer_id and item['conversation_id']==context.conversation_id
                             and item.get('source_message_id')]
            task_bindings.sort(key=lambda item:(datetime.fromisoformat(item['updated_at']),item['source_message_id']))
            # 首条来源的分类尚无 task_context_id，不能只按后续分类引用筛选。保留最初声明，
            # 再按本人持久来源合并最新条件；源记录查询有界，省略数量显式保留。
            unique_ids = list(dict.fromkeys(item['source_message_id'] for item in task_bindings))
            selected_ids = unique_ids if len(unique_ids)<=30 else [unique_ids[0],*unique_ids[-29:]]
            source_rows = []
            by_message_id = {item['message_id']:item for item in messages}
            for source_id in selected_ids:
                source = by_message_id.get(source_id)
                if source is None:
                    source = self.store.get_group_topic_message(message.channel,message.room_id,customer_id,source_id)
                if source is not None and source['topic_id']==topic_id and datetime.fromisoformat(source['sent_at'])<=message.sent_at:
                    source_rows.append(source)
            source_rows.sort(key=lambda item:(datetime.fromisoformat(item['sent_at']),item['message_id']))
            if source_rows:
                row['goal_summary'] = source_rows[0]['text']
                for source in source_rows:
                    row['facts'].update(self._safe_facts(source.get('classification',{}).get('constraints',{})))
            row['condition_sources_omitted'] = max(0,len(unique_ids)-len(selected_ids))
            checkpoint = self.store.get_task_context_checkpoint(task_id)
            if checkpoint is not None:
                row['facts'].update(self._safe_facts(checkpoint.facts))
            tasks.setdefault(task_id,row)
        return list(tasks.values())

    @staticmethod
    def _safe_facts(value: dict) -> dict:
        """对本人 checkpoint 也只取强类型条件；内部备注与整段摘要不是路由所需资料。"""
        if not isinstance(value,dict):
            return {}
        facts = {}
        for key in PUBLIC_TOPIC_FIELDS:
            if key in value:
                try:
                    facts.update(validate_public_topic_summary({key:value[key]}))
                except (ValueError,TypeError):
                    pass
        for key in ('planned_start_at','latest_arrival_at','latest_wait_at','end_at'):
            if isinstance(value.get(key),str):
                try:
                    parsed = datetime.fromisoformat(value[key])
                    if parsed.tzinfo is not None:
                        facts[key] = parsed.isoformat()
                except ValueError:
                    pass
        if type(value.get('party_size')) is int and 1<=value['party_size']<=4:
            facts['party_size'] = value['party_size']
        return facts


__all__ = ['TopicRoutingService']
