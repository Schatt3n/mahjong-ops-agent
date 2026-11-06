"""群模拟观测视图必须以真实投递为依据，不将入口标签当作外发通道。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from simulation_observability import simulation_observability_payload


def test_automatic_group_outputs_project_only_authoritative_private_targets(tmp_path):
    """自动模拟与手动模拟共用投影；不把给B的通知伪装成B发过A的原话。"""
    run_id='20261105T010000Z_projection'
    folder=tmp_path/'runs'/run_id
    folder.mkdir(parents=True)
    outputs=[
        {'channel':'group','recipient_id':'room','text':'私聊回你'},
        {'channel':'private','recipient_id':'a','recipient_name':'客户A','text':'给A的内容','conversation_id':'private:a'},
        {'channel':'private','recipient_id':'b','recipient_name':'客户B','text':'给B的通知','conversation_id':'private:b'},
        {'channel':'private','recipient_id':'unknown','text':'只有原始证据，无会话目标'},
    ]
    turn={'sequence':1,'conversation_id':'group:room','channel':'group','thread_id':'source-thread',
          'user':{'customer_id':'a','display_name':'客户A','text':'A的群内原话'},
          'agent':{'reply':'给A的内容','trace_id':'trace:1'},
          'group_processing':{'entrypoint':'GroupMessageHandler','routing':{'action':'private_switch'},'outputs':outputs}}
    (folder/'report.json').write_text(json.dumps({'transcript':[turn]}),encoding='utf-8')
    result=simulation_observability_payload(runtime_dir=tmp_path,selected_run_id=run_id)['selected_run']
    views={view['conversation_id']:view for view in result['conversations']}
    assert set(views)=={'group:room','private:a','private:b'}
    assert result['turn_count']==1
    assert result['transcript']==[turn]
    a=views['private:a']['turns'][0]
    b=views['private:b']['turns'][0]
    assert a['user']['text']=='A的群内原话'
    assert a['agent']['reply']=='给A的内容'
    assert b['user']['text']=='（群事件产生的主动私聊通知）'
    assert b['agent']['reply']=='给B的通知'
    assert a['sequence']==b['sequence']==1
    assert len(a['private_delivery_outputs'])==len(b['private_delivery_outputs'])==1


def test_legacy_group_transcript_does_not_invent_delivery_evidence(tmp_path):
    """旧直连记录原样保留，不能补造群入口或私聊发送证明。"""
    run_id='20261105T010001Z_legacy'
    folder=tmp_path/'runs'/run_id
    folder.mkdir(parents=True)
    turn={'sequence':1,'conversation_id':'old:group','channel':'group',
          'user':{'customer_id':'a','display_name':'客户A','text':'有人吗'},'agent':{'reply':'旧回复'}}
    (folder/'report.json').write_text(json.dumps({'transcript':[turn]}),encoding='utf-8')
    result=simulation_observability_payload(runtime_dir=tmp_path,selected_run_id=run_id)['selected_run']
    assert result['transcript']==[turn]
    assert len(result['conversations'])==1
    assert 'group_processing' not in result['conversations'][0]['turns'][0]
