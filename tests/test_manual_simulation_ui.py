"""用可控 Promise 验证手动输入区的并发行为，不启动浏览器或真实 HTTP。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "scripts" / "static" / "simulation_observability.html"


@pytest.fixture
def node_binary() -> str:
    """优先使用已配置的 Node；缺少运行时应明确标记测试未执行。"""
    binary = shutil.which("node")
    if not binary and Path("/opt/homebrew/bin/node").is_file():
        binary = "/opt/homebrew/bin/node"
    if not binary:
        pytest.skip("手动输入区行为测试需要 Node.js，当前环境未安装可执行的 node")
    return binary


def _composer_functions() -> str:
    """直接执行页面中的真实交互函数，DOM 与网络依赖才使用最小桩。"""
    html = HTML_PATH.read_text(encoding="utf-8")
    functions = []
    for name in ("renderManualComposer", "manualControl", "retryManualInput"):
        match = re.search(
            rf"^(?:async )?function {name}\([^\n]*\)\{{\n.*?^\}}",
            html,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None, f"页面缺少可提取的函数 {name}"
        functions.append(match.group())
    return "\n".join(functions)


HARNESS = r"""
const elements = new Map();
function $(id) {
 if (!elements.has(id)) elements.set(id, {
  value: '', disabled: true, hidden: false, checked: false,
  innerHTML: '', textContent: '', dataset: {}, closest: () => null,
 });
 return elements.get(id);
}
const sendButton = {disabled: true};
const actionButton = {disabled: false, closest: () => null};
const document = {querySelectorAll(selector) {
 if (selector === '.manual-send') return [sendButton];
 if (selector === '.manual-action,.manual-recall') return [actionButton];
 throw new Error('非预期 DOM 查询: ' + selector);
}};
const requests = [], refreshRunIds = [], navigationUrls = [];
let generatedRequestCount = 0;
const crypto = {randomUUID: () => `request-${++generatedRequestCount}`};
const location = {href: 'https://simulation.invalid/?run_id=run-a'};
const history = {replaceState(_state, _title, url) {navigationUrls.push(String(url));}};
function makeRun(runId) {
 return {run_id: runId, transcript: [], report: {
  manual_session: true, manual_workspace_version: 2, default_actor_id: 'actor-a',
  manual_actors: [{customer_id: 'actor-a', display_name: '客户A'},
                  {customer_id: 'actor-b', display_name: '客户B'}],
 }};
}
const state = {
 data: {selected_run: makeRun('run-a')}, runId: 'run-a', viewId: 'initial-view',
 sequence: 7, manualActorId: 'actor-a', manualBusy: false,
 manualPending: {}, manualRetries: {}, manualLatestRequest: {},
};
$('manualChannel').value = 'group';
$('manualQuote').value = '7';
$('followLive').checked = false;
function currentRun() {return state.data.selected_run;}
function esc(value) {return String(value ?? '');}
function ensureManualWorkspaceUI() {}
function currentViews() {
 return [{view_id: `${currentRun().run_id}-view`, turns: [{sequence: 1}, {sequence: 2}]}];
}
function idFor(view) {return view.view_id;}
function renderViews() {}
function renderChat() {}
function chooseManualView() {}
async function refresh() {
 refreshRunIds.push(state.runId);
 state.data = {selected_run: makeRun(state.runId)};
 renderManualComposer();
}
function fetch(url, options) {
 assert.equal(url, '/api/simulations/manual', '不得发起任何真实请求');
 assert.equal(options.method, 'POST');
 let resolve, reject;
 const promise = new Promise((accept, refuse) => {resolve = accept; reject = refuse;});
 requests.push({body: JSON.parse(options.body), resolve, reject});
 return promise;
}
function complete(index, body = {}) {
 const request = requests[index];
 request.resolve({ok: true, json: async () => ({
  ok: true, run_id: request.body.run_id, actor_id: request.body.actor_id,
  sequence: index + 1, ...body,
 })});
}
function setEditor({text = '', actor = 'actor-a', channel = 'group', quote = '7'} = {}) {
 $('manualText').value = text;
 state.manualActorId = actor;
 $('manualChannel').value = channel;
 $('manualQuote').value = quote;
}
"""


def _run_ui_case(node_binary: str, case: str) -> None:
    """只在隔离 VM 中驱动页面函数，所有请求必须显式由用例完成。"""
    javascript = HARNESS + "\n" + _composer_functions() + "\n(async () => {\n" + case + "\n})()"
    launcher = (
        "const vm = require('node:vm');\n"
        "const assert = require('node:assert/strict');\n"
        f"const source = {json.dumps(javascript)};\n"
        "Promise.resolve(vm.runInNewContext(source, {assert, URL}, {timeout: 1000}))"
        ".then(() => process.stdout.write('ok\\n'))"
        ".catch(error => {process.stderr.write(error.stack + '\\n'); process.exitCode = 1;});\n"
    )
    result = subprocess.run(
        [node_binary, "-"], input=launcher, text=True, capture_output=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout == "ok\n", "异步用例没有执行到完成；可能有未完成的请求"


def test_concurrent_sends_capture_payload_and_late_completion_preserves_new_draft(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 const firstText = '  第一条原话\n先问下午  ';
 setEditor({text: firstText});
 const first = manualControl('send');
 assert.equal(requests.length, 1);
 assert.equal($('manualText').value, '', '等待请求完成前就要清空已提交正文');
 assert.equal(state.manualBusy, true);
 renderManualComposer();
 assert.equal($('manualText').disabled, false);
 assert.equal(sendButton.disabled, false);
 assert.equal($('manualActor').disabled, false);
 assert.equal($('manualChannel').disabled, false);
 assert.equal(actionButton.disabled, true, '并发发送不应顺带解锁其他写操作');

 const secondText = '第二条改成晚上';
 setEditor({text: secondText, actor: 'actor-b', channel: 'private', quote: '99'});
 const second = manualControl('send');
 assert.equal(requests.length, 2, '首个请求 pending 时仍可提交另一条消息');
 assert.notEqual(requests[0].body.request_id, requests[1].body.request_id);
 assert.deepEqual(
  [requests[0].body.text, requests[0].body.actor_id, requests[0].body.channel,
   requests[0].body.quote_sequence, requests[0].body.run_id],
  [firstText, 'actor-a', 'group', 7, 'run-a'],
 );
 assert.deepEqual(
  [requests[1].body.text, requests[1].body.actor_id, requests[1].body.channel,
   requests[1].body.quote_sequence, requests[1].body.run_id],
  [secondText, 'actor-b', 'private', null, 'run-a'],
 );
 assert.equal($('manualText').value, '');

 complete(1);
 await second;
 assert.equal(state.sequence, 2);
 assert.equal(state.manualBusy, true, '第二条结束时第一条仍在处理');
 $('manualText').value = '第三条尚未发送的草稿';
 complete(0);
 await first;
 assert.equal($('manualText').value, '第三条尚未发送的草稿');
 assert.equal(state.sequence, 2, '晚到完成不能选回更早的轮次');
 assert.equal(state.viewId, 'run-a-view');
 assert.equal(state.manualActorId, 'actor-b', '晚到完成不能切回旧角色');
 assert.equal($('manualChannel').value, 'private');
 assert.equal(state.runId, 'run-a');
 assert.equal(state.manualBusy, false);
 assert.equal(Object.keys(state.manualPending).length, 0);
 """)


def test_completed_request_does_not_replace_actor_selected_while_pending(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 setEditor({text: '客户A的消息'});
 const pending = manualControl('send');
 setEditor({text: '客户B正在写的草稿', actor: 'actor-b', channel: 'private'});
 complete(0);
 await pending;
 assert.equal(state.manualActorId, 'actor-b');
 assert.equal($('manualChannel').value, 'private');
 assert.equal($('manualText').value, '客户B正在写的草稿');
 """)


def test_late_completion_cannot_reactivate_previous_run(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 setEditor({text: '旧运行的消息'});
 const pending = manualControl('send');
 state.runId = 'run-b';
 state.data = {selected_run: makeRun('run-b')};
 state.sequence = 88;
 state.viewId = 'run-b-selected-view';
 setEditor({text: '新运行的草稿', actor: 'actor-b', channel: 'private'});
 complete(0);
 await pending;
 assert.equal(requests[0].body.run_id, 'run-a');
 assert.equal(state.runId, 'run-b');
 assert.equal(currentRun().run_id, 'run-b');
 assert.equal(state.sequence, 88);
 assert.equal(state.viewId, 'run-b-selected-view');
 assert.equal(state.manualActorId, 'actor-b');
 assert.equal($('manualText').value, '新运行的草稿');
 assert.equal(navigationUrls.length, 0);
 assert.deepEqual(refreshRunIds, ['run-b']);
 """)


def test_latest_failure_restores_matching_empty_composer_and_reuses_retry_id(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 const original = '  网络断开时保留我的原文\n不要丢失  ';
 setEditor({text: original});
 const pending = manualControl('send');
 requests[0].reject(new Error('连接中断'));
 assert.equal(await pending, null);
 assert.equal($('manualText').value, original);
 assert.equal($('controlStatus').textContent, '连接中断');
 assert.equal(state.manualBusy, false);
 const retry = manualControl('send');
 assert.equal($('manualText').value, '');
 assert.equal(requests.length, 2);
 assert.equal(requests[1].body.request_id, requests[0].body.request_id,
              '只重试同一快照时复用请求ID，避免已收消息的重复副作用');
 assert.equal(requests[1].body.text, original);
 complete(1);
 await retry;
 assert.equal(Object.keys(state.manualRetries).length, 0);
 """)


def test_failure_never_overwrites_new_draft_or_reuses_id_for_different_text(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 setEditor({text: '失败的旧消息'});
 const pending = manualControl('send');
 $('manualText').value = '已经在写的新消息';
 requests[0].reject(new Error('连接中断'));
 assert.equal(await pending, null);
 assert.equal($('manualText').value, '已经在写的新消息');
 const next = manualControl('send');
 assert.notEqual(requests[1].body.request_id, requests[0].body.request_id);
 assert.equal(requests[1].body.text, '已经在写的新消息');
 complete(1);
 await next;
 """)


def test_stale_failure_does_not_restore_text_or_replace_latest_status(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 setEditor({text: '第一条'});
 const first = manualControl('send');
 setEditor({text: '第二条'});
 const second = manualControl('send');
 const latestStatus = $('controlStatus').textContent;
 requests[0].reject(new Error('第一条的迟到失败'));
 assert.equal(await first, null);
 assert.equal($('manualText').value, '');
 assert.equal($('controlStatus').textContent, latestStatus);
 assert.equal(state.manualBusy, true);
 complete(1);
 await second;
 assert.equal(state.sequence, 2);
 assert.equal(state.manualBusy, false);
 """)


@pytest.mark.parametrize("changed_target", ["actor", "channel", "quote", "run"])
def test_failure_restores_only_unchanged_request_target(node_binary: str, changed_target: str) -> None:
    change = {
        "actor": "state.manualActorId = 'actor-b';",
        "channel": "$('manualChannel').value = 'private';",
        "quote": "$('manualQuote').value = '99';",
        "run": "state.runId = 'run-b'; state.data = {selected_run: makeRun('run-b')};",
    }[changed_target]
    _run_ui_case(node_binary, r"""
 setEditor({text: '只能恢复到原角色和通道的正文'});
 const pending = manualControl('send');
 """ + change + r"""
 requests[0].reject(new Error('旧目标请求失败'));
 assert.equal(await pending, null);
 assert.equal($('manualText').value, '', '不能将旧请求正文恢复到另一个目标');
 assert.equal(Object.keys(state.manualRetries).length, 1, '原请求仍应保留可重试身份');
 assert.equal(state.manualBusy, false);
 """)


def test_pending_input_recovery_preserves_draft_and_replays_original_request_snapshot(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 const originalText = '  原请求的完整正文\n保留换行和空格  ';
 const savedInput = {
  action: 'send', text: originalText, actor_id: 'actor-a', channel: 'group',
  quote_sequence: 7, message_id: 'original-message',
 };
 currentRun().transcript = [{sequence: 8, manual_processing: true,
  manual_request_id: 'persisted-request', manual_input: savedInput}];
 currentRun().report.manual_operations = {
  'persisted-request': {action: 'send', sequence: 8, status: 'processing'},
 };
 setEditor({text: '第三条尚未发送的草稿', actor: 'actor-b', channel: 'private', quote: '99'});
 const recovery = retryManualInput(8);
 assert.equal(requests.length, 1);
 assert.deepEqual(
  [requests[0].body.action, requests[0].body.request_id, requests[0].body.text,
   requests[0].body.actor_id, requests[0].body.channel, requests[0].body.quote_sequence,
   requests[0].body.run_id, requests[0].body.message_id],
  ['send', 'persisted-request', originalText, 'actor-a', 'group', 7,
   'run-a', 'original-message'],
 );
 assert.equal(requests[0].body.preserve_composer, true);
 assert.equal($('manualText').value, '第三条尚未发送的草稿', '恢复旧输入不能清空新草稿');
 assert.equal(state.manualActorId, 'actor-b');
 assert.equal($('manualChannel').value, 'private');
 assert.equal(savedInput.action, 'send', '恢复不能改写已保存的原始输入');
 complete(0, {sequence: 8});
 await recovery;
 assert.equal($('manualText').value, '第三条尚未发送的草稿');
 assert.equal(state.manualActorId, 'actor-b');
 assert.equal($('manualChannel').value, 'private');
 assert.equal(state.manualBusy, false);
 """)


def test_batch_recovery_uses_saved_operation_action_instead_of_inner_send_action(node_binary: str) -> None:
    _run_ui_case(node_binary, r"""
 const savedInput = {
  action: 'send', text: '批次里正在处理的原文', actor_id: 'actor-a',
  channel: 'group', quote_sequence: 7, message_id: 'batch-original-message',
 };
 currentRun().transcript = [{sequence: 5, manual_processing: true,
  manual_request_id: 'persisted-batch-request', manual_input: savedInput}];
 currentRun().report.manual_operations = {
  'persisted-batch-request': {action: 'batch_next', sequence: 5, status: 'processing'},
 };
 setEditor({text: '批次以外正在写的草稿', actor: 'actor-b', channel: 'private', quote: '99'});
 const recovery = retryManualInput(5);
 assert.equal(requests.length, 1);
 assert.equal(requests[0].body.action, 'batch_next',
              '已保存的展开send输入不能覆盖原批次操作类型');
 assert.equal(state.manualPending['persisted-batch-request'].action, 'batch_next');
 assert.deepEqual(
  [requests[0].body.request_id, requests[0].body.text, requests[0].body.actor_id,
   requests[0].body.channel, requests[0].body.quote_sequence, requests[0].body.run_id],
  ['persisted-batch-request', '批次里正在处理的原文', 'actor-a', 'group', 7, 'run-a'],
 );
 assert.equal(requests[0].body.preserve_composer, true);
 assert.equal($('manualText').value, '批次以外正在写的草稿');
 assert.equal(savedInput.action, 'send');
 complete(0, {sequence: 5});
 await recovery;
 assert.equal($('manualText').value, '批次以外正在写的草稿');
 assert.equal(state.manualActorId, 'actor-b');
 assert.equal(state.manualBusy, false);
 """)
