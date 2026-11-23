param([switch]$CheckOnly, [switch]$Restart, [switch]$Stop)
# Windows 旧桥接入口：一份本机配置同时启动 Runtime 与桥接，不切换微信账号或发送授权。
$ErrorActionPreference = 'Stop'
if (@($CheckOnly,$Restart,$Stop | Where-Object { $_ }).Count -gt 1) { throw 'CheckOnly、Restart 和 Stop 只能选择一项。' }
$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repoRoot
$configFile = Join-Path $repoRoot '.env.wechat-stack.local'
if (-not (Test-Path -LiteralPath $configFile)) { throw '缺少 .env.wechat-stack.local，请按 README 的 Windows 桥接启动说明配置。' }
foreach ($line in Get-Content -LiteralPath $configFile -Encoding UTF8) {
    if ($line -match '^\s*((?:MAHJONG_[A-Z0-9_]+)|HTTPS?_PROXY)\s*=(.*)$') {
        $name = $Matches[1]; $value = $Matches[2].Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) { $value = $value.Substring(1, $value.Length-2) }
        # 启动器以配置文件为共同事实源，避免继承旧终端留下的半套配置。
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
function Required-Value([string]$Name) {
    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($value)) { throw "缺少启动配置：$Name" }
    return $value.Trim()
}
function File-Value([string]$Name) {
    $file = Required-Value $Name
    if (-not [IO.Path]::IsPathRooted($file)) { $file = Join-Path $repoRoot $file }
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "配置指向的文件不存在：$Name" }
    $value = [IO.File]::ReadAllText($file).Trim()
    if (-not $value) { throw "凭证文件为空：$Name" }
    return $value
}
function Source-Build([string[]]$Files) {
    $bytes = [IO.MemoryStream]::new()
    try {
        foreach($file in $Files) { $part = [IO.File]::ReadAllBytes((Join-Path $repoRoot $file)); $bytes.Write($part,0,$part.Length) }
        return ([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes.ToArray()))).Replace('-','').ToLowerInvariant()
    } finally { $bytes.Dispose() }
}
$primary = Required-Value 'MAHJONG_AGENT_WECHAT_ACCOUNT'
$python = Required-Value 'MAHJONG_WECHAT_PYTHON_BIN'
$null = File-Value 'MAHJONG_AGENT_WECHAT_TOKEN_PATH'
$apiToken = File-Value 'MAHJONG_AGENT_API_TOKEN_PATH'
$controlToken = File-Value 'MAHJONG_WECHATY_CONTROL_TOKEN_PATH'
$null = Required-Value 'MAHJONG_WECHATY_LOCAL_DATA_DIR'
foreach($name in @('MAHJONG_AGENT_API_TOKEN','MAHJONG_WECHATY_AGENT_API_TOKEN','MAHJONG_WECHATY_CONTROL_TOKEN')) {
    [Environment]::SetEnvironmentVariable($name,$null,'Process')
}
$env:MAHJONG_WECHAT_TRANSPORT = 'agent-wechat'
# 明确选择备用桥接控制协议，避免 Runtime 使用 .env 中的独立适配器令牌。
$env:MAHJONG_WECHATY_CONTROL_TRANSPORT = 'bridge'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $repoRoot 'src'
$runtimePort = if($env:MAHJONG_AGENT_PORT){[int]$env:MAHJONG_AGENT_PORT}else{8790}
$bridgePort = if($env:MAHJONG_WECHATY_OUTBOUND_PORT){[int]$env:MAHJONG_WECHATY_OUTBOUND_PORT}else{8791}
$runtimeUrl = "http://127.0.0.1:$runtimePort"
$bridgeUrl = "http://127.0.0.1:$bridgePort"
$env:MAHJONG_WECHATY_RAW_ENDPOINT = "$runtimeUrl/api/channels/wechaty/raw"
# 引用回执和入站必须写入同一 Runtime，尤其是同时运行多个本地实例时。
$env:MAHJONG_WECHATY_REFERENCE_ENDPOINT = "$runtimeUrl/api/message-references/link"
$env:MAHJONG_WECHATY_OUTBOUND_BASE_URL = $bridgeUrl
$env:MAHJONG_WECHATY_OUTBOUND_ENABLED = 'true'
$env:MAHJONG_AGENT_WECHAT_CONFIG_PATH = if($env:MAHJONG_AGENT_WECHAT_CONFIG_PATH){$env:MAHJONG_AGENT_WECHAT_CONFIG_PATH}else{Join-Path $repoRoot 'data\agent_wechat_setup.local.json'}
$active = if(Test-Path -LiteralPath $env:MAHJONG_AGENT_WECHAT_CONFIG_PATH){Get-Content -LiteralPath $env:MAHJONG_AGENT_WECHAT_CONFIG_PATH -Raw | ConvertFrom-Json}else{[pscustomobject]@{account=$primary;revision='initial'}}
$runtimeHeaders = @{Authorization="Bearer $apiToken";'X-Mahjong-Bridge-Token'=$controlToken}
$bridgeHeaders = @{'X-Mahjong-Bridge-Token'=$controlToken}
function Check-Stack {
    $runtime = Invoke-RestMethod "$runtimeUrl/api/channels/wechaty/preflight" -Headers $runtimeHeaders -TimeoutSec 5
    if($runtime.protocol -ne 1 -or $runtime.build -ne (Source-Build @('scripts/agent_runtime_app.py'))) { throw 'Runtime 运行旧版代码，请同步重启。' }
    if($runtime.primary_account -ne $primary -or $runtime.account -ne $active.account -or $runtime.revision -ne $active.revision) { throw 'Runtime 的账号或接入版本与共同配置不一致。' }
    $bridge = Invoke-RestMethod "$bridgeUrl/listener-status" -Headers $bridgeHeaders -TimeoutSec 5
    $expected = Source-Build @('integrations/wechaty/mahjong-wechaty-bridge/src/bridge.mjs','integrations/wechaty/mahjong-wechaty-bridge/src/agent-wechat.mjs','integrations/wechaty/mahjong-wechaty-bridge/src/bridge-operations.mjs')
    if($bridge.protocol -ne 1 -or $bridge.build -ne $expected) { throw '桥接运行旧版代码，请同步重启。' }
    if($bridge.primary_account -ne $primary -or $bridge.agent_wechat_account -ne $active.account -or $bridge.agent_wechat_revision -ne $active.revision) { throw '桥接的账号或接入版本与共同配置不一致。' }
    if(-not $bridge.listener.ready) { throw "监听未就绪：$($bridge.listener.last_error)" }
    # 同时校验工作记录实际经由 Runtime 转发的控制路径和凭证。
    $proxied = Invoke-RestMethod "$runtimeUrl/api/wechat/listener-status" -Headers $runtimeHeaders -TimeoutSec 5
    if($proxied.build -ne $bridge.build -or -not $proxied.listener.ready) { throw 'Runtime 无法转发微信监听状态，请检查控制配置。' }
    return $bridge
}
if($CheckOnly) { Check-Stack | Select-Object primary_account,agent_wechat_account,listener | ConvertTo-Json -Depth 4; exit 0 }
$dataDir = Join-Path $repoRoot 'data'
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$lock = [IO.File]::Open((Join-Path $dataDir 'wechat-bridge-launch.lock'), 'OpenOrCreate','ReadWrite','None')
$stateFile = Join-Path $dataDir 'wechat-bridge-processes.local.json'
$children = @()
try {
    if($Restart -or $Stop) {
        if(-not (Test-Path -LiteralPath $stateFile)) { throw '现有服务不由此入口管理，请先通过原启动方式停止，再使用此入口。' }
        $owned = @(Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json)
        # 先核对全部进程，再停止；不依据端口或失效 PID 杀死其他服务。
        foreach($entry in $owned) {
            $process = Get-Process -Id $entry.id -ErrorAction SilentlyContinue
            if($process) {
                $info = Get-CimInstance Win32_Process -Filter "ProcessId=$($entry.id)"
                if($process.StartTime.ToUniversalTime().Ticks -ne ([datetime]$entry.started).ToUniversalTime().Ticks -or -not $info.CommandLine.Contains($entry.script)) { throw '进程身份已变化，拒绝停止，请检查当前服务。' }
            }
        }
        foreach($entry in @($owned | Sort-Object order -Descending)) { Stop-Process -Id $entry.id -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $stateFile
        if($Stop) { Write-Output '已停止此入口管理的 Runtime 与桥接。'; exit 0 }
    }
    $occupied = @(Get-NetTCPConnection -State Listen -LocalPort $runtimePort,$bridgePort -ErrorAction SilentlyContinue)
    if($occupied.Count) { $null = Check-Stack; Write-Output '已有整套服务且链路校验通过，已复用。'; exit 0 }
    $native = if($env:MAHJONG_AGENT_WECHAT_URL){$env:MAHJONG_AGENT_WECHAT_URL}else{'http://127.0.0.1:6174'}
    $uri = [Uri]$native
    if($uri.Scheme -ne 'http' -or $uri.Host -notin @('127.0.0.1','localhost','[::1]') -or $uri.UserInfo -or $uri.AbsolutePath -ne '/') { throw '微信后台必须使用本机 HTTP 地址。' }
    $nativeHeaders = @{Authorization='Bearer '+(File-Value 'MAHJONG_AGENT_WECHAT_TOKEN_PATH')}
    $account = Invoke-RestMethod ($native.TrimEnd('/')+'/v1/accounts/'+[Uri]::EscapeDataString($active.account)) -Headers $nativeHeaders -TimeoutSec 5
    if($account.id -ne $active.account -or $account.authenticated -ne $true) { throw '微信后台的所选账号未登录，请先在微信工作台登录。' }
    $logDir = Join-Path $repoRoot 'logs\wechat-bridge-stack'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $node = (Get-Command node.exe).Source
    foreach($entry in @(@{name='runtime';exe=$python;script=(Join-Path $repoRoot 'scripts\run_agent_app.py')},@{name='bridge';exe=$node;script=(Join-Path $repoRoot 'integrations\wechaty\mahjong-wechaty-bridge\src\bridge.mjs')})) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $child = Start-Process -FilePath $entry.exe -ArgumentList ('"'+$entry.script+'"') -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "$stamp-$($entry.name).out.log") -RedirectStandardError (Join-Path $logDir "$stamp-$($entry.name).err.log") -PassThru
        $children += [pscustomobject]@{id=$child.Id;started=$child.StartTime.ToUniversalTime().ToString('o');script=$entry.script;order=$children.Count}
        ConvertTo-Json -InputObject @($children) | Set-Content -LiteralPath $stateFile -Encoding UTF8
        if($entry.name -eq 'runtime') {
            $ready = $false
            for($attempt=0;$attempt -lt 60;$attempt++) {
                if($child.HasExited){throw 'Runtime 启动失败，请查看 logs/wechat-bridge-stack。'}
                try { $null = Invoke-RestMethod "$runtimeUrl/api/channels/wechaty/preflight" -Headers $runtimeHeaders -TimeoutSec 2; $ready=$true; break } catch { Start-Sleep -Milliseconds 250 }
            }
            if(-not $ready){throw 'Runtime 未就绪，请查看启动日志。'}
        }
    }
    $ready=$false
    for($attempt=0;$attempt -lt 60;$attempt++) {
        try { $null = Check-Stack; $ready=$true; break } catch { Start-Sleep -Milliseconds 250 }
    }
    if(-not $ready){throw '链路预检查未通过，请查看启动日志；未报告启动成功。'}
    Write-Output "启动与链路校验通过：$runtimeUrl/agent-work。外发开关沿用当前接入配置的持久设置。"
} catch {
    foreach($entry in $children) { Stop-Process -Id $entry.id -ErrorAction SilentlyContinue }
    if($children.Count -and (Test-Path -LiteralPath $stateFile)){Remove-Item -LiteralPath $stateFile}
    throw
} finally { $lock.Dispose() }
exit 0
