#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys


APPLESCRIPT = r'''
on run argv
    set recipientId to item 1 of argv
    set messageText to item 2 of argv
    set appName to item 3 of argv

    set the clipboard to messageText
    tell application appName to activate
    delay 0.8

    tell application "System Events"
        keystroke "f" using {command down}
        delay 0.4
        keystroke recipientId
        delay 0.8
        key code 36
        delay 0.6
        keystroke "v" using {command down}
        delay 0.2
        key code 36
    end tell
end run
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one test message through Mac WeChat UI automation.")
    parser.add_argument("--send", action="store_true", help="Actually send. Without this flag the script only prints.")
    parser.add_argument("--app-name", default="WeChat", help="macOS app name, usually WeChat or 微信.")
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read())
    target = payload["target"]["target_id"]
    text = payload["text"]

    if not args.send:
        print(f"dry_run:wechat:{target}")
        return 0

    result = subprocess.run(
        ["osascript", "-e", APPLESCRIPT, target, text, args.app_name],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout or f"osascript exit {result.returncode}")
        return result.returncode
    print(f"wechat:{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
