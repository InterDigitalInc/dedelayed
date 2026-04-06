# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import os
from shlex import quote


def commit_version(rev: str = "", root: str = ".") -> str:
    cmd_flags = "--long --always --tags --match='v[0-9]*'"
    cmd = f"git -C {quote(root)} describe {cmd_flags}"
    cmd += " --dirty" if rev == "" else f" {quote(rev)}"
    return os.popen(cmd).read().rstrip()
