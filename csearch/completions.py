"""Static shell completion scripts + install/uninstall, via
`python -m csearch completions {bash|zsh|fish|install|uninstall}`.

Hand-written (argparse cannot emit completions) and intentionally static: no
argcomplete dependency, no Python startup per TAB press. Completions target
the command name `csearch`; `install` materializes it as a wrapper script in
~/.local/bin plus the per-shell completion files:

    bash  ~/.local/share/bash-completion/completions/csearch
          + one managed source line in ~/.bashrc (works without bash-completion;
          the XDG path is lazy-loaded when the package is present)
    zsh   ~/.local/share/zsh/site-functions/_csearch
          + one managed fpath line in ~/.zshenv (dir is not in default fpath)
    fish  ~/.config/fish/completions/csearch.fish              (auto-loaded)

Every installed file carries the MANAGED marker; `uninstall` only removes
marker-carrying files and refuses (keeps) anything else. If the argparse
choices in cli.py change, keep the scripts' flag lists in sync.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MANAGED = "managed by csearch completions install"
SHELLS = ("bash", "zsh", "fish")

BASH = r"""# bash completion for csearch -*- shell-script -*-
# setup: python -m csearch completions install bash
#   (manual: eval "$(<python> -m csearch completions bash)" after defining `csearch`)
_csearch() {
  local cur prev sub sub_at i flags
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  COMPREPLY=()
  sub=""
  sub_at=0
  for ((i = 1; i < COMP_CWORD; i++)); do
    case "${COMP_WORDS[i]}" in
      index|update|search|status|stats|completions|i|u|s|st)
        sub="${COMP_WORDS[i]}"
        sub_at=$i
        case "$sub" in
          i)  sub=index ;;
          u)  sub=update ;;
          s)  sub=search ;;
          st) sub=status ;;
        esac
        break ;;
    esac
  done
  if [[ -z "$sub" ]]; then
    mapfile -t COMPREPLY < <(compgen -W 'index update search status stats completions' -- "$cur")
    return 0
  fi
  case "$sub" in
    completions)
      if (( COMP_CWORD == sub_at + 1 )); then
        mapfile -t COMPREPLY < <(compgen -W 'bash zsh fish install uninstall' -- "$cur")
      else
        mapfile -t COMPREPLY < <(compgen -W 'bash zsh fish' -- "$cur")
      fi
      return 0 ;;
  esac
  case "$prev" in
    --backend) mapfile -t COMPREPLY < <(compgen -W 'ort torch' -- "$cur") && return 0 ;;
    --variant) mapfile -t COMPREPLY < <(compgen -W 'q4f16 int8' -- "$cur") && return 0 ;;
    --dim)     mapfile -t COMPREPLY < <(compgen -W '128 256 512 768' -- "$cur") && return 0 ;;
    --model)   mapfile -t COMPREPLY < <(compgen -d -- "$cur") && return 0 ;;
  esac
  flags='--model --backend --variant --dim --batch --cpu --local --multimodal'
  [[ "$sub" == search ]] && flags="$flags -k --show"
  if [[ "$cur" == -* ]]; then
    mapfile -t COMPREPLY < <(compgen -W "$flags" -- "$cur")
  else
    mapfile -t COMPREPLY < <(compgen -d -- "$cur")
  fi
  return 0
}
complete -o default -o filenames -F _csearch csearch
"""

ZSH = r"""#compdef csearch
# zsh completion for csearch
# setup: python -m csearch completions install zsh
#   (manual: eval "$(<python> -m csearch completions zsh)" after compinit)
_csearch() {
  local -a subs
  subs=(
    'index:build the index or force an incremental rebuild'
    'update:incrementally refresh; instant no-op when fresh'
    'search:semantic query -> file:line hits'
    'status:index freshness without loading the model'
    'stats:one-line index metadata'
    'completions:print, install or uninstall shell completions'
  )
  _arguments -C '1:sub:->sub' '*::arg:->args'
  case $state in
    sub)
      _describe -t csearch-cmd 'csearch command' subs ;;
    args)
      case $words[1] in
        index|update|i|u)
          _arguments \
            '1:repo:_directories' \
            '(--model)--model=[model dir]:dir:_directories' \
            '(--backend)--backend=[inference backend]:backend:(ort torch)' \
            '(--variant)--variant=[onnx weights]:variant:(q4f16 int8)' \
            '(--dim)--dim=[MRL dim]:dim:(128 256 512 768)' \
            '(--batch)--batch=[encode batch size]:batch:' \
            '(--cpu)--cpu[force CPU inference]' \
            '(--local)--local[embed in this process instead of the shared daemon]' \
            '(--multimodal)--multimodal[experimental: also index images and PDFs]' ;;
        search|s)
          _arguments \
            '1:repo:_directories' \
            '2:query:' \
            '(-k)-k=[number of hits]:k:' \
            '(--show)--show[print source of each hit]' \
            '(--model)--model=[model dir]:dir:_directories' \
            '(--backend)--backend=[inference backend]:backend:(ort torch)' \
            '(--variant)--variant=[onnx weights]:variant:(q4f16 int8)' \
            '(--dim)--dim=[MRL dim]:dim:(128 256 512 768)' \
            '(--batch)--batch=[encode batch size]:batch:' \
            '(--cpu)--cpu[force CPU inference]' \
            '(--local)--local[embed in this process instead of the shared daemon]' \
            '(--multimodal)--multimodal[experimental: also index images and PDFs]' ;;
        status|stats|st)
          _arguments '1:repo:_directories' ;;
        completions)
          _arguments \
            '1:action or shell:(bash zsh fish install uninstall)' \
            '*:shell:(bash zsh fish)' ;;
      esac ;;
  esac
}
if (( $+functions[compdef] )); then compdef _csearch csearch; fi
"""

FISH = r"""# fish completion for csearch
# setup: python -m csearch completions install fish

complete -c csearch -f -n 'not __fish_seen_subcommand_from index update search status stats completions i u s st' \
  -a 'index update search status stats completions'
# completions subcommand: action/shell at arg 1, shells afterwards
complete -c csearch -f -n '__fish_seen_subcommand_from completions; and test (count (commandline -opc)) -eq 2' \
  -a 'bash zsh fish install uninstall'
complete -c csearch -f -n '__fish_seen_subcommand_from completions; and test (count (commandline -opc)) -ge 3' \
  -a 'bash zsh fish'
# positional: repo root (directories)
complete -c csearch -n 'test (count (commandline -opc)) -eq 2; and __fish_seen_subcommand_from index update search status stats i u s st' \
  -a '(__fish_complete_directories)'
# shared model flags (index/update/search, including --local and --multimodal)
complete -c csearch -n '__fish_seen_subcommand_from index update search i u s' -l model -r -d 'model dir' -a '(__fish_complete_directories)'
complete -c csearch -f -n '__fish_seen_subcommand_from index update search i u s' -l backend -r -a 'ort torch' -d 'inference backend'
complete -c csearch -f -n '__fish_seen_subcommand_from index update search i u s' -l variant -r -a 'q4f16 int8' -d 'onnx weights'
complete -c csearch -f -n '__fish_seen_subcommand_from index update search i u s' -l dim -r -a '128 256 512 768' -d 'MRL dim'
complete -c csearch -n '__fish_seen_subcommand_from index update search i u s' -l batch -r -d 'encode batch size'
complete -c csearch -n '__fish_seen_subcommand_from index update search i u s' -l cpu -d 'force CPU inference'
complete -c csearch -n '__fish_seen_subcommand_from index update search i u s' -l local -d 'embed in this process instead of the shared daemon'
complete -c csearch -n '__fish_seen_subcommand_from index update search i u s' -l multimodal -d 'experimental: also index images and PDFs'
# search-only
complete -c csearch -f -n '__fish_seen_subcommand_from search s' -s k -r -d 'number of hits'
complete -c csearch -n '__fish_seen_subcommand_from search s' -l show -d 'print source of each hit'
"""

COMPLETIONS = {"bash": BASH, "zsh": ZSH, "fish": FISH}


def _wrapper_script() -> str:
    repo = Path(__file__).resolve().parent.parent
    return (
        "#!/bin/bash\n"
        f"# {MANAGED} — remove via: python -m csearch completions uninstall\n"
        f'export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        f'exec "{sys.executable}" -m csearch "$@"\n'
    )


def _targets(shell: str) -> list[tuple[Path, str, bool]]:
    """(path, content, executable) for one shell's completion files + wrapper."""
    home = Path.home()
    wrapper = (home / ".local/bin/csearch", _wrapper_script(), True)
    if shell == "bash":
        return [(home / ".local/share/bash-completion/completions/csearch", BASH, False), wrapper]
    if shell == "zsh":
        return [(home / ".local/share/zsh/site-functions/_csearch", ZSH, False), wrapper]
    return [(home / ".config/fish/completions/csearch.fish", FISH, False), wrapper]


def _build(content: str) -> str:
    """Ensure the MANAGED marker is present as a comment (idempotent)."""
    if MANAGED in content:
        return content
    lines = content.splitlines(keepends=True)
    if lines and (lines[0].startswith("#!") or lines[0].startswith("#compdef")):
        # shebang / zsh's #compdef must stay on line 1
        return lines[0] + f"# {MANAGED}\n" + "".join(lines[1:])
    return f"# {MANAGED}\n" + content


def _write(path: Path, content: str, executable: bool) -> str:
    """Returns 'installed' | 'updated' | 'unchanged' | 'refused'."""
    content = _build(content)
    if path.exists():
        old = path.read_text(errors="replace")
        if MANAGED not in old:
            return "refused"
        if old != content:
            path.write_text(content)
            what = "updated"
        else:
            what = "unchanged"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        what = "installed"
    if executable and not os.access(path, os.X_OK):
        path.chmod(0o755)
    return what


RC_LINES = {
    # bash has no user fpath; the script is standalone (no bash-completion
    # needed), so one source line works everywhere. bash-completion would
    # also lazy-load the same file from its XDG dir - harmless overlap.
    "bash": (
        "~/.bashrc",
        '[ -r "$HOME/.local/share/bash-completion/completions/csearch" ] && '
        '. "$HOME/.local/share/bash-completion/completions/csearch"',
    ),
    # ~/.zshenv runs before compinit in any interactive shell; the user-level
    # site-functions dir is not in zsh's default fpath.
    "zsh": ("~/.zshenv", 'fpath=("$HOME/.local/share/zsh/site-functions" $fpath)'),
}


def _rc_line(shell: str, add: bool) -> str | None:
    """Add/remove the one managed rc line a shell needs; None = not needed."""
    spec = RC_LINES.get(shell)
    if spec is None:
        return None
    rel, line = spec
    path = Path(rel).expanduser()
    try:
        text = path.read_text() if path.exists() else ""
    except OSError as e:
        return f"{rel}: skipped ({e})"
    is_ours = lambda s: MANAGED in s
    if add:
        if any(is_ours(s) for s in text.splitlines()):
            return f"{rel}: line already present"
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text + line + f"  # {MANAGED}\n")
        return f"{rel}: added line"
    if not any(is_ours(s) for s in text.splitlines()):
        return f"{rel}: nothing to remove"
    kept = [s for s in text.splitlines() if not is_ours(s)]
    path.write_text("\n".join(kept) + ("\n" if kept else ""))
    return f"{rel}: removed line"


def install(shell: str) -> int:
    rc = 0
    for path, content, executable in _targets(shell):
        what = _write(path, content, executable)
        if what == "refused":
            print(f"  {path}: exists, not installed by csearch — remove it first", file=sys.stderr)
            rc = 1
        else:
            print(f"  {path}: {what}")
    note = _rc_line(shell, add=True)
    if note:
        print(f"  {note}")
    return rc


def uninstall(shell: str) -> int:
    for path, _, _ in _targets(shell):
        if not path.exists():
            continue
        if MANAGED not in path.read_text(errors="replace"):
            print(f"  {path}: kept (not installed by csearch)")
            continue
        path.unlink()
        print(f"  {path}: removed")
    note = _rc_line(shell, add=False)
    if note:
        print(f"  {note}")
    return 0


def default_shell() -> str | None:
    """$SHELL when recognizable, else the passwd login shell."""
    candidates = [os.environ.get("SHELL", "")]
    try:
        import pwd

        candidates.append(pwd.getpwuid(os.getuid()).pw_shell)
    except Exception:
        pass
    for path in candidates:
        name = path.rpartition("/")[2]
        if name in SHELLS:
            return name
    return None
