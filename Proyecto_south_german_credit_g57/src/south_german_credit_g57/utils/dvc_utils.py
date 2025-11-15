from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Iterable, Optional


class DVCError(RuntimeError):
    """Excepción específica para errores de ejecución de DVC."""


def _find_dvc_root(start_path: Optional[str] = None) -> str:
    """
    Busca hacia arriba el directorio que contiene ".dvc".

    - start_path: punto de inicio para la búsqueda. Si es None, usa cwd.
    - Retorna el primer directorio (ascendiendo) que contenga ".dvc".
      Si no encuentra, retorna el start_path original.
    """
    cur = os.path.abspath(start_path or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(cur, ".dvc")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # Llegamos a la raíz; devuelve el inicio original
            return os.path.abspath(start_path or os.getcwd())
        cur = parent


@dataclass
class DVCManager:
    """
    Pequeño helper para ejecutar `dvc pull` y `dvc push` desde Python.

    Ejemplos
    --------
    manager = DVCManager()            # autodetecta el root con .dvc
    manager.pull()                    # trae los artefactos
    # ... tu lógica de trabajo ...
    manager.push()                    # sube cambios al remote
    """

    repo_path: Optional[str] = None
    verbose: bool = True

    def __post_init__(self) -> None:
        # Autodetecta el root con .dvc si no se especifica
        if self.repo_path is None:
            self.repo_path = _find_dvc_root()

    def _ensure_dvc(self) -> None:
        if shutil.which("dvc") is None:
            raise DVCError(
                "No se encontró 'dvc' en el PATH. Instala DVC o activa el entorno virtual."
            )

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        self._ensure_dvc()
        cwd = self.repo_path or os.getcwd()
        result = subprocess.run(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        if self.verbose and result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            raise DVCError(f"Fallo al ejecutar: {' '.join(args)}")
        return result

    def pull(
        self,
        remote: Optional[str] = None,
        targets: Optional[Iterable[str]] = None,
        jobs: Optional[int] = None,
        force: bool = False,
    ) -> subprocess.CompletedProcess:
        """Ejecuta `dvc pull` con opciones opcionales."""
        cmd = ["dvc", "pull"]
        if remote:
            cmd += ["-r", remote]
        if jobs:
            cmd += ["-j", str(jobs)]
        if force:
            cmd += ["--force"]
        if targets:
            cmd += list(targets)
        return self._run(cmd)

    def push(
        self,
        remote: Optional[str] = None,
        targets: Optional[Iterable[str]] = None,
        jobs: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Ejecuta `dvc push` con opciones opcionales."""
        cmd = ["dvc", "push"]
        if remote:
            cmd += ["-r", remote]
        if jobs:
            cmd += ["-j", str(jobs)]
        if targets:
            cmd += list(targets)
        return self._run(cmd)


class dvc_session(ContextDecorator):
    """
    Context manager y decorador para hacer `pull` al entrar y `push` al salir.

    Uso como context manager:

        with dvc_session(remote="origin", push_on_success_only=True) as dvc:
            # datos/artefactos disponibles tras pull
            # ... tu código ...
            # en __exit__ hará push (solo si no hubo excepción)

    Uso como decorador:

        @dvc_session()
        def mi_funcion():
            ...
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        remote: Optional[str] = None,
        push_on_success_only: bool = True,
        verbose: bool = True,
        pull_force: bool = False,
    ) -> None:
        self.manager = DVCManager(repo_path=repo_path, verbose=verbose)
        self.remote = remote
        self.push_on_success_only = push_on_success_only
        self.pull_force = pull_force

    # Context manager
    def __enter__(self) -> DVCManager:
        self.manager.pull(remote=self.remote, force=self.pull_force)
        return self.manager

    def __exit__(self, exc_type, exc, tb) -> bool:
        should_push = exc is None or not self.push_on_success_only
        if should_push:
            try:
                self.manager.push(remote=self.remote)
            except DVCError:
                # No suprime excepción ajena; deja que burbujee
                if exc is None:
                    raise
        # No suprime excepciones del bloque 'with'
        return False


if __name__ == "__main__":
    # Ejecución directa de ejemplo: pull y luego push
    try:
        mgr = DVCManager(verbose=True)
        print("[DVC] Ejecutando pull...")
        mgr.pull()
        print("[DVC] Ejecutando push...")
        mgr.push()
    except DVCError as e:
        print(f"[DVC] Error: {e}", file=sys.stderr)
        sys.exit(1)
