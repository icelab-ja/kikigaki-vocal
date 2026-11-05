import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


PACKAGE_FILE_KEYS = ("dtpr", "audio", "audio_vocals", "candidate", "log")


def _package_error(error_type, source_text: str, **values):
    error = error_type(source_text.format(**values))
    error.translation_source = source_text
    error.translation_values = values
    return error


def package_error_message(error: Exception, translate) -> str:
    source_text = getattr(error, "translation_source", None)
    values = getattr(error, "translation_values", {})
    if not source_text:
        return str(error)
    try:
        return translate(source_text).format(**values)
    except (KeyError, ValueError):
        return str(error)


@dataclass(frozen=True)
class ExtractedKkgkPackage:
    pid: dict
    pid_member: str
    members: dict[str, str]
    paths: dict[str, str]


class KkgkPackageSession:
    def __init__(self):
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self.package_path: str | None = None
        self.midi_member: str | None = None
        self.dtpr_member: str | None = None
        self.log_member: str | None = None

    def prepare_load(
        self,
        package_path: str | Path,
    ) -> tuple[tempfile.TemporaryDirectory, ExtractedKkgkPackage]:
        temp_dir = tempfile.TemporaryDirectory(prefix="kikigaki_zip_")
        try:
            package = extract_kkgk_package(package_path, temp_dir.name)
        except Exception:
            temp_dir.cleanup()
            raise
        return temp_dir, package

    def commit_load(
        self,
        package_path: str | Path,
        temp_dir: tempfile.TemporaryDirectory,
        members: dict[str, str],
    ) -> None:
        old_tempdir = self._tempdir
        self._tempdir = temp_dir
        self.package_path = str(package_path)
        self.midi_member = members.get("midi")
        self.dtpr_member = members.get("dtpr")
        self.log_member = members.get("log")
        if old_tempdir is not None and old_tempdir is not temp_dir:
            try:
                old_tempdir.cleanup()
            except Exception:
                pass

    @staticmethod
    def discard_load(temp_dir: tempfile.TemporaryDirectory | None) -> None:
        if temp_dir is not None:
            temp_dir.cleanup()

    def clear_metadata(self) -> None:
        self.package_path = None
        self.midi_member = None
        self.dtpr_member = None
        self.log_member = None

    def snapshot_metadata(self) -> dict:
        return {
            "loaded_zip_path": self.package_path,
            "loaded_zip_midi_member": self.midi_member,
            "loaded_zip_dtpr_member": self.dtpr_member,
            "loaded_zip_log_member": self.log_member,
        }

    def restore_metadata(self, state: dict) -> None:
        self.package_path = state.get("loaded_zip_path")
        self.midi_member = state.get("loaded_zip_midi_member")
        self.dtpr_member = state.get("loaded_zip_dtpr_member")
        self.log_member = state.get("loaded_zip_log_member")

    def can_export(self) -> bool:
        return bool(self.package_path and self.dtpr_member)

    def _member_parent_and_stem(
        self,
        member: str | None = None,
        fallback_paths=(),
    ) -> tuple[str, str]:
        selected = member or self.midi_member or self.dtpr_member
        if not selected:
            selected = next((path for path in fallback_paths if path), "output.mid")

        member_path = Path(str(selected).replace("\\", "/"))
        parent = "" if str(member_path.parent) == "." else str(member_path.parent).replace("\\", "/")
        stem = member_path.stem
        if stem.endswith("_ipr"):
            stem = stem[:-4]
        if stem.endswith("_dtpr"):
            stem = stem[:-5]
        return parent, stem

    def edit_log_path(
        self,
        *,
        package_path: str | Path | None = None,
        member: str | None = None,
        log_member: str | None = None,
        fallback_paths=(),
    ) -> Path:
        source_path = package_path or self.package_path
        if source_path is None:
            raise ValueError("KKGK package path is not set.")
        selected_log_member = log_member or self.log_member
        if selected_log_member:
            log_name = Path(str(selected_log_member).replace("\\", "/")).name
            if log_name:
                return Path(source_path).parent / log_name
        _, stem = self._member_parent_and_stem(member, fallback_paths)
        return Path(source_path).parent / f"{stem}.edit_log.jsonl"

    def output_dtpr_member(self, *fallback_paths) -> str:
        if self.dtpr_member:
            return str(self.dtpr_member).replace("\\", "/")
        parent, stem = self._member_parent_and_stem(fallback_paths=fallback_paths)
        name = f"{stem}_dtpr.txt"
        return f"{parent}/{name}" if parent else name

    def output_log_member(self, process_id: str | None = None, *fallback_paths) -> str:
        if self.log_member:
            return str(self.log_member).replace("\\", "/")
        parent, stem = self._member_parent_and_stem(fallback_paths=fallback_paths)
        name_stem = str(process_id).strip() if process_id else stem
        name = f"{name_stem}.edit_log.jsonl"
        return f"{parent}/{name}" if parent else name


def _normalize_pid_file_ref(value, key: str) -> str:
    if value is None:
        raise ValueError(f"{key} is null.")
    if not isinstance(value, str):
        raise _package_error(
            ValueError,
            "files.{key} in pid.json must be a string or null.",
            key=key,
        )
    name = value.strip()
    if not name:
        raise _package_error(
            ValueError,
            "files.{key} in pid.json is empty.",
            key=key,
        )
    normalized = name.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise _package_error(
            ValueError,
            "files.{key} in pid.json contains an unsafe path: {value}",
            key=key,
            value=value,
        )
    return normalized


def _find_zip_member(zf: zipfile.ZipFile, filename: str, *, base_dir: str = "") -> str:
    names = [name for name in zf.namelist() if not name.endswith("/")]
    normalized = filename.replace("\\", "/").lstrip("/")

    candidates = []
    if base_dir:
        candidates.append(f"{base_dir.rstrip('/')}/{normalized}")
    candidates.append(normalized)

    for candidate in candidates:
        if candidate in names:
            return candidate

    basename_matches = [name for name in names if Path(name).name == Path(normalized).name]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise _package_error(
            FileNotFoundError,
            "The package contains multiple files named {filename}: {matches}",
            filename=filename,
            matches=", ".join(basename_matches),
        )
    raise _package_error(
        FileNotFoundError,
        "The package does not contain {filename}.",
        filename=filename,
    )


def _load_pid_json(zf: zipfile.ZipFile) -> tuple[dict, str]:
    pid_member = _find_zip_member(zf, "pid.json")
    try:
        raw = zf.read(pid_member).decode("utf-8-sig")
        pid = json.loads(raw)
    except Exception as exc:
        raise _package_error(
            ValueError,
            "Could not read pid.json: {error}",
            error=exc,
        ) from exc
    if not isinstance(pid, dict):
        raise _package_error(
            ValueError,
            "The top level of pid.json must be an object.",
        )
    if not isinstance(pid.get("files"), dict):
        raise _package_error(
            ValueError,
            "pid.json does not contain a files object.",
        )
    _validate_pid_fun_id(pid)
    _validate_pid_log_ref(pid)
    return pid, pid_member


def _validate_pid_fun_id(pid: dict) -> None:
    if "funId" not in pid:
        return
    fun_id = pid["funId"]
    if not isinstance(fun_id, str) or not fun_id.strip():
        raise _package_error(
            ValueError,
            "funId in pid.json must be a non-empty string.",
        )


def _validate_pid_log_ref(pid: dict) -> None:
    process_id = str(pid.get("processId") or "").strip()
    log_value = (pid.get("files") or {}).get("log")
    if not process_id or log_value is None:
        return
    log_ref = _normalize_pid_file_ref(log_value, "log")
    expected = f"{process_id}.edit_log.jsonl"
    if Path(log_ref).name != expected:
        raise _package_error(
            ValueError,
            "files.log in pid.json must be {expected} to match processId: {value}",
            expected=expected,
            value=log_value,
        )


def _pid_base_dir(pid_member: str) -> str:
    base_dir = str(Path(pid_member).parent).replace("\\", "/")
    return "" if base_dir == "." else base_dir


def _extract_member(zf: zipfile.ZipFile, member: str, destination: Path, key: str) -> str:
    suffix = Path(member).suffix
    safe_name = f"{key}{suffix}" if suffix else key
    out_path = destination / safe_name
    with zf.open(member, "r") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(out_path)


def extract_kkgk_package(package_path: str | Path, destination: str | Path) -> ExtractedKkgkPackage:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(package_path, "r") as zf:
        pid, pid_member = _load_pid_json(zf)
        base_dir = _pid_base_dir(pid_member)
        files = pid["files"]

        members: dict[str, str] = {}
        paths: dict[str, str] = {}
        for key in PACKAGE_FILE_KEYS:
            value = files.get(key)
            if value is None:
                continue
            filename = _normalize_pid_file_ref(value, key)
            member = _find_zip_member(zf, filename, base_dir=base_dir)
            members[key] = member
            paths[key] = _extract_member(zf, member, destination, key)

    if "dtpr" not in paths:
        raise _package_error(
            ValueError,
            "files.dtpr is required in pid.json.",
        )

    return ExtractedKkgkPackage(
        pid=pid,
        pid_member=pid_member,
        members=members,
        paths=paths,
    )


def _pid_for_export(source_pid: dict, dtpr_member: str, log_member: str | None = None) -> dict:
    files = dict((source_pid or {}).get("files") or {})
    output_files = {
        "dtpr": dtpr_member,
        "audio": files.get("audio"),
        "audio_vocals": files.get("audio_vocals"),
        "candidate": files.get("candidate"),
    }
    if log_member is not None:
        output_files["log"] = log_member
    elif files.get("log") is not None:
        output_files["log"] = files.get("log")

    output_pid = {
        "processId": (source_pid or {}).get("processId", ""),
    }
    if "funId" in (source_pid or {}):
        output_pid["funId"] = source_pid.get("funId")
    output_pid["files"] = output_files
    return output_pid


def _copy_pid_member(
    src: zipfile.ZipFile,
    dst: zipfile.ZipFile,
    *,
    key: str,
    value,
    base_dir: str,
) -> None:
    if value is None:
        return
    filename = _normalize_pid_file_ref(value, key)
    member = _find_zip_member(src, filename, base_dir=base_dir)
    dst.writestr(member, src.read(member))


def _write_package(
    source_path: Path,
    output_path: Path,
    dtpr_path: Path,
    dtpr_member: str,
    log_path: Path | None = None,
    log_member: str | None = None,
) -> None:
    with zipfile.ZipFile(source_path, "r") as src, zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as dst:
        source_pid, pid_member = _load_pid_json(src)
        base_dir = _pid_base_dir(pid_member)
        files = dict((source_pid or {}).get("files") or {})

        for key in ("audio", "audio_vocals", "candidate"):
            _copy_pid_member(src, dst, key=key, value=files.get(key), base_dir=base_dir)

        dst.write(dtpr_path, dtpr_member)
        if log_path is not None and log_member is not None:
            dst.write(log_path, log_member)
        elif files.get("log") is not None:
            _copy_pid_member(src, dst, key="log", value=files.get("log"), base_dir=base_dir)
        dst.writestr(
            pid_member,
            json.dumps(_pid_for_export(source_pid, dtpr_member, log_member), ensure_ascii=False, indent=2) + "\n",
        )


def write_kkgk_package(
    source_path: str | Path,
    output_path: str | Path,
    dtpr_path: str | Path,
    dtpr_member: str,
    log_path: str | Path | None = None,
    log_member: str | None = None,
) -> None:
    source_path = Path(source_path)
    output_path = Path(output_path)
    dtpr_path = Path(dtpr_path)
    log_path = Path(log_path) if log_path is not None else None
    same_output = output_path.resolve() == source_path.resolve()
    temporary_output: Path | None = None

    try:
        if same_output:
            temp_file = tempfile.NamedTemporaryFile(
                prefix=f".{output_path.stem}_",
                suffix=".kkgk",
                dir=str(output_path.parent),
                delete=False,
            )
            temp_file.close()
            write_path = Path(temp_file.name)
            temporary_output = write_path
        else:
            write_path = output_path

        _write_package(source_path, write_path, dtpr_path, dtpr_member, log_path, log_member)

        if same_output:
            write_path.replace(output_path)
            temporary_output = None
    finally:
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()
