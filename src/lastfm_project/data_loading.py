"""Validated, deterministic loading of the frozen HetRec Last.fm archive."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable
from zipfile import BadZipFile, ZipFile

import pandas as pd


ARCHIVE_SHA256 = "6738f48195667ff03caaab4d32ca9a3133d8cc026b7c3cdaf6ce1010e913c59c"
REQUIRED_MEMBERS = frozenset({"user_artists.dat", "user_friends.dat", "artists.dat"})
FROZEN_MEMBERS = frozenset(
    {
        "artists.dat",
        "readme.txt",
        "tags.dat",
        "user_artists.dat",
        "user_friends.dat",
        "user_taggedartists.dat",
        "user_taggedartists-timestamps.dat",
    }
)

_INT_RE = re.compile(r"^[+-]?\d+$")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class ArchiveIntegrityError(ValueError):
    """Raised when the archive identity or ZIP integrity is invalid."""


class DataValidationError(ValueError):
    """Raised when a required source-table contract is violated."""


@dataclass(frozen=True)
class FriendshipTables:
    """Validated friendship representations."""

    directed: pd.DataFrame
    undirected: pd.DataFrame


@dataclass(frozen=True)
class HetRecData:
    """All P2.1 tables and deterministic raw-ID mappings."""

    interactions: pd.DataFrame
    friendships: FriendshipTables
    artists: pd.DataFrame
    user_index: pd.DataFrame
    artist_index: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArchiveIntegrityError(f"Cannot read archive {path}: {exc}") from exc
    return digest.hexdigest()


def _open_checked_archive(archive_path: Path, *, verify_checksum: bool) -> ZipFile:
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ArchiveIntegrityError(f"Archive does not exist: {archive_path}")
    if verify_checksum:
        actual = _sha256(archive_path)
        if actual != ARCHIVE_SHA256:
            raise ArchiveIntegrityError(
                f"Archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
            )
    try:
        zf = ZipFile(archive_path)
    except (BadZipFile, OSError) as exc:
        raise ArchiveIntegrityError(f"Invalid ZIP archive {archive_path}: {exc}") from exc
    names = [info.filename for info in zf.infolist()]
    if len(names) != len(set(names)):
        zf.close()
        raise ArchiveIntegrityError("ZIP contains duplicate member names")
    actual_names = frozenset(names)
    if not REQUIRED_MEMBERS <= actual_names:
        missing = sorted(REQUIRED_MEMBERS - actual_names)
        zf.close()
        raise ArchiveIntegrityError(f"ZIP is missing required members: {missing}")
    if actual_names not in (REQUIRED_MEMBERS, FROZEN_MEMBERS):
        unexpected = sorted(actual_names - FROZEN_MEMBERS)
        missing = sorted(FROZEN_MEMBERS - actual_names)
        zf.close()
        raise ArchiveIntegrityError(
            f"Unexpected ZIP member set; missing={missing}, unexpected={unexpected}"
        )
    try:
        bad = zf.testzip()
    except (BadZipFile, OSError, RuntimeError) as exc:
        zf.close()
        raise ArchiveIntegrityError(f"ZIP CRC validation failed: {exc}") from exc
    if bad is not None:
        zf.close()
        raise ArchiveIntegrityError(f"ZIP CRC validation failed for member {bad!r}")
    return zf


def _read_rows(
    zf: ZipFile,
    member: str,
    expected_header: tuple[str, ...],
    *,
    nullable_fields: frozenset[int] = frozenset(),
) -> list[tuple[int, list[str]]]:
    try:
        raw = zf.read(member)
    except (KeyError, BadZipFile) as exc:
        raise ArchiveIntegrityError(f"Cannot read ZIP member {member!r}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{member}: invalid UTF-8 at byte {exc.start}") from exc
    if not text.endswith("\n"):
        raise DataValidationError(f"{member}: missing final newline")
    lines = text.splitlines()
    if not lines:
        raise DataValidationError(f"{member}: empty file")
    header = lines[0].split("\t")
    if tuple(header) != expected_header:
        raise DataValidationError(
            f"{member}: invalid header {header!r}; expected {list(expected_header)!r}"
        )
    rows: list[tuple[int, list[str]]] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != len(expected_header):
            raise DataValidationError(
                f"{member}: row {line_number} has {len(fields)} fields; "
                f"expected {len(expected_header)}"
            )
        if any(field == "" and index not in nullable_fields for index, field in enumerate(fields)):
            raise DataValidationError(f"{member}: row {line_number} contains a null field")
        rows.append((line_number, fields))
    return rows


def _integer(value: str, member: str, line: int, field: str) -> int:
    if not _INT_RE.fullmatch(value):
        raise DataValidationError(f"{member}: row {line} field {field!r} is not an integer: {value!r}")
    parsed = int(value)
    if not _INT64_MIN <= parsed <= _INT64_MAX:
        raise DataValidationError(
            f"{member}: row {line} field {field!r} is outside signed int64 range: {value!r}"
        )
    return parsed


def _frame(rows: Iterable[tuple[int, list[str]]], member: str, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([fields for _, fields in rows], columns=columns)


def _validate_positive_ids(frame: pd.DataFrame, member: str, fields: Iterable[str]) -> None:
    for field in fields:
        if (frame[field] <= 0).any():
            bad = int(frame.loc[frame[field] <= 0, field].iloc[0])
            raise DataValidationError(f"{member}: {field} must be positive; found {bad}")


def load_artists(archive_path: Path, *, verify_checksum: bool = True) -> pd.DataFrame:
    """Load and validate ``artists.dat`` with canonical snake_case columns."""
    zf = _open_checked_archive(Path(archive_path), verify_checksum=verify_checksum)
    try:
        rows = _read_rows(
            zf, "artists.dat", ("id", "name", "url", "pictureURL"), nullable_fields=frozenset({3})
        )
    finally:
        zf.close()
    parsed = []
    for line, fields in rows:
        artist_id = _integer(fields[0], "artists.dat", line, "id")
        if not fields[1] or not fields[2]:
            raise DataValidationError(f"artists.dat: row {line} has an empty name or url")
        parsed.append((artist_id, fields[1], fields[2], fields[3] or pd.NA))
    frame = pd.DataFrame(parsed, columns=["artist_id", "name", "url", "picture_url"])
    frame = frame.astype({"artist_id": "int64", "name": "string", "url": "string", "picture_url": "string"})
    _validate_positive_ids(frame, "artists.dat", ("artist_id",))
    if frame["artist_id"].duplicated().any():
        raise DataValidationError("artists.dat: duplicate artist_id values")
    return frame.sort_values("artist_id", kind="mergesort", ignore_index=True)


def load_user_artists(
    archive_path: Path, *, artist_ids: Iterable[int] | None = None, verify_checksum: bool = True
) -> pd.DataFrame:
    """Load validated listening interactions, sorted by ``user_id, artist_id``."""
    zf = _open_checked_archive(Path(archive_path), verify_checksum=verify_checksum)
    try:
        rows = _read_rows(zf, "user_artists.dat", ("userID", "artistID", "weight"))
    finally:
        zf.close()
    parsed = [tuple(_integer(v, "user_artists.dat", line, field) for v, field in zip(fields, ("userID", "artistID", "weight"))) for line, fields in rows]
    frame = pd.DataFrame(parsed, columns=["user_id", "artist_id", "weight"], dtype="int64")
    _validate_positive_ids(frame, "user_artists.dat", ("user_id", "artist_id", "weight"))
    if frame.duplicated(["user_id", "artist_id"]).any():
        raise DataValidationError("user_artists.dat: duplicate user_id/artist_id pair")
    if artist_ids is not None:
        known = set(int(value) for value in artist_ids)
        missing = sorted(set(frame["artist_id"]) - known)
        if missing:
            raise DataValidationError(f"user_artists.dat: artist foreign keys missing from artists.dat: {missing[:10]}")
    return frame.sort_values(["user_id", "artist_id"], kind="mergesort", ignore_index=True)


def load_user_friends(
    archive_path: Path,
    *,
    interaction_user_ids: Iterable[int] | None = None,
    verify_checksum: bool = True,
) -> FriendshipTables:
    """Load reciprocal directed friendships and canonical undirected edges."""
    zf = _open_checked_archive(Path(archive_path), verify_checksum=verify_checksum)
    try:
        rows = _read_rows(zf, "user_friends.dat", ("userID", "friendID"))
    finally:
        zf.close()
    parsed = [tuple(_integer(v, "user_friends.dat", line, field) for v, field in zip(fields, ("userID", "friendID"))) for line, fields in rows]
    directed = pd.DataFrame(parsed, columns=["user_id", "friend_id"], dtype="int64")
    _validate_positive_ids(directed, "user_friends.dat", ("user_id", "friend_id"))
    if (directed["user_id"] == directed["friend_id"]).any():
        raise DataValidationError("user_friends.dat: self-loop detected")
    if directed.duplicated().any():
        raise DataValidationError("user_friends.dat: duplicate directed friendship")
    if interaction_user_ids is not None:
        known = set(int(value) for value in interaction_user_ids)
        missing = sorted((set(directed["user_id"]) | set(directed["friend_id"])) - known)
        if missing:
            raise DataValidationError(f"user_friends.dat: endpoint users absent from interactions: {missing[:10]}")
    pairs = set(map(tuple, directed[["user_id", "friend_id"]].itertuples(index=False, name=None)))
    missing_reverse = sorted((b, a) for a, b in pairs if (b, a) not in pairs)
    if missing_reverse:
        raise DataValidationError(f"user_friends.dat: missing reciprocal edges, e.g. {missing_reverse[:5]}")
    directed = directed.sort_values(["user_id", "friend_id"], kind="mergesort", ignore_index=True)
    undirected = directed.assign(
        user_low_id=directed[["user_id", "friend_id"]].min(axis=1),
        user_high_id=directed[["user_id", "friend_id"]].max(axis=1),
    )[["user_low_id", "user_high_id"]].drop_duplicates().sort_values(
        ["user_low_id", "user_high_id"], kind="mergesort", ignore_index=True
    )
    undirected = undirected.astype("int64")
    return FriendshipTables(directed=directed, undirected=undirected)


def load_hetrec(archive_path: Path, *, verify_checksum: bool = True) -> HetRecData:
    """Validate and load all P2.1 tables directly from a ZIP archive."""
    archive_path = Path(archive_path)
    artists = load_artists(archive_path, verify_checksum=verify_checksum)
    interactions = load_user_artists(archive_path, artist_ids=artists["artist_id"], verify_checksum=verify_checksum)
    friendships = load_user_friends(
        archive_path, interaction_user_ids=interactions["user_id"], verify_checksum=verify_checksum
    )
    user_index = pd.DataFrame({"user_id": sorted(interactions["user_id"].unique())}, dtype="int64")
    user_index["index"] = pd.Series(range(len(user_index)), dtype="int32")
    artist_index = pd.DataFrame({"artist_id": artists["artist_id"].to_numpy()}, dtype="int64")
    artist_index["index"] = pd.Series(range(len(artist_index)), dtype="int32")
    return HetRecData(interactions, friendships, artists, user_index, artist_index)


__all__ = [
    "ARCHIVE_SHA256",
    "ArchiveIntegrityError",
    "DataValidationError",
    "FriendshipTables",
    "HetRecData",
    "load_artists",
    "load_hetrec",
    "load_user_artists",
    "load_user_friends",
]
